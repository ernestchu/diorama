"""Make a trained network safe for Core ML's fp16 (the Neural Engine computes in fp16, max 65504).

FastViT-MA36's last stage carries 'massive activation' channels: per-channel near-constant offsets of up to ~24000
in its residual stream, about 1900x their variation. In fp16 the stage's BatchNorms amplify the rounding of those
channels, and our pyramid turns the offsets into ~3e5 at the lateral and ~2e7 at the smoothing conv (overflow). Two
changes, exact in fp32 up to rounding (make_fp16_safe), so the whole network runs in fp16 on the Neural Engine:
  1. last stage: the residual stream carries only its deviation from per-position means K (calibration frames); the
     offsets go into per-position bias maps of the BatchNorms, the depthwise convs and the branch outputs, V's mean
     is removed before attention (softmax rows sum to 1), and the layer scales are folded into proj and fc2;
  2. pyramid: remove each lateral's per-channel mean (over calibration frames), scale the laterals by A, and fold
     the smoothing conv's BatchNorm, the scale, and the removed constant (with its zero-padding border) into the
     conv weights and a constant per-pixel bias map (Smooth16).
"""
import glob
import numpy as np, cv2, torch
import torch.nn.functional as F

A = 1 / 4096


class Smooth16(torch.nn.Module):
  def __init__(self, W, B):
    super().__init__()
    self.conv = torch.nn.Conv2d(W.shape[1], W.shape[0], 3, padding=1, bias=False)
    self.conv.weight.data = W
    self.register_buffer('B', B)

  def forward(self, x):
    return torch.relu(self.conv(x) + self.B)


def calib_frames(n=64, d=None):
  import json, os
  from paths import C1M
  d = d or f'{C1M}/yuv'
  train = {k for k, v in json.load(open(os.path.join(os.path.dirname(d), 'items.json'))).items() if v['split'] == 'train'}
  X = []
  for f in [f for f in sorted(glob.glob(f'{d}/*.png')) if os.path.basename(f)[:-4] in train][5::23][:n]:
    img = cv2.imread(f)
    y = torch.from_numpy(cv2.resize(img[:, :, 0], (512, 256), interpolation=cv2.INTER_AREA).astype(np.float32))[None, None]
    uv = torch.from_numpy(cv2.resize(img[:, :, 1:], (256, 128), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
                          .astype(np.float32))[None]
    X.append((y, uv))
  return X


def fold_pyramid(m, calib, a=A):
  outs, shapes = {i: [] for i in range(len(m.lat))}, {}
  hs = [l.register_forward_hook((lambda i: (lambda mod, inp, o: outs[i].append(o.detach())))(i)) for i, l in enumerate(m.lat)]
  hs.append(m.smooth.register_forward_hook(lambda mod, inp, o: shapes.__setitem__('p', inp[0].shape)))
  with torch.no_grad():
    for y, uv in calib:
      m(y, uv)
  for h in hs:
    h.remove()
  with torch.no_grad():
    mus = [torch.cat(outs[i]).mean((0, 2, 3)) for i in range(len(m.lat))]
    k = sum(mus)
    for l, mu in zip(m.lat, mus):
      if isinstance(l, AddMap):
        l.B.sub_(mu.view(1, -1, 1, 1)); l.conv.weight.mul_(a); l.B.mul_(a)
      else:
        l.bias.sub_(mu); l.weight.mul_(a); l.bias.mul_(a)
    c, bn = m.smooth[0], m.smooth[1]
    f = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    _, C, H, W = shapes['p']
    cmap = F.conv2d(k.view(1, -1, 1, 1).expand(1, C, H, W), c.weight, None, padding=1)
    B = f.view(1, -1, 1, 1) * (c.bias.view(1, -1, 1, 1) + cmap - bn.running_mean.view(1, -1, 1, 1)) + bn.bias.view(1, -1, 1, 1)
    m.smooth = Smooth16((c.weight * f.view(-1, 1, 1, 1) / a).clone(), B.clone())
  return m


def fp32_op(op):
  """Core ML op selector: True keeps the op in fp16. False for the last stage (stride 32: 8x16 maps, its 128
  attention tokens) and its lateral."""
  for o in op.outputs:
    s = tuple(getattr(o, 'shape', ()) or ())
    if len(s) >= 2 and s[-2:] == (8, 16):
      return False
    if 128 in s and not {64, 256, 512} & set(s):  # attention tokens (8 x 16 = 128); not the pyramid or the inputs
      return False
  return True


# ---- the last stage: remove the residual stream's offsets (exact in fp32) ----

class AddMap(torch.nn.Module):
  def __init__(self, conv, bmap):
    """conv without its bias, then a per-position bias map (1, C, H, W)."""
    super().__init__()
    self.conv = conv
    self.conv.bias = None
    self.register_buffer('B', bmap)

  def forward(self, x):
    return self.conv(x) + self.B


class Block16(torch.nn.Module):
  """An AttentionBlock on x' = x - K_in, giving x_out - K_out, with the offsets folded into per-position maps."""
  def __init__(self, blk, K_in, K_mid, K_out):
    super().__init__()
    with torch.no_grad():
      bn = blk.norm
      a = bn.weight / torch.sqrt(bn.running_var + bn.eps)
      self.na = torch.nn.Parameter(a.view(1, -1, 1, 1).clone(), requires_grad=False)
      self.register_buffer('nmap', a.view(1, -1, 1, 1) * (K_in - bn.running_mean.view(1, -1, 1, 1)) + bn.bias.view(1, -1, 1, 1))
      self.attn = blk.token_mixer
      ls1 = blk.layer_scale_1.gamma.view(1, -1, 1, 1)
      self.ls1 = torch.nn.Parameter(ls1.clone(), requires_grad=False)
      pb = self.attn.proj.bias.view(1, -1, 1, 1) - (K_mid - K_in) / ls1          # proj bias as a map
      self.attn.proj.bias = None
      self.register_buffer('pmap', pb)
      c, cbn = blk.mlp.conv.conv, blk.mlp.conv.bn
      s = cbn.weight / torch.sqrt(cbn.running_var + cbn.eps)
      dk = F.conv2d(K_mid, c.weight, None, padding=c.padding, groups=c.groups)    # depthwise conv of the offsets
      self.dw = torch.nn.Conv2d(c.in_channels, c.out_channels, c.kernel_size, padding=c.padding, groups=c.groups, bias=False)
      self.dw.weight.data = c.weight * s.view(-1, 1, 1, 1)
      self.register_buffer('dmap', s.view(1, -1, 1, 1) * (dk - cbn.running_mean.view(1, -1, 1, 1)) + cbn.bias.view(1, -1, 1, 1))
      self.fc1, self.act, self.fc2 = blk.mlp.fc1, blk.mlp.act, blk.mlp.fc2
      ls2 = blk.layer_scale_2.gamma.view(1, -1, 1, 1)
      self.ls2 = torch.nn.Parameter(ls2.clone(), requires_grad=False)
      fb = self.fc2.bias.view(1, -1, 1, 1) - (K_out - K_mid) / ls2
      self.fc2.bias = None
      self.register_buffer('fmap', fb)

  def forward(self, x):
    y = x * self.na + self.nmap
    x = x + self.ls1 * (self.attn(y) + self.pmap)
    h = self.dw(x) + self.dmap
    x = x + self.ls2 * (self.fc2(self.act(self.fc1(h))) + self.fmap)
    return x


class Stage16(torch.nn.Module):
  def __init__(self, stage, Ks):
    super().__init__()
    self.downsample = stage.downsample
    self.pos = AddMap(stage.pos_emb.reparam_conv, stage.pos_emb.reparam_conv.bias.view(1, -1, 1, 1) - Ks[0])
    self.blocks = torch.nn.Sequential(*[Block16(b, Ks[2 * i], Ks[2 * i + 1], Ks[2 * i + 2]) for i, b in enumerate(stage.blocks)])
    self.register_buffer('K_out', Ks[-1])

  def forward(self, x):
    return self.blocks(self.pos(self.downsample(x)))   # = stage output - K_out


def stream_means(m, calib):
  """Per-position means over the calibration frames of the last stage's residual stream: after the position
  encoding, after each block's attention half and after each block."""
  st = m.c5
  acc = []
  def run(x):
    out = []
    x = st.pos_emb(st.downsample(x)); out.append(x)
    for b in st.blocks:
      x = x + b.layer_scale_1(b.token_mixer(b.norm(x))); out.append(x)
      x = x + b.layer_scale_2(b.mlp(x)); out.append(x)
    return out
  h = m.c4.register_forward_hook(lambda mod, inp, o: acc.append(run(o)))
  with torch.no_grad():
    for y, uv in calib:
      m(y, uv)
  h.remove()
  return [torch.cat([a[i] for a in acc]).mean(0, keepdim=True) for i in range(len(acc[0]))]


def fold_last_stage(m, calib):
  """Replace c5 with Stage16 and give lat[-1] the removed offset back as a bias map (exact in fp32)."""
  Ks = stream_means(m, calib)
  with torch.no_grad():
    l = m.lat[-1]
    lmap = l.bias.view(1, -1, 1, 1) + F.conv2d(Ks[-1], l.weight)
    m.c5 = Stage16(m.c5, Ks)
    m.lat[-1] = AddMap(l, lmap)
  return m


def center_values(m, calib):
  """Attention averages its values (softmax rows sum to 1), so a constant part of V passes through unchanged and the
  projection turns it into large outputs. Give qkv a bias that removes V's mean and add W_proj @ mean to the map."""
  blocks = list(m.c5.blocks)
  acc = {i: [] for i in range(len(blocks))}
  hs = [b.attn.qkv.register_forward_hook((lambda i: (lambda mod, inp, o: acc[i].append(o.detach().reshape(-1, o.shape[-1]))))(i))
        for i, b in enumerate(blocks)]
  with torch.no_grad():
    for y, uv in calib:
      m(y, uv)
  for h in hs:
    h.remove()
  with torch.no_grad():
    for i, b in enumerate(blocks):
      q = b.attn.qkv
      C = q.out_features // 3
      vm = torch.cat(acc[i])[:, 2 * C:].mean(0)
      bias = torch.zeros(q.out_features); bias[2 * C:] = -vm
      q.bias = torch.nn.Parameter(bias, requires_grad=False)
      b.pmap.add_((b.attn.proj.weight @ vm).view(1, -1, 1, 1))
  return m


def make_fp16_safe(m, calib=None):
  """FastViT: the whole network computes in fp16 within range; exact in fp32 up to rounding."""
  calib = calib or calib_frames()
  if any(type(x).__name__ == 'QKNormAttention' for x in m.modules()):
    m = rescale_qk(m, calib)
  if getattr(m, 'dfine', False):  # D-FINE neck, trained with the activation penalty: no offsets to fold, only fusing
    m.neck.deploy()
    return fuse_conv_norms(m)[0]
  m = fold_pyramid(fold_layer_scales(center_values(fold_last_stage(m, calib), calib)), calib)
  return fuse_conv_norms(m)[0]


def fold_layer_scales(m):
  """ls * (W a + map) -> (ls W) a + ls map: the branch's large pre-scale outputs never form."""
  with torch.no_grad():
    for b in m.c5.blocks:
      l1, l2 = b.ls1.view(-1), b.ls2.view(-1)
      b.attn.proj.weight.mul_(l1.view(-1, 1)); b.pmap.mul_(b.ls1); b.ls1.fill_(1.0)
      b.fc2.weight.mul_(l2.view(-1, 1, 1, 1)); b.fmap.mul_(b.ls2); b.ls2.fill_(1.0)
  return m


def fuse_conv_norms(m):
  """Fold every remaining conv -> BatchNorm pair (timm ConvNormAct) into the conv, in fp32, so no BatchNorm statistic
  (one running_var is ~1e5, above fp16's 65504) is ever stored or computed in fp16."""
  n = 0
  with torch.no_grad():
    for name, mod in m.named_modules():
      if hasattr(mod, 'conv') and hasattr(mod, 'bn') and isinstance(mod.conv, torch.nn.Conv2d) and isinstance(mod.bn, torch.nn.BatchNorm2d):
        c, bn = mod.conv, mod.bn
        s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        b = (c.bias if c.bias is not None else torch.zeros_like(bn.running_mean))
        c.weight.mul_(s.view(-1, 1, 1, 1))
        c.bias = torch.nn.Parameter((b - bn.running_mean) * s + bn.bias, requires_grad=False)
        act = getattr(bn, 'act', torch.nn.Identity())
        mod.bn = act if isinstance(act, torch.nn.Module) else torch.nn.Identity()
        n += 1
  return m, n


def strict_fp16(m):
  """A stricter stand-in for Core ML fp16 than .half() on a GPU: every parameter and buffer rounded to fp16, and every
  module's output rounded to fp16, with a record of anything at or above 65504 (inf in fp16). Returns hooks' log."""
  import copy
  m = copy.deepcopy(m).float()
  with torch.no_grad():
    for t in list(m.parameters()) + list(m.buffers()):
      if t.is_floating_point():
        t.copy_(t.half().float())
  log = {}
  def hook(name):
    def f(mod, inp, o):
      if isinstance(o, torch.Tensor) and o.is_floating_point():
        big = o.abs().max().item()
        if big >= 65504 or not np.isfinite(big):
          log[name] = max(log.get(name, 0), big if np.isfinite(big) else float('inf'))
        return o.half().float()
    return f
  import types
  from model import QKNormAttention
  for n_, mod in m.named_modules():
    if n_:
      mod.register_forward_hook(hook(n_))
    if isinstance(mod, QKNormAttention):
      mod.forward = types.MethodType(strict_attention_forward(log, n_), mod)
  return m, log


def attention_logits(m, frames):
  """Per last-stage block: max |q k^T * scale| over the frames, raw and with k centred over the tokens (exact for
  softmax), and the median gap between each row's top two logits (how hard the softmax is)."""
  import types
  stats = {}
  def patched(i):
    def fwd(self, x):
      B, C, H, W = x.shape; N = H * W
      x = x.flatten(2).transpose(-2, -1)
      qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
      q, k, v = qkv.unbind(0)
      z = (q * self.scale) @ k.transpose(-2, -1)
      zc = (q * self.scale) @ (k - k.mean(2, keepdim=True)).transpose(-2, -1)
      top2 = z.topk(2, -1).values
      s = stats.setdefault(i, {'raw': 0, 'centred': 0, 'gaps': []})
      s['raw'] = max(s['raw'], z.abs().max().item()); s['centred'] = max(s['centred'], zc.abs().max().item())
      s['gaps'].append((top2[..., 0] - top2[..., 1]).flatten())
      a = z.softmax(-1)
      x = (a @ v).transpose(1, 2).reshape(B, N, C)
      x = self.proj(x)
      return x.transpose(-2, -1).reshape(B, C, H, W)
    return fwd
  blocks = list(m.c5.blocks)
  for i, b in enumerate(blocks):
    att = b.attn if hasattr(b, 'attn') else b.token_mixer
    att.forward = types.MethodType(patched(i), att)
  with torch.no_grad():
    for y, uv in frames:
      m(y, uv)
  return {i: (s['raw'], s['centred'], torch.cat(s['gaps']).median().item(), (torch.cat(s['gaps']) < 1).float().mean().item()) for i, s in stats.items()}


def rescale_qk(m, calib, target=8.0):
  """QK-normalized attention is invariant to the scale of q and k, but |q| = sqrt(sum q^2) can overflow fp16 when q is
  large. Scale the q and k rows of qkv (exact) so their largest entry on the calibration frames is about `target`."""
  from model import QKNormAttention
  atts = [mod for mod in m.modules() if isinstance(mod, QKNormAttention)]
  mx = {id(a): [0.0, 0.0] for a in atts}
  def hook(a):
    def f(mod, inp, o):
      C = o.shape[-1] // 3
      mx[id(a)][0] = max(mx[id(a)][0], o[..., :C].abs().max().item())
      mx[id(a)][1] = max(mx[id(a)][1], o[..., C:2 * C].abs().max().item())
    return f
  hs = [a.qkv.register_forward_hook(hook(a)) for a in atts]
  with torch.no_grad():
    for y, uv in calib:
      m(y, uv)
  for h in hs:
    h.remove()
  with torch.no_grad():
    for a in atts:
      C = a.qkv.out_features // 3
      for j, part in enumerate((slice(0, C), slice(C, 2 * C))):
        s = target / max(mx[id(a)][j], 1e-6)
        a.qkv.weight[part] *= s
        if a.qkv.bias is not None:
          a.qkv.bias[part] *= s
  return m


def strict_attention_forward(log, name):
  """QKNormAttention.forward with every intermediate rounded to fp16 and overflow recorded (the module-output hooks
  of strict_fp16 cannot see inside the attention)."""
  def r(t, what):
    big = t.abs().max().item()
    if not np.isfinite(big) or big >= 65504:
      log[f'{name}.{what}'] = big
    return t.half().float()
  def fwd(self, x):
    from model import TAU_MAX
    B, C, H, W = x.shape
    N = H * W
    x = x.flatten(2).transpose(-2, -1)
    qkv = r(self.qkv(x), 'qkv').reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q = r(q / r(torch.sqrt(r((q * q).sum(-1, keepdim=True), 'q_sumsq')), 'q_norm').clamp(min=1e-6), 'q_hat')
    k = r(k / r(torch.sqrt(r((k * k).sum(-1, keepdim=True), 'k_sumsq')), 'k_norm').clamp(min=1e-6), 'k_hat')
    tau = self.log_tau.exp().clamp(max=TAU_MAX)
    z = r((q * tau) @ k.transpose(-2, -1), 'logits')
    a = r(z.softmax(-1), 'softmax')
    x = r(a @ v, 'av').transpose(1, 2).reshape(B, N, C)
    return r(self.proj(x), 'proj').transpose(-2, -1).reshape(B, C, H, W)
  return fwd
