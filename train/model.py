"""The stop-line network: a 512x256 frame -> the stop line (two points).

Input is the frame as the comma camera encodes it: Y at 512x256 and, with color=True, U and V at 256x128
(YUV420). U and V join after the first stride-2 convolution, which already works at their resolution.

A trunk (ResNet-18/34 from torchvision, or a FastViT from timm), a neck down to stride 4 (128x64), and one
CenterNet-style head:
  line: heat (1), sub-cell centre offset (2), end points from the centre (4), line mask (1, training aid)
The neck is a small FPN, or D-FINE's hybrid encoder (neck.py, neck='dfine'). The published model since v5 is
FastViT-MA36 with QK-normalized attention and the D-FINE neck; every op converts to Core ML and runs in fp16 on the
Neural Engine (fp16safe.py, export.py).
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
import torchvision

STRIDE = 4
END_SCALE = 32.0  # end-point offsets are predicted in cells / END_SCALE
TAU_MAX = 30.0    # QK-norm: largest attention temperature, so the logits stay far inside fp16 range


def head(c, out, bias=0.0):
  h = nn.Sequential(nn.Conv2d(c, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, out, 1))
  nn.init.constant_(h[-1].bias, bias)
  return h


class Net(nn.Module):
  def __init__(self, pretrained=True, width=96, color=False, arch='resnet18', qknorm=False, drop_path=0.0,
               neck='fpn'):
    super().__init__()
    self.vit = arch.startswith('fastvit')
    if self.vit:  # timm FastViT (Apple ml-fastvit weights): stem block 0 (stride 2) replaces conv1
      import timm
      m = timm.create_model(arch, pretrained=pretrained, drop_path_rate=drop_path)
      b = m.stem[0]
      for c in (b.conv_kxk[0].conv, b.conv_scale.conv):  # grey: the RGB filters summed
        c.weight = nn.Parameter(c.weight.data.sum(1, keepdim=True)); c.in_channels = 1
      self.conv1, self.stem = b, m.stem[1:]
      self.c2, self.c3, self.c4, self.c5 = m.stages
      chans = [f['num_chs'] for f in m.feature_info if f['module'].startswith('stages')][-4:]
      if qknorm:  # bounded attention logits, for fp16 on the Neural Engine
        for st in m.stages:
          for blk in st.blocks:
            if hasattr(blk, 'token_mixer') and type(blk.token_mixer).__name__ == 'Attention':
              blk.token_mixer = QKNormAttention(blk.token_mixer)
      c1 = b.conv_kxk[0].conv.out_channels
    else:
      r = getattr(torchvision.models, arch)(weights='IMAGENET1K_V1' if pretrained else None)
      conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
      conv1.weight.data = r.conv1.weight.data.sum(1, keepdim=True)  # grey: the RGB filters summed
      self.conv1, self.stem = conv1, nn.Sequential(r.bn1, r.relu, r.maxpool)
      self.c2, self.c3, self.c4, self.c5 = r.layer1, r.layer2, r.layer3, r.layer4
      chans, c1 = (64, 128, 256, 512), 64
    self.dfine = neck == 'dfine'
    if self.dfine:  # D-FINE's hybrid encoder (neck.py), down to stride 4
      from neck import DFineNeck
      top = STRIDE * 2 ** (len(chans) - 1)
      self.neck = DFineNeck(chans, top_hw=(256 // top, 512 // top))
      width = self.neck.out_channels
    else:
      self.lat = nn.ModuleList(nn.Conv2d(c, width, 1) for c in chans)
      self.smooth = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
    self.line = head(width, 8)
    with torch.no_grad():
      self.line[-1].bias[0] = -4.6  # heat starts near 0.01
      self.line[-1].bias[7] = -4.6
    self.uv = None  # made last, so grey and colour nets with one seed start from the same weights
    if color:
      self.uv = nn.Conv2d(2, c1, 3, padding=1, bias=False)
      nn.init.zeros_(self.uv.weight)  # starts as the grey network; colour is learnt on top

  def forward(self, y, uv=None):
    """y: (N, 1, 256, 512) luma 0..255; uv: (N, 2, 128, 256) chroma 0..255 (used when color=True).
    Returns the raw line maps (N, 8, 64, 128)."""
    x = self.conv1((y / 255.0 - 0.449) / 0.226 if self.vit else (y - 110.0) / 60.0)
    if self.uv is not None:
      x = x + self.uv((uv - 128.0) / 16.0)
    c2 = self.c2(self.stem(x))
    c3 = self.c3(c2)
    c4 = self.c4(c3)
    cs = [c2, c3, c4, self.c5(c4)]
    if self.dfine:
      return self.line(self.neck(cs))
    p = self.lat[-1](cs[-1])
    for lat, c in zip(reversed(self.lat[:-1]), reversed(cs[:-1])):
      p = F.interpolate(p, scale_factor=2.0, mode='nearest') + lat(c)
    return self.line(self.smooth(p))


class QKNormAttention(nn.Module):
  """timm FastViT Attention with q and k normalized per head and a learned per-head temperature: the logits are
  cos(q, k) * tau with tau <= TAU_MAX, so they stay far inside fp16 range (the plain attention reached ~1e5)."""
  def __init__(self, att, tau0=10.0):
    super().__init__()
    self.num_heads, self.head_dim = att.num_heads, att.head_dim
    self.qkv, self.proj = att.qkv, att.proj
    self.log_tau = nn.Parameter(torch.full((att.num_heads, 1, 1), math.log(tau0)))

  def forward(self, x):
    B, C, H, W = x.shape
    N = H * W
    x = x.flatten(2).transpose(-2, -1)
    q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    tau = self.log_tau.exp().clamp(max=TAU_MAX)
    a = ((q * tau) @ k.transpose(-2, -1)).softmax(-1)
    x = (a @ v).transpose(1, 2).reshape(B, N, C)
    return self.proj(x).transpose(-2, -1).reshape(B, C, H, W)


def decode(line, line_thresh=0.3, max_lines=4):
  """Raw maps -> per image {'lines': [[x1, y1, x2, y2, score]]} in 512x256 pixels."""
  out = []
  lh = torch.sigmoid(line[:, 0:1])
  peaks = lh * (F.max_pool2d(lh, 5, 1, 2) == lh)
  for n in range(line.shape[0]):
    lines = []
    flat = peaks[n, 0].flatten()
    scores, idx = flat.topk(max_lines)
    for s, i in zip(scores.tolist(), idx.tolist()):
      if s < line_thresh: break
      y, x = divmod(i, line.shape[3])
      cx, cy = x + line[n, 1, y, x].item(), y + line[n, 2, y, x].item()
      e = line[n, 3:7, y, x] * END_SCALE
      lines.append([(cx + e[0].item()) * STRIDE, (cy + e[1].item()) * STRIDE,
                    (cx + e[2].item()) * STRIDE, (cy + e[3].item()) * STRIDE, s])
    out.append({'lines': lines})
  return out


def arch_of(sd):
  """The trunk a checkpoint was trained with: ResNet-34 has a third block in layer1; FastViT has stages of blocks."""
  if any('.blocks.' in k for k in sd):  # FastViT: the variant whose trunk has exactly these parameter shapes
    import timm
    have = {k.replace('module.', ''): tuple(v.shape) for k, v in sd.items()
            if k.replace('module.', '')[:2] in ('c2', 'c3', 'c4', 'c5') and not k.endswith('log_tau')}
    for a in ('fastvit_ma36', 'fastvit_t8', 'fastvit_t12', 'fastvit_s12', 'fastvit_sa12', 'fastvit_sa24', 'fastvit_sa36'):
      m = timm.create_model(a, pretrained=False)
      want = {f'c{i + 2}.{k}': tuple(v.shape) for i, st in enumerate(m.stages) for k, v in st.state_dict().items()}
      if want == have: return a
    raise ValueError('unknown FastViT variant')
  return 'resnet34' if any(k.replace('module.', '').startswith('c2.2.') for k in sd) else 'resnet18'


def has_qknorm(sd):
  return any(k.endswith('log_tau') and '.neck.' not in '.' + k.replace('module.', '') for k in sd)


def neck_of(sd):
  return 'dfine' if any(k.replace('module.', '').startswith('neck.') for k in sd) else 'fpn'


def is_color(sd):
  return any(k.replace('module.', '') == 'uv.weight' for k in sd)


def from_checkpoint(path):
  """A Net built to match a checkpoint's trunk, neck, attention and colour input, with its weights loaded."""
  sd = {k.replace('module.', ''): v for k, v in torch.load(path, map_location='cpu').items()}
  net = Net(pretrained=False, color=is_color(sd), arch=arch_of(sd), qknorm=has_qknorm(sd), neck=neck_of(sd))
  net.load_state_dict(sd)
  return net
