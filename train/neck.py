"""D-FINE's hybrid encoder (Peng et al. 2024, arXiv 2410.13842; github Peterande/D-FINE src/zoo/dfine/hybrid_encoder.py),
adapted as the pilot's neck: trunk levels at strides 4, 8, 16, 32 -> one stride-4 map for the CenterNet heads.

As in D-FINE:
  - input projection per level: 1x1 conv (no bias) + BatchNorm, no activation, to `hidden` channels
  - intra-scale interaction (AIFI) on the coarsest level only: one post-norm transformer layer (8 heads, FFN 4x, GELU,
    LayerNorm), fixed 2D sin-cos position added to q and k only
  - top-down fusion: lateral 1x1 conv + BN on the coarser map, nearest 2x upsample, concatenated with the finer level,
    then RepNCSPELAN4 (GELAN: CSP branches of RepVGG blocks, SiLU, every conv followed by BatchNorm)
  - D-FINE-S sizes: expansion 0.5, depth multiplier 0.34 (one RepVGG block per branch)
Changed for us:
  - one more top-down step, to stride 4 (D-FINE stops at 8); the bottom-up (PAN) path is dropped, since only the
    stride-4 map feeds the heads and the PAN only produces the coarser outputs
  - the attention is QK-normalized (cosine logits times a learned temperature <= 30, as in model.QKNormAttention) so
    its logits stay in fp16 range on the Neural Engine; D-FINE uses plain nn.MultiheadAttention
Why it suits fp16: every convolution's input is a BatchNorm'd (or LayerNorm'd) map and convolutions carry no bias, and
levels are fused by concatenation, not by an unnormalized sum - the old FPN's sum of bias-carrying laterals into a conv
let a per-channel offset grow unseen behind the next BatchNorm until it overflowed fp16.

deploy() folds every BatchNorm into its conv and every RepVGG block into one 3x3 conv, for export (not named fuse():
timm's reparameterize_model calls any module's fuse()).
"""
import math
import torch
import torch.nn.functional as F
from torch import nn

TAU_MAX = 30.0


class ConvBN(nn.Module):
  def __init__(self, cin, cout, k=1, s=1, g=1, act=False):
    super().__init__()
    self.conv = nn.Conv2d(cin, cout, k, s, (k - 1) // 2, groups=g, bias=False)
    self.bn = nn.BatchNorm2d(cout)
    self.act = nn.SiLU() if act else nn.Identity()

  def forward(self, x):
    return self.act(self.bn(self.conv(x)))

  def fused(self):
    """(weight, bias) of the conv with the BatchNorm folded in."""
    bn = self.bn
    f = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = self.conv.weight * f.view(-1, 1, 1, 1)
    b = (self.conv.bias if self.conv.bias is not None else 0) * f + bn.bias - bn.running_mean * f
    return w, b

  def fold(self):
    w, b = self.fused()
    c = self.conv
    conv = nn.Conv2d(c.in_channels, c.out_channels, c.kernel_size, c.stride, c.padding, groups=c.groups, bias=True)
    conv.weight.data, conv.bias.data = w.detach().clone(), b.detach().clone()
    self.conv, self.bn = conv, nn.Identity()


class VGGBlock(nn.Module):
  """RepVGG: SiLU(3x3 conv + BN  +  1x1 conv + BN); fold() -> one 3x3 conv."""
  def __init__(self, c):
    super().__init__()
    self.conv1, self.conv2, self.act = ConvBN(c, c, 3), ConvBN(c, c, 1), nn.SiLU()
    self.rep = None

  def forward(self, x):
    if self.rep is not None:
      return self.act(self.rep(x))
    return self.act(self.conv1(x) + self.conv2(x))

  def fold(self):
    w3, b3 = self.conv1.fused()
    w1, b1 = self.conv2.fused()
    c = self.conv1.conv
    self.rep = nn.Conv2d(c.in_channels, c.out_channels, 3, 1, 1, bias=True)
    self.rep.weight.data = (w3 + F.pad(w1, [1, 1, 1, 1])).detach().clone()
    self.rep.bias.data = (b3 + b1).detach().clone()
    del self.conv1, self.conv2


class CSPLayer(nn.Module):
  def __init__(self, cin, cout, n=1, expansion=1.0):
    super().__init__()
    h = int(cout * expansion)
    self.conv1, self.conv2 = ConvBN(cin, h, 1, act=True), ConvBN(cin, h, 1, act=True)
    self.bottlenecks = nn.Sequential(*[VGGBlock(h) for _ in range(n)])
    self.conv3 = ConvBN(h, cout, 1, act=True) if h != cout else nn.Identity()

  def forward(self, x):
    return self.conv3(self.bottlenecks(self.conv1(x)) + self.conv2(x))


class RepNCSPELAN4(nn.Module):
  """GELAN block (D-FINE's CSP replacement): split, two CSP+3x3 branches in series, concatenate everything, 1x1."""
  def __init__(self, c1, c2, c3, c4, n=1):
    super().__init__()
    self.c = c3 // 2
    self.cv1 = ConvBN(c1, c3, 1, act=True)
    self.cv2 = nn.Sequential(CSPLayer(c3 // 2, c4, n), ConvBN(c4, c4, 3, act=True))
    self.cv3 = nn.Sequential(CSPLayer(c4, c4, n), ConvBN(c4, c4, 3, act=True))
    self.cv4 = ConvBN(c3 + 2 * c4, c2, 1, act=True)

  def forward(self, x):
    y = list(self.cv1(x).split((self.c, self.c), 1))
    y.extend(m(y[-1]) for m in (self.cv2, self.cv3))
    return self.cv4(torch.cat(y, 1))


def sincos_2d(h, w, dim, temperature=10000.0):
  """D-FINE's build_2d_sincos_position_embedding: (1, h*w, dim) as [w.sin, w.cos, h.sin, h.cos]."""
  gy, gx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing='ij')
  d = dim // 4
  omega = 1.0 / temperature ** (torch.arange(d, dtype=torch.float32) / d)
  ox, oy = gx.flatten()[:, None] * omega[None], gy.flatten()[:, None] * omega[None]
  return torch.cat([ox.sin(), ox.cos(), oy.sin(), oy.cos()], 1)[None]


class AIFI(nn.Module):
  """One post-norm transformer encoder layer over the coarsest map's tokens, QK-normalized attention."""
  def __init__(self, dim, hw, heads=8, ffn=4):
    super().__init__()
    self.heads, self.hd = heads, dim // heads
    self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
    self.log_tau = nn.Parameter(torch.full((heads, 1, 1), math.log(10.0)))
    self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
    self.ffn = nn.Sequential(nn.Linear(dim, ffn * dim), nn.GELU(), nn.Linear(ffn * dim, dim))
    self.register_buffer('pos', sincos_2d(*hw, dim), persistent=False)

  def forward(self, x):
    B, C, H, W = x.shape
    t = x.flatten(2).transpose(1, 2)  # (B, N, C)
    qk = t + self.pos
    split = lambda z: z.reshape(B, H * W, self.heads, self.hd).transpose(1, 2)
    q, k, v = F.normalize(split(self.q(qk)), dim=-1), F.normalize(split(self.k(qk)), dim=-1), split(self.v(t))
    a = ((q * self.log_tau.exp().clamp(max=TAU_MAX)) @ k.transpose(-2, -1)).softmax(-1)
    t = self.norm1(t + self.o((a @ v).transpose(1, 2).reshape(B, H * W, C)))
    t = self.norm2(t + self.ffn(t))
    return t.transpose(1, 2).reshape(B, C, H, W)


class DFineNeck(nn.Module):
  def __init__(self, chans, hidden=128, top_hw=(8, 16), expansion=0.5, depth_mult=0.34):
    """chans: trunk channels at strides 4, 8, 16, 32; top_hw: the stride-32 map's size (512x256 input -> 8x16)."""
    super().__init__()
    n, c4 = round(3 * depth_mult), round(expansion * hidden // 2)
    self.proj = nn.ModuleList(ConvBN(c, hidden, 1) for c in chans)
    self.aifi = AIFI(hidden, top_hw)
    self.lateral = nn.ModuleList(ConvBN(hidden, hidden, 1) for _ in chans[1:])  # coarse -> fine: s32, s16, s8
    self.fpn = nn.ModuleList(RepNCSPELAN4(2 * hidden, hidden, 2 * hidden, c4, n) for _ in chans[1:])
    self.out_channels = hidden

  def forward(self, cs):
    p = [proj(c) for proj, c in zip(self.proj, cs)]
    x = self.aifi(p[-1])
    for i, (lat, blk) in enumerate(zip(self.lateral, self.fpn)):
      x = blk(torch.cat([F.interpolate(lat(x), scale_factor=2.0, mode='nearest'), p[-2 - i]], 1))
    return x

  def deploy(self):
    for m in list(self.modules()):
      if isinstance(m, VGGBlock) and m.rep is None:
        m.fold()
    for m in list(self.modules()):
      if isinstance(m, ConvBN) and not isinstance(m.bn, nn.Identity):
        m.fold()
    return self
