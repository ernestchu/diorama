"""The pilot network: a 512x256 frame -> the stop line (two points).

Input is the frame as the comma camera encodes it: Y at 512x256 and, with color=True, U and V at 256x128
(YUV420). U and V join after the first stride-2 convolution, which already works at their resolution.

Plain convolutions, batch norm, ReLU and nearest upsampling only, so it converts to Core ML and runs on the
Neural Engine. One ResNet-18 trunk, a small FPN down to stride 4 (128x64), and a CenterNet-style head:
  line: heat (1), sub-cell centre offset (2), end points from the centre (4), line mask (1, training aid)
"""
import torch
import torch.nn.functional as F
from torch import nn
import torchvision

STRIDE = 4
END_SCALE = 32.0  # end-point offsets are predicted in cells / END_SCALE


def head(c, out, bias=0.0):
  h = nn.Sequential(nn.Conv2d(c, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, out, 1))
  nn.init.constant_(h[-1].bias, bias)
  return h


class Net(nn.Module):
  def __init__(self, pretrained=True, width=96, color=False):
    super().__init__()
    r = torchvision.models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    conv1.weight.data = r.conv1.weight.data.sum(1, keepdim=True)  # grey: the RGB filters summed
    self.conv1, self.stem = conv1, nn.Sequential(r.bn1, r.relu, r.maxpool)
    self.c2, self.c3, self.c4, self.c5 = r.layer1, r.layer2, r.layer3, r.layer4
    self.lat = nn.ModuleList(nn.Conv2d(c, width, 1) for c in (64, 128, 256, 512))
    self.smooth = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
    self.line = head(width, 8)
    with torch.no_grad():
      self.line[-1].bias[0] = -4.6  # heat starts near 0.01
      self.line[-1].bias[7] = -4.6
    self.uv = None  # made last, so grey and colour nets with one seed start from the same weights
    if color:
      self.uv = nn.Conv2d(2, 64, 3, padding=1, bias=False)
      nn.init.zeros_(self.uv.weight)  # starts as the grey network; colour is learnt on top

  def forward(self, y, uv=None):
    """y: (N, 1, 256, 512) luma 0..255; uv: (N, 2, 128, 256) chroma 0..255 (used when color=True).
    Returns the raw line maps (N, 8, 64, 128)."""
    x = self.conv1((y - 110.0) / 60.0)
    if self.uv is not None:
      x = x + self.uv((uv - 128.0) / 16.0)
    c2 = self.c2(self.stem(x))
    c3 = self.c3(c2)
    c4 = self.c4(c3)
    c5 = self.c5(c4)
    p = self.lat[3](c5)
    for lat, c in ((self.lat[2], c4), (self.lat[1], c3), (self.lat[0], c2)):
      p = F.interpolate(p, scale_factor=2.0, mode='nearest') + lat(c)
    p = self.smooth(p)
    return self.line(p)


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
