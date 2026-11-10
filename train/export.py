"""Export a checkpoint to Core ML (ML program, fp16) and ONNX. The Core ML model takes Y as a 512x256 grey image
and, for a colour model, U and V as a (1, 2, 128, 256) tensor (the interleaved CbCr plane of a 4:2:0 frame), and
returns one output, `line` (1, 8, 64, 128): heat (sigmoid), centre offset (2), end points (4), mask (sigmoid).
Decoding (a peak search) is done by the app.

FastViT trunks have their train-time branches folded into single convolutions, then fp16safe.make_fp16_safe makes
every op stay in fp16 range on the Neural Engine (exact in fp32 up to rounding); a D-FINE neck has its RepVGG blocks
and BatchNorms folded.

  python export.py runs/v5/v5-s0/last.pt out/
"""
import sys, os
import numpy as np, torch
import coremltools as ct
from model import from_checkpoint

ckpt, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
net = from_checkpoint(ckpt).eval()
color = net.uv is not None
ref = (torch.rand(1, 1, 256, 512) * 255,) + ((torch.rand(1, 2, 128, 256) * 255,) if color else ())
if net.vit:  # FastViT: fold the train-time branches (MobileOne / RepMixer) into single convolutions
  from timm.utils.model import reparameterize_model
  with torch.no_grad(): a = net(*ref)
  net = reparameterize_model(net)
  with torch.no_grad(): b = net(*ref)
  print('reparameterized; max diff', (a - b).abs().max().item())
  from fp16safe import make_fp16_safe
  net = make_fp16_safe(net)
  with torch.no_grad(): c = net(*ref)
  print('folded for fp16; max diff', (b - c).abs().max().item())
elif net.dfine:  # D-FINE neck on a ResNet trunk: fold its BatchNorms and RepVGG blocks
  net.neck.deploy()


class Exported(torch.nn.Module):
  """The heat and mask maps go through sigmoid on device; the offsets and ends stay raw."""
  def __init__(self, net):
    super().__init__(); self.net = net

  def forward(self, *x):
    line = self.net(*x)
    return torch.cat([torch.sigmoid(line[:, :1]), line[:, 1:7], torch.sigmoid(line[:, 7:8])], 1)


m = Exported(net).eval()
names = ['frame', 'uv'][:len(ref)]
with torch.no_grad():
  traced = torch.jit.trace(m, ref)
torch.onnx.export(m, ref, f'{out}/stopline_pilot.onnx', input_names=names, output_names=['line'], opset_version=17,
                  dynamo=False)

inputs = [ct.ImageType(name='frame', shape=(1, 1, 256, 512), color_layout=ct.colorlayout.GRAYSCALE)]
if color:
  inputs.append(ct.TensorType(name='uv', shape=(1, 2, 128, 256), dtype=np.float16))
ml = ct.convert(traced, inputs=inputs, outputs=[ct.TensorType(name='line')],
                convert_to='mlprogram', compute_precision=ct.precision.FLOAT16,
                minimum_deployment_target=ct.target.iOS17)
ml.short_description = ('The stop line (two points) from a 512x256 openpilot medmodel frame'
                        + (' (Y + 4:2:0 UV).' if color else ' (Y only).'))
ml.save(f'{out}/StoplinePilot.mlpackage')
n = sum(p.numel() for p in net.parameters())
print(f'params {n / 1e6:.2f} M; saved {out}/StoplinePilot.mlpackage and stopline_pilot.onnx')
