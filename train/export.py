"""Export the pilot to Core ML (ML program, fp16) and ONNX. The Core ML model takes Y as a 512x256 grey image
and, for a colour model, U and V as a (1, 2, 128, 256) tensor (the interleaved CbCr plane of a 4:2:0 frame),
and returns the raw line and lead maps; decoding (a peak search) is done by the app.

  python export.py runs/pilot/last.pt runs/pilot/export
"""
import sys, os
import numpy as np, torch
import coremltools as ct
from model import Net

ckpt, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
sd = {k.replace('module.', ''): v for k, v in torch.load(ckpt, map_location='cpu').items()}
color = 'uv.weight' in sd
net = Net(pretrained=False, color=color)
net.load_state_dict(sd)
net.eval()


class Exported(torch.nn.Module):
  """Heat maps go through sigmoid on device; everything else stays raw."""
  def __init__(self, net):
    super().__init__(); self.net = net

  def forward(self, *x):
    line, lead = self.net(*x)
    line = torch.cat([torch.sigmoid(line[:, :1]), line[:, 1:7], torch.sigmoid(line[:, 7:8])], 1)
    lead = torch.cat([torch.sigmoid(lead[:, :1]), lead[:, 1:]], 1)
    return line, lead


m = Exported(net).eval()
x = (torch.rand(1, 1, 256, 512) * 255,) + ((torch.rand(1, 2, 128, 256) * 255,) if color else ())
names = ['frame', 'uv'][:len(x)]
with torch.no_grad():
  traced = torch.jit.trace(m, x)
torch.onnx.export(m, x, f'{out}/stopline_pilot.onnx', input_names=names, output_names=['line', 'lead'], opset_version=17,
                  dynamo=False)

inputs = [ct.ImageType(name='frame', shape=(1, 1, 256, 512), color_layout=ct.colorlayout.GRAYSCALE)]
if color:
  inputs.append(ct.TensorType(name='uv', shape=(1, 2, 128, 256), dtype=np.float16))
ml = ct.convert(traced, inputs=inputs,
                outputs=[ct.TensorType(name='line'), ct.TensorType(name='lead')],
                convert_to='mlprogram', compute_precision=ct.precision.FLOAT16,
                minimum_deployment_target=ct.target.iOS17)
ml.short_description = ('Stop lines (two points) and the lead car (a box) from a 512x256 openpilot medmodel frame'
                        + (' (Y + 4:2:0 UV).' if color else ' (Y only).'))
ml.save(f'{out}/StoplinePilot.mlpackage')
n = sum(p.numel() for p in net.parameters())
print(f'params {n / 1e6:.2f} M; saved {out}/StoplinePilot.mlpackage and stopline_pilot.onnx')
