"""Lead boxes against the teacher on the mined slow frames' test split (mine_near.py), by the lead's size.

For each run: recall and precision at score >= 0.3 and IoU >= 0.5 with the teacher's lead, the median IoU of the
matches, and the share of frames without a lead given one. Ambiguous frames are left out.

  python eval_near.py RUN_DIR [RUN_DIR ...]   (each with last.pt; seeds of one recipe are averaged if given together)
"""
import json, os, sys
import numpy as np, torch
from torch.utils.data import DataLoader
from model import Net
from train import iou
from train_c1m import C1M, decode_all
from mine_near import OUT, size

SIZES = ['cut', 'tall', 'mid', 'small']


class Near(C1M):
  def __init__(self, items):
    self.items, self.train = sorted(items.items()), False


def predict(ck, ds, device):
  sd = torch.load(ck, map_location='cpu')
  model = Net(pretrained=False, color='uv.weight' in sd).to(device).eval()
  model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
  out = {}
  with torch.no_grad():
    for x, c, _, names in DataLoader(ds, 32, num_workers=8):
      line, lead = model(x.to(device), c.to(device))
      for n, d in enumerate(decode_all(line.cpu(), lead.cpu())):
        out[names[n]] = d['lead']
  return out


def table(items, pred, th=0.3):
  res = {}
  for c in SIZES + ['none']:
    tp = fn = fp = n = 0; ious = []
    for k, v in items.items():
      if size(v['lead']) != c:
        continue
      n += 1
      p = pred[k][1:] if pred[k][0] >= th else None
      g = v['lead']
      if g is None:
        fp += p is not None
      elif p is not None and iou(p, g) >= 0.5:
        tp += 1; ious.append(iou(p, g))
      else:
        fn += 1; fp += p is not None
    res[c] = {'n': n, 'false lead': fp / max(1, n)} if c == 'none' else \
             {'n': n, 'R': tp / max(1, tp + fn), 'P': tp / max(1, tp + fp), 'IoU': float(np.median(ious)) if ious else None}
  return res


if __name__ == '__main__':
  items = {k: v for k, v in json.load(open(f'{OUT}/items.json')).items() if v['split'] == 'test' and not v['ambiguous']}
  ds, device = Near(items), torch.device('cuda', 0)
  runs = [table(items, predict(f'{r}/last.pt', ds, device)) for r in sys.argv[1:]]
  print(' '.join(os.path.basename(r.rstrip('/')) for r in sys.argv[1:]), f'({len(items)} test frames)')
  for c in SIZES:
    v = {m: np.array([r[c][m] for r in runs], float) for m in ('R', 'P', 'IoU')}
    print(f"{c:6s} n={runs[0][c]['n']:5d}  R {v['R'].mean():.3f}±{v['R'].std():.3f}  P {v['P'].mean():.3f}  "
          f"IoU {np.nanmean(v['IoU']):.2f}")
  f = np.array([r['none']['false lead'] for r in runs])
  print(f"none   n={runs[0]['none']['n']:5d}  false lead {f.mean():.3f}±{f.std():.3f}")
