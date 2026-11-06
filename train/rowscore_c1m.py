"""A distance-oriented stop-line score for the comma1M runs: a labelled line counts as found when a predicted line
overlaps at least half of it horizontally and its row is within ROW_TOL px there (the row sets the distance).
Reports precision, recall and F1 over score thresholds, per configuration."""
import glob, json, re, sys
import numpy as np

ROW_TOL = 4.0


def row_at(l, x):
  x1, y1, x2, y2 = l
  return (y1 + y2) / 2 if abs(x2 - x1) < 1e-3 else y1 + (x - x1) / (x2 - x1) * (y2 - y1)


def match(g, p):
  lo, hi = max(min(g[0], g[2]), min(p[0], p[2])), min(max(g[0], g[2]), max(p[0], p[2]))
  if hi - lo < 0.5 * max(1, abs(g[2] - g[0])): return False
  return np.median([abs(row_at(g, x) - row_at(p, x)) for x in np.linspace(lo, hi, 5)]) <= ROW_TOL


def prf(recs, th, sub=lambda r: True):
  tp = npred = ngt = 0
  for r in recs:
    if not sub(r): continue
    preds = [p[1:] for p in r['lines'] if p[0] >= th]
    npred += len(preds); ngt += len(r['gl'])
    used = set()
    for g in r['gl']:
      j = next((j for j, p in enumerate(preds) if j not in used and match(g, p)), None)
      if j is not None: used.add(j); tp += 1
  p, rc = tp / max(1, npred), tp / max(1, ngt)
  return p, rc, 2 * p * rc / max(1e-9, p + rc)


if __name__ == '__main__':
  pat = sys.argv[1] if len(sys.argv) > 1 else 'runs/*/*-s[0-9]/test_preds.json'
  cfgs = {}
  for f in sorted(glob.glob(pat)):
    cfgs.setdefault(re.search(r'([\w-]+)-s\d+/test_preds', f).group(1), []).append(json.load(open(f)))
  for cfg, runs in cfgs.items():
    print(cfg)
    for th in (0.1, 0.2, 0.3, 0.5):
      a = np.array([prf(r, th) for r in runs])
      print(f'  th {th}:  P {a[:,0].mean():.3f}  R {a[:,1].mean():.3f}  F1 {a[:,2].mean():.3f} ± {a[:,2].std():.3f}')
