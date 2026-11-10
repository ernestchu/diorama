"""Split the VLM-labelled approach frames: those on which the VLM and our model agree are taken as labelled
(auto_items.py), the rest go to a person (queue_auto.py).

The model ("student") is a checkpoint trained without these frames. A frame goes to review when the student is
unsure (its best line scores in [LO, HI)), when the two disagree on whether there is a line (the student's at TH),
or when both find one but an end differs by more than TOL px (model frame, 512x256).

  python auto_split.py runs/<name>/<name>-s0/last.pt   -> $STOPLINE_C1M/auto/student.json (id -> [score, x1, y1,
                                                          x2, y2]), auto/to_review.json
"""
import json, sys
import numpy as np, torch
from torch.utils.data import DataLoader

from model import from_checkpoint
from train_c1m import C1M, decode_all
from paths import C1M as DATA

A = f'{DATA}/auto'
LO, HI, TH, TOL = 0.2, 0.5, 0.3, 20.0


def end_diff(a, b):
  """The larger end-point distance between two lines, whichever way round their ends are."""
  d = lambda p, q: np.hypot(p[0] - q[0], p[1] - q[1])
  return min(max(d(a[:2], b[:2]), d(a[2:], b[2:])), max(d(a[:2], b[2:]), d(a[2:], b[:2])))


def predict(ckpt, items):
  class Auto(C1M):
    def __init__(self, items):
      self.items, self.train, self.extra, self.n_extra = sorted(items.items()), False, [], 0
  dev = torch.device('cuda', 0)
  model = from_checkpoint(ckpt).to(dev).eval()
  out = {}
  with torch.no_grad():
    for x, c, _, names in DataLoader(Auto(items), 32, num_workers=8):
      for n, d in zip(names, decode_all(model(x.to(dev), c.to(dev)).cpu())):
        out[n] = [round(v, 3) for v in d['lines'][0]]
  return out


def main():
  items = json.load(open(f'{A}/items.json'))
  vlm = json.load(open(f'{A}/vlm.json'))
  stu = predict(sys.argv[1], items)
  json.dump(stu, open(f'{A}/student.json', 'w'))
  review = []
  for k in sorted(items):
    sc, *sl = stu[k]
    h = vlm.get(k)
    if LO <= sc < HI or (h is None) == (sc >= TH) or (h is not None and end_diff(h, sl) > TOL):
      review.append(k)
  json.dump(review, open(f'{A}/to_review.json', 'w'))
  print(len(items), 'frames;', len(review), 'to review;', len(items) - len(review), 'agreed')


if __name__ == '__main__':
  main()
