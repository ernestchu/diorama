"""Training items from the VLM-labelled approach frames on which the VLM and the student agree (auto_split.py; the
frames not sent for review): no line where both say none; where both find one (ends within 20 px), the mean of the
two. A frame a person has since labelled is a hand label (labels/store.json, build_c1m.py) and is left out here.

  python auto_items.py  -> $STOPLINE_C1M/auto_items.json
"""
import json
import numpy as np

from paths import C1M, LABELS

A = f'{C1M}/auto'


def same_dir(h, s):
  """The student's line with its ends in the order of the VLM's."""
  a = np.hypot(s[0] - h[0], s[1] - h[1]) + np.hypot(s[2] - h[2], s[3] - h[3])
  b = np.hypot(s[2] - h[0], s[3] - h[1]) + np.hypot(s[0] - h[2], s[1] - h[3])
  return s if a <= b else s[2:] + s[:2]


def main():
  items = json.load(open(f'{A}/items.json'))
  vlm = json.load(open(f'{A}/vlm.json'))
  stu = json.load(open(f'{A}/student.json'))
  review = set(json.load(open(f'{A}/to_review.json')))
  hand = {k for k, v in json.load(open(f'{LABELS}/store.json')).items() if v.get('reviewed', True)}
  out = {}
  for k in sorted(k for k in items if k not in review and k not in hand):
    v = dict(items[k])
    h, (sc, *s) = vlm[k], stu[k]
    if h:
      assert sc >= 0.3, k
      s = same_dir(h, s)
      line = [round((a + b) / 2, 2) for a, b in zip(h, s)]
      v['lines'] = [line if line[0] <= line[2] else line[2:] + line[:2]]
    else:
      v['lines'] = []
    out[k] = v
  json.dump(out, open(f'{C1M}/auto_items.json', 'w'))
  print(len(out), 'frames,', sum(bool(v['lines']) for v in out.values()), 'with a line')


if __name__ == '__main__':
  main()
