"""Pilot data: the SLSDNet stop-line dataset.

Each image -> grey (BT.601 video-range luma, as the comma frame's Y plane) and its stop line as two points (the
dataset's thickness is dropped: our protocol is two points)."""
import json, os, sys
import numpy as np, cv2

from paths import SLSD_RAW as RAW, SLSD as OUT


def luma(rgb):
  r, g, b = [rgb[..., i].astype(np.float32) for i in range(3)]
  return np.clip(16 + 0.257 * r + 0.504 * g + 0.098 * b, 0, 255).astype(np.uint8)


if __name__ == '__main__':
  os.makedirs(f'{OUT}/img', exist_ok=True)
  split = {f[:-4]: s for s in ('train', 'test') for f in open(f'{RAW}/{s}.txt').read().split()}
  names = sorted(split)
  items = {}
  for i in range(0, len(names), 16):
    chunk = names[i:i + 16]
    rgbs = [cv2.cvtColor(cv2.imread(f'{RAW}/img/{n}.jpg'), cv2.COLOR_BGR2RGB) for n in chunk]
    for n, rgb in zip(chunk, rgbs):
      cv2.imwrite(f'{OUT}/img/{n}.png', luma(rgb))
      x1, y1, x2, y2, hh = map(int, open(f'{RAW}/labels/{n}.txt').read().split())
      line = None if hh < 0 else ([x1, y1, x2, y2] if x1 <= x2 else [x2, y2, x1, y1])
      h, w = rgb.shape[:2]
      items[n] = {'split': split[n], 'w': w, 'h': h, 'line': line}
  json.dump(items, open(f'{OUT}/items.json', 'w'))
  print(len(items), 'images;', sum(v['line'] is not None for v in items.values()), 'with a line')
