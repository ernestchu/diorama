"""Colour for the pilot: each image as full-resolution YUV (BT.601 video range, the comma camera's encoding; the
loader subsamples U and V to 4:2:0 after augmentation), and how much each stop line stands out from the road
around it in luma and in chroma, so the grey-vs-colour comparison can be read on low-contrast lines.

Run after prep.py; adds 'contrast': {'y', 'c'} to data/items.json for every image with a line.
"""
import json, os
import numpy as np, cv2

from paths import SLSD_RAW as RAW, SLSD as OUT


def yuv(rgb):
  r, g, b = [rgb[..., i].astype(np.float32) for i in range(3)]
  y = 16 + 0.257 * r + 0.504 * g + 0.098 * b
  u = 128 - 0.148 * r - 0.291 * g + 0.439 * b
  v = 128 + 0.439 * r - 0.368 * g - 0.071 * b
  return np.clip(np.stack([y, u, v], -1), 0, 255).astype(np.uint8)


def contrast(img, x1, y1, x2, y2, h):
  """Mean |paint - road| along the line: paint within h/4 of its centre line, road in bands just past its
  edges on both sides. Returns (luma difference, chroma (U, V) distance)."""
  d = np.array([x2 - x1, y2 - y1], np.float32)
  n = np.array([-d[1], d[0]]) / max(1e-6, np.linalg.norm(d))
  t = np.linspace(0.05, 0.95, 64)[:, None]
  base = np.array([x1, y1], np.float32) + t * d

  def sample(offsets):
    pts = (base[:, None, :] + offsets[None, :, None] * n).reshape(-1, 2).astype(np.float32)
    xs = cv2.remap(img.astype(np.float32), pts[:, 0].reshape(-1, 1), pts[:, 1].reshape(-1, 1), cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE)
    return xs.reshape(-1, 3).mean(0)

  q = max(1.0, h / 4)
  paint = sample(np.linspace(-q, q, 3))
  gap = h / 2 + 3
  road = sample(np.concatenate([np.linspace(gap, gap + h, 3), -np.linspace(gap, gap + h, 3)]))
  return float(abs(paint[0] - road[0])), float(np.hypot(paint[1] - road[1], paint[2] - road[2]))


if __name__ == '__main__':
  os.makedirs(f'{OUT}/yuv', exist_ok=True)
  items = json.load(open(f'{OUT}/items.json'))
  for n, v in items.items():
    img = yuv(cv2.cvtColor(cv2.imread(f'{RAW}/img/{n}.jpg'), cv2.COLOR_BGR2RGB))
    cv2.imwrite(f'{OUT}/yuv/{n}.png', img)  # channels Y, U, V (read back with IMREAD_UNCHANGED)
    x1, y1, x2, y2, h = map(int, open(f'{RAW}/labels/{n}.txt').read().split())
    if h >= 0:
      cy, cc = contrast(img, x1, y1, x2, y2, h)
      v['contrast'] = {'y': round(cy, 2), 'c': round(cc, 2)}
  json.dump(items, open(f'{OUT}/items.json', 'w'))
  cs = np.array([[v['contrast']['y'], v['contrast']['c']] for v in items.values() if 'contrast' in v])
  print(len(items), 'images; line contrast median luma', np.median(cs[:, 0]).round(1), 'chroma', np.median(cs[:, 1]).round(1))
  print('luma quartiles', np.percentile(cs[:, 0], [25, 50, 75]).round(1))
