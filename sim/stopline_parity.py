"""
Copyright (c) 2026-, Ernie Chu.

This file is part of diorama and is licensed under the MIT License.

Frames for the Swift decoder's parity test: the stop-line model's ONNX twin
over a recorded drive, decoded as diorama/stopline.py decodes (less the
sigmoid, which this export does in the graph). Writes the packed frames and
what the decode found to models/parity, which DioramaKit's tests read.

    uv run python stopline_parity.py ~/Downloads/jetlink/driving_footage
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from commasim.route import Route

LINE_SCORE = 0.3
STRIDE, END_SCALE = 4, 32.0


def decode(line):
  lh = line[0]
  gh, gw = lh.shape
  pad = np.pad(lh, 2, constant_values=-1)
  peak = np.max([pad[dy:dy + gh, dx:dx + gw] for dy in range(5) for dx in range(5)], axis=0)
  peaks = np.where(lh == peak, lh, 0).ravel()
  i = int(np.argmax(peaks))
  found = None
  if peaks[i] >= LINE_SCORE:
    y, x = divmod(i, gw)
    cx, cy = x + line[1, y, x], y + line[2, y, x]
    e = line[3:7, y, x] * END_SCALE
    found = [float(peaks[i])] + [float(v * STRIDE) for v in (cx + e[0], cy + e[1], cx + e[2], cy + e[3])]
  return found


def main():
  root = Path(__file__).resolve().parent.parent / 'models'
  sess = ort.InferenceSession(str(root / 'stopline_pilot.onnx'), providers=['CPUExecutionProvider'])
  route = Route.open([Path(p) for p in sys.argv[1:]])
  with_line, without = [], []
  for k, _, warped in route.warped(0, 1200):
    if k % 4:
      continue
    p = warped[0]
    y = np.empty((256, 512), np.float32)
    y[0::2, 0::2], y[1::2, 0::2], y[0::2, 1::2], y[1::2, 1::2] = p[:4]
    (line,) = sess.run(['line'], {'frame': y[None, None], 'uv': p[4:6].astype(np.float32)[None]})
    found = decode(line[0])
    (with_line if found else without).append((k, p, found))
    if len(with_line) >= 8 and len(without) >= 4:
      break
  picked = with_line[:8] + without[:4]
  out = root / 'parity'
  out.mkdir(exist_ok=True)
  np.stack([p for _, p, _ in picked]).tofile(out / 'frames.u8')
  (out / 'expected.json').write_text(json.dumps([{'frame': k, 'line': f} for k, _, f in picked], indent=1))
  print(f'{len(with_line[:8])} frames with a line, {len(without[:4])} without, in {out}')


if __name__ == '__main__':
  main()
