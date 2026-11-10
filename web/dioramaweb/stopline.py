"""
Copyright (c) 2026-, Zeph Leggett.

This file comes from jetlink's Diorama (MIT) and is part of diorama, under the
same license: see the LICENSE file in the root directory.

The stop-line model on recorded drives, as the iPhone runs it: the ONNX twin of
the published Core ML model (stopline_pilot.onnx, heat maps through sigmoid in
the graph; since v5 a FastViT-MA36 trunk with a D-FINE neck), which takes the
narrow camera's packed frame (Y at 512x256, U and V at 256x128, exactly as the
comma sends them) and finds the stop line, two end points: its one output,
`line`. Only the narrow frame, in both views: one camera's lines agree from frame
to frame, which is what the page's tracking needs.

The page came from jetlink, where a segmentation model fed it, so this sends a
seg message (S) with an empty query map, which says the model ran on the frame
and for how long, and a stop-line message:

    L  stop lines  u32 frame, u8 wide, u8 count, count x f32[5] (score, x1, y1, x2, y2) in the frame's pixels

No road surface, crosswalks or other classes: the page draws the lane lines,
road edges and plan from the driving model as before.
"""
from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np

from dioramaweb import packets
from commasim.driving import providers

STOPLINE = b'L'
LINES = 1             # a frame has at most one stop line, in the ego lane
LINE_SCORE = 0.3      # a line counts from here (the test set's best F1)
STRIDE, END_SCALE = 4, 32.0  # the model's output cells (train/model.py)
DS = 2                # the seg message's query map at half the frame's resolution


def stoplines(frame: int, wide: bool, lines: list[list[float]]) -> bytes:
  out = STOPLINE + struct.pack('<IBB', frame, int(wide), len(lines))
  return out + b''.join(struct.pack('<5f', *l) for l in lines)


def unpack(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """(6, 128, 256) packed frame -> Y (256, 512) and UV (2, 128, 256), the model's inputs."""
  y = np.empty((256, 512), np.float32)
  y[0::2, 0::2], y[1::2, 0::2], y[0::2, 1::2], y[1::2, 1::2] = packed[:4]
  return y, packed[4:6].astype(np.float32)


def decode(line: np.ndarray) -> list[list[float]]:
  """One frame's `line` map (its heat map already through sigmoid) -> stop lines [score, x1, y1, x2, y2] above
  LINE_SCORE."""
  lh = line[0]
  gh, gw = lh.shape
  pad = np.pad(lh, 2, constant_values=-1)
  peak = np.max([pad[dy:dy + gh, dx:dx + gw] for dy in range(5) for dx in range(5)], axis=0)
  peaks = np.where(lh == peak, lh, 0).ravel()
  lines = []
  for i in np.argsort(-peaks)[:LINES]:
    if peaks[i] < LINE_SCORE:
      break
    y, x = divmod(int(i), gw)
    cx, cy = x + line[1, y, x], y + line[2, y, x]
    e = line[3:7, y, x] * END_SCALE
    lines.append([float(peaks[i])] + [float(v * STRIDE) for v in (cx + e[0], cy + e[1], cx + e[2], cy + e[3])])
  return lines


class StoplineModels:
  """The stop-line model's ONNX on onnxruntime, and the messages the page reads from it."""

  def __init__(self, onnx: Path, device: str):
    import onnxruntime as ort
    self.sess = ort.InferenceSession(str(onnx), providers=providers(device))
    # what offline.drive_meta reads: openpilot's medmodel frame, and a query map of no classes
    self.meta = {'frame': {'w': 512, 'h': 256, 'f': 910.0, 'cx': 256.0, 'cy': 47.6},
                 'seg': {'labels': [], 'road_class': {}, 'things': {}, 'ds': DS, 'narrow_only': True}}

  def messages(self, frame: int, packed: np.ndarray) -> tuple[list[bytes], float]:
    """The narrow frame's messages, for both views (narrow_only: the page takes its seg messages in bird's-eye too)."""
    t = time.perf_counter()
    y, uv = unpack(packed)
    (line,) = self.sess.run(['line'], {'frame': y[None, None], 'uv': uv[None]})
    lines = decode(line[0])
    seconds = time.perf_counter() - t
    # an empty query map: the model ran on this frame, in this long
    qmap = np.full((256 // DS, 512 // DS), 255, np.uint8)
    seg = packets.seg(frame, 1000 * seconds, qmap.astype(np.float32), np.zeros(0, np.uint8), np.zeros(0), wide=False)
    return [seg, stoplines(frame, False, lines)], seconds
