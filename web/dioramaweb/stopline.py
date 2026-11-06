"""
Copyright (c) 2026-, Zeph Leggett.

This file comes from jetlink's Diorama (MIT) and is part of diorama, under the
same license: see the LICENSE file in the root directory.

The stop-line model on recorded drives, as the iPhone runs it: train/export.py's
ONNX twin of the Core ML model (stopline_pilot.onnx, heat maps through sigmoid
in the graph), a ResNet-18 FPN that takes the narrow camera's packed frame (Y at
512x256, U and V at 256x128, exactly as the comma sends them) and finds the stop
line, two end points, and the lead vehicle's box. Only the narrow frame, in
both views: one camera's lines agree from frame to frame, which is what the
page's tracking needs.

The page came from jetlink, where a segmentation model fed it, so this sends
what that would have: a seg message (S) whose query map holds the lead box as
one Car, which the page's lead check reads as it reads a segmentation's
vehicles, and a stop-line message:

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
LEAD_SCORE = 0.3
STRIDE, END_SCALE, WH_SCALE = 4, 32.0, 16.0  # the model's output cells (train/model.py)
DS = 2                # the seg message's query map at half the frame's resolution
# A box seen side-on is a vehicle crossing or turning in front, not the lead: wider than SIDE_ASPECT x its height, or
# wider on the road than any vehicle's rear (SIDE_WIDTH m, from the flat-road distance of its bottom). Only from
# SIDE_NEAR m out, and never for a box cut by the frame's bottom (from row NEAR_ROW): a vehicle close ahead fills the
# landscape frame, its box clipped to it, and looks side-on whichever way it faces.
SIDE_ASPECT, SIDE_WIDTH, SIDE_NEAR, NEAR_ROW = 1.9, 3.0, 8.0, 250.0
F, CY, HGT = 910.0, 47.6, 1.25  # the model frame's focal length and horizon row, and the camera's nominal height [m]


def stoplines(frame: int, wide: bool, lines: list[list[float]]) -> bytes:
  out = STOPLINE + struct.pack('<IBB', frame, int(wide), len(lines))
  return out + b''.join(struct.pack('<5f', *l) for l in lines)


def unpack(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """(6, 128, 256) packed frame -> Y (256, 512) and UV (2, 128, 256), the model's inputs."""
  y = np.empty((256, 512), np.float32)
  y[0::2, 0::2], y[1::2, 0::2], y[0::2, 1::2], y[1::2, 1::2] = packed[:4]
  return y, packed[4:6].astype(np.float32)


def side_on(x1: float, y1: float, x2: float, y2: float) -> bool:
  """Is this lead box a vehicle seen side-on (see SIDE_ASPECT)?"""
  if y2 >= NEAR_ROW or y2 <= CY:
    return False
  d = HGT * F / (y2 - CY)
  w = x2 - x1
  return d >= SIDE_NEAR and (w > SIDE_ASPECT * (y2 - y1) or w * d / F > SIDE_WIDTH)


def decode(line: np.ndarray, lead: np.ndarray) -> tuple[list[list[float]], list[float] | None]:
  """One frame's maps (heat maps already through sigmoid) -> stop lines [score, x1, y1, x2, y2] above LINE_SCORE, and
  the lead [score, box] or None (also None when the box is side-on)."""
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
  ch = lead[0]
  i = int(ch.argmax())
  y, x = divmod(i, gw)
  if ch.flat[i] < LEAD_SCORE:
    return lines, None
  cx, cy = x + lead[1, y, x], y + lead[2, y, x]
  w, h = lead[3:5, y, x] * WH_SCALE
  box = [float(v * STRIDE) for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)]
  if side_on(*box):
    return lines, None
  return lines, [float(ch.flat[i])] + box


class StoplineModels:
  """The stop-line model's ONNX on onnxruntime, and the messages the page reads from it."""

  def __init__(self, onnx: Path, device: str):
    import onnxruntime as ort
    self.sess = ort.InferenceSession(str(onnx), providers=providers(device))
    # what offline.drive_meta reads: openpilot's medmodel frame, and the one class the query map holds
    self.meta = {'frame': {'w': 512, 'h': 256, 'f': 910.0, 'cx': 256.0, 'cy': 47.6},
                 'seg': {'labels': ['Car'], 'road_class': {}, 'things': {'Car': 'car'}, 'ds': DS, 'narrow_only': True}}

  def messages(self, frame: int, packed: np.ndarray) -> tuple[list[bytes], float]:
    """The narrow frame's messages, for both views (narrow_only: the page takes its seg messages in bird's-eye too)."""
    t = time.perf_counter()
    y, uv = unpack(packed)
    line, lead = self.sess.run(None, {'frame': y[None, None], 'uv': uv[None]})
    lines, box = decode(line[0], lead[0])
    seconds = time.perf_counter() - t
    # the lead box as the query map's one Car (query 0), every other cell empty (255)
    qmap = np.full((256 // DS, 512 // DS), 255, np.uint8)
    if box is not None:
      x1, y1, x2, y2 = (int(round(v / DS)) for v in box[1:])
      qmap[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 0
    seg = packets.seg(frame, 1000 * seconds, qmap.astype(np.float32), np.array([0]), np.array([1.0 if box else 0.0]),
                      wide=False)
    return [seg, stoplines(frame, False, lines)], seconds
