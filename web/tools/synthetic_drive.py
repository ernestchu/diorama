"""
Copyright (c) 2026-, Ernie Chu.

This file is part of diorama and is licensed under the MIT License.

A made-up drive for the parity tests, with no footage and nothing from a real
drive: 24 s on a gently curving road, scripted.

  0-6 s    cruising at 15 m/s: the road view
  6-12 s   a lead car ahead, the car slowing behind it to 6 m/s: traffic
  12-20 s  the lead gone, a painted stop line ahead, the car stopping short of it
  20-24 s  pulling away again

The driving model's heads are what a perfect model would say of that (the plan
is the script's own future), in the raw form a drive message carries: means
and log stds, logits. The stop-line model's messages are an empty query map and
the stop line's ends projected into the narrow frame (f 910, cy 47.6) from where
it stands, every other frame, as the phone sends them. The stop line is given
as the real model gives it, not as it is: it runs on into the next lanes
(OVERRUN), and near, the frame's edges cut it short. The page fits it back to
the ego lane, which is what the parity tests must see the app do too.

    python tools/synthetic_drive.py ../DioramaKit/Tests/Fixtures/synthetic/drive.diorama
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dioramaweb import packets  # noqa: E402
from dioramaweb.stopline import stoplines  # noqa: E402

FPS, SECONDS = 20, 24.0
IDX_N = 33
T_IDXS = 10.0 * (np.arange(IDX_N) / 32) ** 2
X_IDXS = 192.0 * (np.arange(IDX_N) / 32) ** 2
LEAD_T = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
HEIGHT = 1.37  # the camera above the road, m
F, CX, CY = 910.0, 256.0, 47.6  # openpilot's medmodel frame, 512x256
CURVE = 4e-4  # the road's y = CURVE * x^2 ahead of the car, m
STOP_AT = 170.0  # the stop line's distance along the road from the start, m
OVERRUN = (0.7, 1.3)  # how far the model's stop line runs past the ego lane's left and right lines, m
FIELDS = (('plan', 990), ('lane_lines', 528), ('lane_lines_prob', 8), ('road_edges', 264), ('lead', 144), ('lead_prob', 3),
          ('pose', 12))


def speed(t: float) -> float:
  """The script's speed at t seconds, m/s."""
  if t < 6:
    return 15.0
  if t < 12:
    return 15.0 - 9.0 * (t - 6) / 6  # down to 6 m/s behind the lead
  if t < 20:
    return 0.0  # placeholder: the stop is solved below, so the car stops short of the line
  return min(1.5 * (t - 20), 10.0)


def profile() -> tuple[np.ndarray, np.ndarray]:
  """Speed and distance along the road every 50 ms, out to the plan's 10 s past the end, with the 12-20 s stretch
  braking evenly from 6 m/s to stop 2.5 m short of the stop line."""
  t = np.arange(0, SECONDS + 10.5, 1 / FPS)
  v = np.array([speed(x) for x in t])
  x = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 / FPS)])
  i12 = int(12 * FPS)
  room = STOP_AT - 2.5 - x[i12]
  brake = 6.0 ** 2 / (2 * room)  # an even stop from 6 m/s in the room left
  stop_t = 12 + 6.0 / brake
  for i in range(i12, int(20 * FPS)):
    v[i] = max(6.0 - brake * (t[i] - 12), 0.0) if t[i] < stop_t else 0.0
  x = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 / FPS)])
  return v, x


def lead_at(t: float) -> float | None:
  """The lead's distance ahead, m, while there is one."""
  if 6 <= t < 12:
    return 30.0 - 16.0 * (t - 6) / 6  # closing to 14 m
  return None


def heads(k: int, v: np.ndarray, x: np.ndarray) -> np.ndarray:
  t = k / FPS
  steps = np.clip(np.round((t + T_IDXS) * FPS).astype(int), 0, len(v) - 1)
  px = x[steps] - x[k]
  pv = v[steps]
  pa = np.gradient(v)[steps] * FPS
  plan = np.zeros((IDX_N, 15))
  plan[:, 0], plan[:, 1] = px, CURVE * px ** 2
  plan[:, 3], plan[:, 4] = pv, 2 * CURVE * px * pv
  plan[:, 6] = pa
  plan_std = np.full((IDX_N, 15), 0.5)

  lanes = np.zeros((4, IDX_N, 2))
  for i, off in enumerate((-5.4, -1.8, 1.8, 5.4)):
    lanes[i, :, 0] = CURVE * X_IDXS ** 2 + off
  lane_prob = np.zeros(8)
  for i, p in enumerate((0.6, 0.95, 0.95, 0.6)):
    lane_prob[2 * i], lane_prob[2 * i + 1] = -np.log(p / (1 - p)), np.log(p / (1 - p))
  edges = np.zeros((2, IDX_N, 2))
  for i, off in enumerate((-5.6, 5.6)):
    edges[i, :, 0] = CURVE * X_IDXS ** 2 + off

  lead = np.zeros((3, 6, 4))
  d = lead_at(t)
  if d is not None:
    vl = 6.0
    lead[:, :, 0] = d + (vl - v[k]) * LEAD_T
    lead[:, :, 2] = vl
  lead_prob = np.full(3, 3.0 if d is not None else -3.0)

  pose = np.array([v[k], 0, 0, 0, 0, 2 * CURVE * 0 * v[k]])  # straight at the car: the curve starts ahead of it
  log = lambda s: np.log(np.full(np.shape(s), s) if np.isscalar(s) else s)  # noqa: E731
  return np.concatenate([
    plan.ravel(), log(plan_std).ravel(),
    lanes.ravel(), np.full(lanes.size, np.log(0.2)), lane_prob,
    edges.ravel(), np.full(edges.size, np.log(0.3)),
    lead.ravel(), np.full(lead.size, np.log(0.5)), lead_prob,
    pose, np.full(6, np.log(0.1)),
  ]).astype(np.float32)


def pixel(ahead: float, right: float, down: float) -> tuple[float, float]:
  """A road point (m ahead, m right, m below the camera) in the narrow frame's pixels."""
  return F * right / ahead + CX, F * down / ahead + CY


def seg_and_lines(k: int, x: np.ndarray) -> list[bytes]:
  """The stop-line model's two messages for frame k, as dioramaweb/stopline.py sends them: an empty query map (the
  model ran, in 18 ms), and the stop line."""
  qmap = np.full((128, 256), 255, np.uint8)  # ds 2
  seg = packets.seg(k, 18.0, qmap.astype(np.float32), np.zeros(0, np.uint8), np.zeros(0))
  lines = []
  rel = STOP_AT - x[k]
  if rel > 0:
    # the line across the ego lane, on the curving road there, where the narrow frame still sees it
    y0 = CURVE * rel ** 2
    (u1, v1), (u2, v2) = pixel(rel, y0 - 1.8 - OVERRUN[0], HEIGHT), pixel(rel, y0 + 1.8 + OVERRUN[1], HEIGHT)
    t0, t1 = max(0.0, (0 - u1) / (u2 - u1)), min(1.0, (511 - u1) / (u2 - u1))  # cut at the frame's sides
    u1, v1, u2, v2 = u1 + t0 * (u2 - u1), v1 + t0 * (v2 - v1), u1 + t1 * (u2 - u1), v1 + t1 * (v2 - v1)
    if max(v1, v2) < 250 and rel < 45:
      lines.append([0.8, u1, v1, u2, v2])
  return [seg, stoplines(k, False, lines)]


def main(out: Path) -> None:
  v, x = profile()
  meta = {
    'version': 1, 'fps': FPS, 'frame': {'w': 512, 'h': 256, 'f': F, 'cx': CX, 'cy': CY}, 'height': HEIGHT, 'live': False,
    'drive': [[n, c] for n, c in FIELDS],
    'seg': {'labels': [], 'road_class': {}, 'things': {}, 'ds': 2, 'narrow_only': True},
    'source': {'route': 'synthetic'},
  }
  out.parent.mkdir(parents=True, exist_ok=True)
  with open(out, 'wb') as fh:
    packets.write(fh, packets.meta(meta))
    for k in range(int(SECONDS * FPS)):
      packets.write(fh, packets.drive(k, k / FPS, 0.0, heads(k, v, x)))
      if k % 2 == 0:
        for msg in seg_and_lines(k, x):
          packets.write(fh, msg)
  print(f'{int(SECONDS * FPS)} frames, stopped {STOP_AT - x[int(20 * FPS)]:.1f} m short of the line: {out}')


if __name__ == '__main__':
  main(Path(sys.argv[1]))
