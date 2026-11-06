"""Cutting a labelled stop line to the ego lane, in the narrow model frame's pixels.

The driving model's lane lines either side of the car (ego_c1m.py) are projected into the frame, with their own road
height; the line is cut where it crosses them. A boundary is the road edge instead where that lies inside the lane
line. Boundaries that do not reach the line's row are extended along their last stretch.
"""
import numpy as np
from cams import FrameCam

X_IDXS = 192.0 * (np.arange(33) / 32) ** 2
MIN_LEN = 8.0     # [px] shorter than this after cutting: the line is not in the ego lane
CAM = FrameCam()


def boundary(rec, side):
  """Ego-lane boundary as road points (33, 3): side 0 left, 1 right."""
  lane, edge = np.array(rec['lanes'][1 + side]), np.array(rec['edges'][side])
  inner = lane[:, 0] if side else -lane[:, 0]
  edge_inner = edge[:, 0] if side else -edge[:, 0]
  use_edge = edge_inner < inner
  yz = np.where(use_edge[:, None], edge, lane)
  return np.column_stack([X_IDXS, yz])


def image_curve(pts):
  """Road points -> pixels (N, 2) ahead of the camera, ordered near to far."""
  uv, depth = CAM.pixel(pts)
  return uv[(depth > 0.5) & (pts[:, 0] >= 1.0)]


def cross(line, curve):
  """Where the curve crosses the line's infinite extension: t along the line (0 at its first end, 1 at its second)."""
  a, b = np.array(line[:2], float), np.array(line[2:], float)
  d = b - a
  n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-6)
  s = (curve - a) @ n
  for i in range(len(s) - 1):
    if s[i] == 0 or s[i] * s[i + 1] < 0:
      p = curve[i] + (curve[i + 1] - curve[i]) * s[i] / (s[i] - s[i + 1])
      return float((p - a) @ d / (d @ d))
  # not reached: extend the nearer end's last stretch (the curve's two points closest to the line)
  i = int(np.argmin(np.abs(s)))
  j = i + 1 if i + 1 < len(s) else i - 1
  if j < 0 or s[i] == s[j]:
    return None
  p = curve[i] + (curve[j] - curve[i]) * s[i] / (s[i] - s[j])
  return float((p - a) @ d / (d @ d))


def trim(line, rec):
  """(line cut to the ego lane or None if it does not reach it, reason, the boundaries in pixels)."""
  curves = [image_curve(boundary(rec, s)) for s in (0, 1)]
  if any(len(c) < 2 for c in curves):
    return None, 'no lane', curves
  ts = [cross(line, c) for c in curves]
  if any(t is None for t in ts):
    return None, 'no crossing', curves
  # left end first: the line's own direction may run either way
  lo, hi = min(ts), max(ts)
  t0, t1 = max(lo, 0.0), min(hi, 1.0)
  if t1 <= t0:
    return None, 'outside lane', curves
  a, b = np.array(line[:2], float), np.array(line[2:], float)
  p, q = a + (b - a) * t0, a + (b - a) * t1
  if np.linalg.norm(q - p) < MIN_LEN:
    return None, 'too short', curves
  return [round(float(v), 2) for v in (*p, *q)], 'ok', curves
