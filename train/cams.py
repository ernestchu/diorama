"""The model frame (openpilot's medmodel frame, the narrow camera) and how its pixels meet the road. Points are in
the calibrated frame: forward, right, down [m], the camera at the origin and the road HGT below it.
"""
import numpy as np

from c1m import MED_FL, MED_CY, MED_W, VIEW_FROM_DEVICE

HGT = 1.25  # camera height above the road [m]


class FrameCam:
  def __init__(self, hgt=HGT):
    self.hgt = hgt
    self.f, self.cx, self.cy = MED_FL, MED_W / 2, MED_CY
    k = np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1]])
    self.proj = k @ VIEW_FROM_DEVICE
    self.inv = np.linalg.inv(self.proj)

  def ground(self, u, v):
    """Pixels -> road points (N, 3); NaN where the ray does not go down to the road."""
    p = np.stack(np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float)), -1).reshape(-1, 2)
    d = np.concatenate([p, np.ones((len(p), 1))], 1) @ self.inv.T
    t = np.where(d[:, 2] > 1e-6, self.hgt / np.maximum(d[:, 2], 1e-6), np.nan)
    return d * t[:, None]

  def pixel(self, pts):
    """Points (N, 3) -> (pixels (N, 2), depth along the view axis (N,))."""
    q = np.asarray(pts, float).reshape(-1, 3) @ self.proj.T
    with np.errstate(divide='ignore', invalid='ignore'):
      return q[:, :2] / q[:, 2:3], q[:, 2]

  def line_dist(self, line, at=None):
    """Forward distance [m] of a line [x1, y1, x2, y2] at column `at` (clamped to the line), default its middle."""
    x1, y1, x2, y2 = line
    t = 0.5 if at is None or abs(x2 - x1) < 1e-3 else min(1.0, max(0.0, (at - x1) / (x2 - x1)))
    g = self.ground(x1 + t * (x2 - x1), y1 + t * (y2 - y1))[0]
    return float(g[0]) if np.isfinite(g[0]) else float('inf')
