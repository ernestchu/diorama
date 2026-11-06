"""Carrying a labelled stop line from one frame of a comma1M segment to another.

A line drawn in frame i is lifted onto the road (flat ground, camera HGT above it) in frame i's calibrated
frame, moved into frame j with the localizer's poses, and projected into frame j's image. Everything is in the
model frame's pixels (512x256) (cams.FrameCam).
"""
import numpy as np
from scipy.spatial.transform import Rotation

from c1m import rot_from_euler, MED_W, MED_H
from cams import FrameCam


class Poses:
  def __init__(self, seg, cam=None):
    fs = seg.loc['frame_states']
    self.pos = fs[:, 0:3]
    self.rot = Rotation.from_quat(np.roll(fs[:, 3:7], -1, axis=1)).as_matrix()  # device -> ECEF, stored w, x, y, z
    self.dev_from_calib = rot_from_euler(seg.rpy)
    self.cam = cam or FrameCam()

  def lift(self, i, pts):
    """Pixels (N, 2) in frame i -> ECEF points on the road. Points at or above the horizon -> NaN."""
    pts = np.asarray(pts, float)
    cal = self.cam.ground(pts[:, 0], pts[:, 1])  # forward, right, down
    if (cal[:, 0] <= 1e-3).any(): cal[:] = np.nan
    dev = cal @ self.dev_from_calib.T
    return self.pos[i] + dev @ self.rot[i].T

  def project(self, j, ecef):
    """ECEF points -> (pixels (N, 2), forward distance (N,)) in frame j."""
    dev = (ecef - self.pos[j]) @ self.rot[j]
    cal = dev @ self.dev_from_calib
    uv, _ = self.cam.pixel(cal)
    return uv, cal[:, 0]


def carry(poses, i, j, line, open_ends=(False, False)):
  """A line [x1, y1, x2, y2] drawn in frame i, as seen in frame j, clipped to the image. An end that touched the
  frame edge in frame i (open) is extended to the image edge in frame j, since the line continued past it.
  Returns the line or None (behind the camera, too close, or out of view)."""
  g = poses.lift(i, [line[:2], line[2:]])
  if np.isnan(g).any():
    return None
  uv, x = poses.project(j, g)
  if (x < 4.0).any():
    return None
  (ax, ay), (bx, by) = uv
  if open_ends[0] or open_ends[1]:
    d = np.array([bx - ax, by - ay]); n = np.linalg.norm(d)
    if n < 1e-6: return None
    d /= n
    if open_ends[0]: ax, ay = ax - d[0] * 2000, ay - d[1] * 2000
    if open_ends[1]: bx, by = bx + d[0] * 2000, by + d[1] * 2000
  return clip(ax, ay, bx, by)


def clip(x1, y1, x2, y2):
  """Liang-Barsky against the 512x256 image."""
  t0, t1, dx, dy = 0.0, 1.0, x2 - x1, y2 - y1
  for p, q in ((-dx, x1), (dx, MED_W - x1), (-dy, y1), (dy, MED_H - y1)):
    if p == 0:
      if q < 0: return None
    else:
      t = q / p
      if p < 0: t0 = max(t0, t)
      else: t1 = min(t1, t)
  if t0 > t1: return None
  return [x1 + t0 * dx, y1 + t0 * dy, x1 + t1 * dx, y1 + t1 * dy]


def open_ends(line, tol=2.0):
  """Which ends of a labelled line touch the left or right edge of the frame."""
  return tuple(x <= tol or x >= MED_W - tol for x in (line[0], line[2]))


def row_at(line, x):
  x1, y1, x2, y2 = line
  return (y1 + y2) / 2 if abs(x2 - x1) < 1e-6 else y1 + (x - x1) / (x2 - x1) * (y2 - y1)
