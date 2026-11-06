"""comma1M access: segment metadata and openpilot's medmodel warp, using the dataset's own calibration."""
from __future__ import annotations
import os
import numpy as np
import av
import cv2
from safetensors.numpy import load_file

from paths import COMMA1M as ROOT

MED_FL, MED_CY, MED_W, MED_H = 910.0, 47.6, 512, 256
MED_K = np.array([[MED_FL, 0, MED_W / 2], [0, MED_FL, MED_CY], [0, 0, 1]])
# narrow-camera intrinsics by fcamera width (openpilot common/transformations/camera.py)
FOCAL = {1928: 2648.0, 1344: 1522.0 * 3 / 4, 1164: 910.0}
# the wide (fisheye) camera, as openpilot's pinhole approximation: only the driving model's second input (ego_c1m.py)
WIDE_FOCAL = {1928: 567.0, 1344: 567.0 * 3 / 4}
VIEW_FROM_DEVICE = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])


def rot_from_euler(rpy):
  r, p, y = rpy
  cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
  rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
  ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
  rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
  return rz @ ry @ rx


class Segment:
  def __init__(self, sid: str):
    self.sid = sid
    d = f'{ROOT}/{sid}'
    self.loc = load_file(f'{d}/localizer.safetensors')
    self.info = load_file(f'{d}/frame_info.safetensors')
    self.w = int(self.info['fcamera/width'][0])
    self.h = int(self.info['fcamera/height'][0])
    self.device = int(self.info['device_type'])
    self.rpy = self.loc['rpy']
    self.video = f'{d}/fcamera.hevc'
    self.wide_video = f'{d}/ecamera.hevc' if os.path.exists(f'{d}/ecamera.hevc') else None

  @property
  def speed(self):
    return np.linalg.norm(self.loc['frame_states'][:, 7:10], axis=1)

  def camera_from_model(self, scale: float = 1.0) -> np.ndarray:
    """camera pixel <- medmodel pixel (scaled by `scale`), as openpilot's get_warp_matrix."""
    f = FOCAL[self.w]
    k = np.array([[f, 0, self.w / 2], [0, f, self.h / 2], [0, 0, 1]])
    calib_from_med = np.linalg.inv(MED_K @ VIEW_FROM_DEVICE)
    m = k @ VIEW_FROM_DEVICE @ rot_from_euler(self.rpy) @ calib_from_med
    return m @ np.diag([1 / scale, 1 / scale, 1])

  def frames(self, wanted):
    """{index: RGB frame} for the wanted frame indices, decoding the segment in order."""
    wanted = set(wanted); out = {}
    with av.open(self.video) as c:
      for i, fr in enumerate(c.decode(video=0)):
        if i in wanted:
          out[i] = fr.to_ndarray(format='rgb24')
          if len(out) == len(wanted):
            break
    return out

  def warp(self, img, scale: float = 1.0):
    w, h = int(MED_W * scale), int(MED_H * scale)
    return cv2.warpPerspective(img, self.camera_from_model(scale), (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_AREA,
                               borderMode=cv2.BORDER_CONSTANT)


def segments():
  return sorted(os.listdir(ROOT))
