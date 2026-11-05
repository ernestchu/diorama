"""
Copyright (c) 2026-, Ernie Chu.

This file is part of diorama and is licensed under the MIT License.

The narrow camera's frame as modeld warps it for the driving model, on the
CPU: the medmodel frame (512x256, f 910, cy 47.6) by nearest neighbour,
rounded and clipped to the image as openpilot's warp_perspective_tinygrad
does, then packed into the six planes the link carries (four Y sub-planes,
U, V). The stop-line model was trained on exactly this.

The sample positions depend only on the calibration, so they are worked out
once per calibration and each frame is three gathers.
"""
from __future__ import annotations

import numpy as np

MODEL_W, MODEL_H = 512, 256
MEDMODEL_FL, MEDMODEL_CY = 910.0, 47.6
VIEW_FROM_DEVICE = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])


def rot_from_euler(rpy) -> np.ndarray:
  r, p, y = rpy
  cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
  rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
  ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
  rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
  return rz @ ry @ rx


def warp_matrix(rpy, intrinsics: np.ndarray) -> np.ndarray:
  """Model-frame pixel -> camera pixel: openpilot's get_warp_matrix for the medmodel frame."""
  model_from_calib = np.array([[MEDMODEL_FL, 0.0, MODEL_W / 2], [0.0, MEDMODEL_FL, MEDMODEL_CY], [0.0, 0.0, 1.0]]) @ VIEW_FROM_DEVICE
  return intrinsics @ VIEW_FROM_DEVICE @ rot_from_euler(rpy) @ np.linalg.inv(model_from_calib)


def _nearest(M: np.ndarray, w_dst: int, h_dst: int, w_src: int, h_src: int, row: int) -> np.ndarray:
  """Flat indices into a source of row length `row` for each destination pixel."""
  x, y = np.meshgrid(np.arange(w_dst, dtype=np.float32), np.arange(h_dst, dtype=np.float32))
  M = M.astype(np.float32)
  sw = M[2, 0] * x + M[2, 1] * y + M[2, 2]
  sx = np.clip(np.round((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / sw), 0, w_src - 1).astype(np.int64)
  sy = np.clip(np.round((M[1, 0] * x + M[1, 1] * y + M[1, 2]) / sw), 0, h_src - 1).astype(np.int64)
  return (sy * row + sx).ravel()


class Warp:
  """One camera's NV12 frames (`stride` bytes a row, the interleaved UV plane at `uv_offset`) into packed frames."""

  def __init__(self, M: np.ndarray, width: int, height: int, stride: int, uv_offset: int):
    self.y_idx = _nearest(M, MODEL_W, MODEL_H, width, height, stride)
    # the chroma planes at half resolution, as M_inv * [[1, 1, .5], [1, 1, .5], [2, 2, 1]] in openpilot
    M_uv = M * np.array([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [2.0, 2.0, 1.0]])
    uv = _nearest(M_uv, MODEL_W // 2, MODEL_H // 2, width // 2, height // 2, stride // 2)
    # uv pairs: (row, col) of the half-resolution plane -> bytes uv_offset + row * stride + 2 * col (+1 for V)
    row, col = np.divmod(uv, stride // 2)
    self.u_idx = uv_offset + row * stride + 2 * col
    self.v_idx = self.u_idx + 1
    self.out = np.empty((6, MODEL_H // 2, MODEL_W // 2), np.uint8)

  def __call__(self, nv12: np.ndarray) -> np.ndarray:
    """The packed (6, 128, 256) frame, in a buffer this Warp reuses."""
    y = nv12[self.y_idx].reshape(MODEL_H, MODEL_W)
    o = self.out
    o[0], o[1], o[2], o[3] = y[0::2, 0::2], y[1::2, 0::2], y[0::2, 1::2], y[1::2, 1::2]
    o[4] = nv12[self.u_idx].reshape(MODEL_H // 2, MODEL_W // 2)
    o[5] = nv12[self.v_idx].reshape(MODEL_H // 2, MODEL_W // 2)
    return o
