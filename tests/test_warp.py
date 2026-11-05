"""The comma's CPU warp against the warp the stop-line model's frames were made with (commasim.route.pack, OpenCV)."""
import numpy as np

from commasim.route import FOCALS, pack, warp_matrix as route_warp
from diorama.warp import Warp, warp_matrix


def test_matches_the_training_warp():
  w, h = 1344, 760
  fl = FOCALS[(w, h)][0]
  rpy = (7.011e-05, 0.043111, 0.014626)
  intrinsics = np.array([[fl, 0, w / 2], [0, fl, h / 2], [0, 0, 1]])
  M = warp_matrix(rpy, intrinsics)
  np.testing.assert_allclose(M, route_warp(rpy, fl, (w, h), big=False), rtol=1e-9)

  # a smooth picture, so a pixel off by one rounding differs by little
  yy, xx = np.mgrid[0:h, 0:w]
  y = (128 + 100 * np.sin(xx / 37.0) * np.cos(yy / 23.0)).astype(np.uint8)
  u = (128 + 60 * np.sin(xx[::2, ::2] / 19.0)).astype(np.uint8)
  v = (128 + 60 * np.cos(yy[::2, ::2] / 13.0)).astype(np.uint8)
  want = pack(np.concatenate([y.ravel(), u.ravel(), v.ravel()]), M, (w, h))

  stride = 1408  # padded rows, as camerad's buffers have
  nv12 = np.zeros(stride * h + stride * h // 2, np.uint8)
  nv12[:stride * h].reshape(h, stride)[:, :w] = y
  uv = nv12[stride * h:].reshape(h // 2, stride)
  uv[:, 0:w:2], uv[:, 1:w:2] = u, v
  got = Warp(M, w, h, stride, stride * h)(nv12)

  assert got.shape == want.shape == (6, 128, 256)
  same = np.mean(got == want)
  close = np.mean(np.abs(got.astype(int) - want.astype(int)) <= 3)
  assert same > 0.98, same
  assert close > 0.999, close
