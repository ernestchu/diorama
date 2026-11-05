"""
Copyright (c) 2026-, Ernie Chu.

This file is part of diorama and is licensed under the MIT License.

A recorded drive through dioramad's own path, without openpilot: the camera
frame as NV12 into diorama.warp, the driving model's output through
openpilot's parse into what modelV2 carries, and that through diorama.heads
and diorama.link to the phone. What a comma's dioramad does, less VisionIpc
and the messages.

    uv run python -m commasim.as_dioramad <segments> --model <driving model .onnx>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the diorama package beside sim/

from commasim.driving import DrivingModel  # noqa: E402
from commasim.outputs import parse  # noqa: E402
from commasim.route import FPS, Route, warp_matrix  # noqa: E402
from diorama import heads  # noqa: E402
from diorama.link import Viewer  # noqa: E402
from diorama.warp import Warp  # noqa: E402


class Log:
  @staticmethod
  def warning(msg):
    print(msg, flush=True)


def model_v2(p):
  """What openpilot's fill_model_msg puts in modelV2 from the parsed heads, as far as heads.encode reads it."""
  plan, std = p['plan'], p['plan_stds']

  def xyz(c, with_std):
    extra = {'xStd': std[:, c], 'yStd': std[:, c + 1], 'zStd': std[:, c + 2]} if with_std else {}
    return NS(x=plan[:, c], y=plan[:, c + 1], z=plan[:, c + 2], **extra)

  def line(a, i):
    return NS(y=a[i, :, 0], z=a[i, :, 1])

  return NS(
    position=xyz(0, True), velocity=xyz(3, False), acceleration=xyz(6, False), orientation=xyz(9, False),
    orientationRate=xyz(12, False),
    laneLines=[line(p['lane_lines'], i) for i in range(4)], laneLineStds=p['lane_lines_stds'][:, 0, 0],
    laneLineProbs=p['lane_lines_prob'][1::2],
    roadEdges=[line(p['road_edges'], i) for i in range(2)], roadEdgeStds=p['road_edges_stds'][:, 0, 0],
    leadsV3=[NS(x=l[:, 0], y=l[:, 1], v=l[:, 2], a=l[:, 3], xStd=s[:, 0], yStd=s[:, 1], vStd=s[:, 2], aStd=s[:, 3], prob=pr)
             for l, s, pr in zip(p['lead'], p['lead_stds'], p['lead_prob'], strict=True)])


def nv12(yuv: np.ndarray, w: int, h: int) -> np.ndarray:
  """A planar I420 frame as camerad's NV12, rows unpadded."""
  out = np.empty(w * h * 3 // 2, np.uint8)
  out[:w * h] = yuv[:w * h]
  uv = out[w * h:].reshape(h // 2, w)
  uv[:, 0::2] = yuv[w * h:w * h * 5 // 4].reshape(h // 2, w // 2)
  uv[:, 1::2] = yuv[w * h * 5 // 4:].reshape(h // 2, w // 2)
  return out


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('segments', nargs='+', type=Path)
  ap.add_argument('--model', required=True)
  ap.add_argument('--device', default='auto')
  ap.add_argument('--listen', default='127.0.0.1')
  ap.add_argument('--port', type=int, default=5600)
  ap.add_argument('--start', type=float, default=0.0)
  args = ap.parse_args()

  route = Route.open(args.segments)
  model = DrivingModel(args.model, args.device)
  cal = route.calibration
  viewer = Viewer(host=args.listen, port=args.port, name='dioramad (Mac)', log=Log)
  viewer.camera = {'height': cal.height, 'wide_from_device_euler': list(cal.wide_from_device)}
  w, h = route.size
  warp = Warp(warp_matrix(cal.rpy, route.narrow_fl, route.size, big=False), w, h, w, w * h)
  while True:
    while not viewer.watched:
      time.sleep(0.2)
    model.reset()
    t0, n = time.monotonic(), 0
    for k, narrow, warped in route.warped(int(args.start * FPS)):
      if not viewer.watched:
        break
      p = parse(model.step(warped), model.slices)
      odo = NS(trans=p['pose'][:3], rot=p['pose'][3:], transStd=p['pose_stds'][:3], rotStd=p['pose_stds'][3:])
      if (wait := t0 + n / FPS - time.monotonic()) > 0:
        time.sleep(wait)
      viewer.show(k, warp(nv12(narrow, w, h)), heads.encode(model_v2(p), odo))
      n += 1
      if n % (5 * FPS) == 0:
        print(f'frame {k}: {n / (time.monotonic() - t0):.1f} Hz, {viewer.sent} sent', flush=True)


if __name__ == '__main__':
  main()
