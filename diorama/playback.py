"""
Copyright (c) 2026-, Ernie Chu.

This file is part of diorama and is licensed under the MIT License.

A drive for dioramad without a car, on the comma itself: a segment's logged
modelV2, cameraOdometry, calibration and narrow camera state played at the
speed they were logged, and one still camera frame served over VisionIpc as
camerad would serve each frame, with the logged frame ids. dioramad's work
(the warp, the encoding, the send) is the same on any picture, so this gives
its real cost and real plumbing; the iPhone draws the drive's path and lanes
over a frozen picture. Only offroad, where nothing else publishes these.

    python -m diorama.playback <segment>/rlog.zst <frame.nv12> <width> <height> [--seconds N]

The frame is raw NV12, rows unpadded (ffmpeg -pix_fmt nv12 -f rawvideo).
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def main() -> None:
  import openpilot.cereal.messaging as messaging
  from msgq.visionipc import VisionIpcServer
  from openpilot.cereal.services import SERVICE_LIST
  from openpilot.cereal.visionipc import VisionStreamType
  from openpilot.tools.lib.logreader import LogReader

  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('rlog')
  ap.add_argument('frame')
  ap.add_argument('width', type=int)
  ap.add_argument('height', type=int)
  ap.add_argument('--seconds', type=float, default=120.0, help='stop after this long (default 120)')
  args = ap.parse_args()

  w, h = args.width, args.height
  frame = np.fromfile(args.frame, np.uint8)
  assert frame.size == w * h * 3 // 2, f'{args.frame} is {frame.size} bytes, not a {w}x{h} NV12 frame'

  calibration = 'extrinsicsCalibration' if 'extrinsicsCalibration' in SERVICE_LIST else 'liveCalibration'
  wanted = {'modelV2', 'cameraOdometry', 'narrowRoadCameraState', 'liveCalibration', 'extrinsicsCalibration'}
  events = [m for m in LogReader(args.rlog) if m.which() in wanted]
  assert events, f'{args.rlog} has none of {sorted(wanted)}'
  print(f'{len(events)} messages over {(events[-1].logMonoTime - events[0].logMonoTime) / 1e9:.1f} s', flush=True)

  pm = messaging.PubMaster(['modelV2', 'cameraOdometry', 'narrowRoadCameraState', calibration])
  server = VisionIpcServer('camerad')
  stream = VisionStreamType.VISION_STREAM_NARROW_ROAD
  server.create_buffers_with_sizes(stream, 4, w, h, frame.size, w, w * h)
  server.start_listener()

  stop = time.monotonic() + args.seconds
  sent = 0
  while time.monotonic() < stop:
    t0, l0 = time.monotonic(), events[0].logMonoTime
    for m in events:
      if (wait := t0 + (m.logMonoTime - l0) / 1e9 - time.monotonic()) > 0:
        time.sleep(wait)
      if time.monotonic() >= stop:
        break
      which = m.which()
      if which == 'narrowRoadCameraState':
        server.send(stream, frame, m.narrowRoadCameraState.frameId)
        sent += 1
        if sent % 200 == 0:
          print(f'{sent} frames', flush=True)
      # a log from stock openpilot names calibrationd's message liveCalibration; send it under this fork's name
      name = calibration if which in ('liveCalibration', 'extrinsicsCalibration') else which
      builder = m.as_builder()
      if name != which:
        continue  # a renamed union member cannot be re-tagged in place; this fork's own logs never need it
      pm.send(name, builder)
    print('the segment ended; again from its start', flush=True)


if __name__ == '__main__':
  main()
