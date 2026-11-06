"""
Copyright (c) 2026-, Ernie Chu.

This file is part of diorama and is licensed under the MIT License.

dioramad: on the comma, the drive for an iPhone on the cable. It reads what
openpilot already publishes (the narrow camera through VisionIpc, modelV2 and
cameraOdometry, liveCalibration), warps each camera frame on the CPU while a
phone watches, and sends it with the driving model's outputs for that frame
(heads.encode). It never touches modeld: it runs as its own process, at a
low priority, and does nothing but listen while no phone is there.

zoompilot's manager runs it onroad while the Diorama setting is on; the phone
reaches it over jetlink's iOS gadget (the cable's network), on port 5600.
"""
from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np

from diorama import heads
from diorama.link import Viewer
from diorama.warp import Warp, warp_matrix

# camera frames kept warped while their model frame is on its way (modeld is a frame or two behind the camera)
KEEP = 8
NICE = 10


def main() -> None:
  from msgq.visionipc import VisionIpcClient
  from openpilot.common.swaglog import cloudlog
  from openpilot.common.transformations.camera import DEVICE_CAMERAS
  try:  # zoompilot keeps cereal inside the openpilot package
    import openpilot.cereal.messaging as messaging
    from openpilot.cereal.services import SERVICE_LIST
    from openpilot.cereal.visionipc import VisionStreamType
  except ImportError:  # stock openpilot's layout
    from cereal import messaging
    from cereal.services import SERVICE_LIST
    from msgq.visionipc import VisionStreamType
  # calibrationd's message: extrinsicsCalibration in zoompilot, liveCalibration in stock openpilot; the same fields
  calibration = 'extrinsicsCalibration' if 'extrinsicsCalibration' in SERVICE_LIST else 'liveCalibration'

  os.nice(NICE)  # the car's processes first
  viewer = Viewer(log=cloudlog)
  sm = messaging.SubMaster(['modelV2', 'cameraOdometry', calibration, 'deviceState', 'narrowRoadCameraState'])
  vipc = VisionIpcClient('camerad', VisionStreamType.VISION_STREAM_NARROW_ROAD, True)

  warp: Warp | None = None
  geometry = None  # (rpy, width, height, stride, uv_offset) the warp was made for
  rpy = None
  warped: OrderedDict[int, np.ndarray] = OrderedDict()
  odometry = None
  connected = False  # msgq's Python VisionIpcClient does not say; connect() does, once

  while True:
    if not connected:
      connected = vipc.connect(False)
      if not connected:
        sm.update(100)
        continue
    buf = vipc.recv(50)
    sm.update(0)

    if sm.updated[calibration]:
      cal = sm[calibration]
      if str(cal.calStatus) == 'calibrated' and len(cal.rpyCalib) == 3 and len(cal.height) == 1:
        rpy = tuple(float(v) for v in cal.rpyCalib)
        viewer.camera = {'height': float(cal.height[0]), 'wide_from_device_euler': [float(v) for v in cal.wideFromDeviceEuler]}
    if sm.updated['cameraOdometry']:
      odometry = sm['cameraOdometry']

    if buf is not None and viewer.watched and rpy is not None and sm.seen['deviceState'] and sm.seen['narrowRoadCameraState']:
      g = (rpy, buf.width, buf.height, buf.stride, buf.uv_offset)
      if g != geometry:
        dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['narrowRoadCameraState'].sensor))]
        warp = Warp(warp_matrix(rpy, dc.narrow_road.intrinsics), buf.width, buf.height, buf.stride, buf.uv_offset)
        geometry = g
        warped.clear()
      warped[vipc.frame_id] = warp(np.asarray(buf.data)).copy()
      while len(warped) > KEEP:
        warped.popitem(last=False)

    if sm.updated['modelV2'] and odometry is not None:
      model = sm['modelV2']
      packed = warped.pop(model.frameId, None)
      if packed is not None:
        viewer.show(model.frameId, packed, heads.encode(model, odometry))


if __name__ == '__main__':
  main()
