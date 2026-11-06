"""
Copyright (c) 2026-, Zeph Leggett.

This file comes from jetlink's Diorama (MIT) and is part of diorama, under the
same license: see the LICENSE file in the root directory.

A recorded drive through the driving model and the stop-line model, into a
folder the Diorama page plays beside the drive's own footage:

    drive.diorama   the messages the page reads (packets.py)
    footage.mp4     the road camera over the same frames, for the side-by-side

The frames are warped and packed as a comma sends them, with the calibration
the comma logged, so the models see what they would see live.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

from commasim.driving import DrivingModel
from commasim.outputs import DRIVE_FIELDS
from commasim.route import FPS, Route
from dioramaweb import packets
from dioramaweb.stopline import StoplineModels

WARMUP = 100  # frames of history the driving model gets before the first one kept


def drive_meta(model: DrivingModel, models: StoplineModels, route: Route, start: int) -> dict:
  m, cal = models.meta, route.calibration
  return {
    'version': 1, 'fps': FPS, 'frame': m['frame'], 'wide': dict(packets.wide_frame(m['frame']), from_device=list(cal.wide_from_device)),
    'height': cal.height, 'live': False,
    'drive': [[name, model.slices[name].stop - model.slices[name].start] for name in DRIVE_FIELDS if name in model.slices],
    'seg': {k: m['seg'][k] for k in ('labels', 'road_class', 'things', 'ds', 'narrow_only') if k in m['seg']},
    'source': {'route': route.segments[0].name.rsplit('--', 1)[0], 'start': start, 'rpy': list(cal.rpy), 'camera': list(route.size)},
  }


def footage_encoder(path: Path, size: tuple[int, int], width: int = 960) -> subprocess.Popen:
  w, h = size
  return subprocess.Popen(
    ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-',
     '-vf', f'scale={width}:-2', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '26', '-pix_fmt', 'yuv420p',
     '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)


def run(paths: list[Path], onnx: str, stopline: Path, out: Path, start_s: float = 0.0, seconds: float | None = None,
        device: str = 'auto', every: int = 2, rhd: bool = False) -> Path:
  """`every`: the stop-line model sees every this many frames, as a phone keeps up."""
  route = Route.open(paths)  # with the calibration the comma logged, or refused
  print(f'{len(route.segments)} segments of {route.segments[0].name.rsplit("--", 1)[0]}, camera {route.size[0]}x{route.size[1]}')
  model = DrivingModel(onnx, device, rhd)
  models = StoplineModels(stopline, device)
  start = int(round(start_s * FPS))
  count = None if seconds is None else int(round(seconds * FPS))
  first = max(start - WARMUP, 0)

  out.mkdir(parents=True, exist_ok=True)
  enc = footage_encoder(out / 'footage.mp4', route.size)
  model.reset()
  timing = {'driving': 0.0, 'stopline': 0.0}
  n = 0
  t0 = time.perf_counter()
  with open(out / 'drive.diorama', 'wb') as fh:
    packets.write(fh, packets.meta(drive_meta(model, models, route, start)))
    for k, narrow_yuv, warped in route.warped(first, None if count is None else start - first + count):
      t = time.perf_counter()
      vector = model.step(warped)
      model_s = time.perf_counter() - t
      timing['driving'] += model_s
      if k < start:
        continue
      frame = k - start
      enc.stdin.write(narrow_yuv.tobytes())
      values = np.concatenate([vector[model.slices[name]] for name in DRIVE_FIELDS if name in model.slices])
      packets.write(fh, packets.drive(frame, frame / FPS, 1000 * model_s, values))
      if frame % every == 0:
        msgs, s = models.messages(frame, warped[0])  # the narrow frame
        timing['stopline'] += s
        for msg in msgs:
          packets.write(fh, msg)
      n += 1
      if n % 200 == 0:
        print(f'{n / FPS:.0f} s of drive in {time.perf_counter() - t0:.0f} s', flush=True)
  enc.stdin.close()
  enc.wait()
  if not n:
    raise SystemExit(f'the drive has no frames from {start_s} s on')
  print(f'{n} frames ({n / FPS:.1f} s): driving model {1000 * timing["driving"] / n:.1f} ms a frame, '
        f'stop-line model {1000 * timing["stopline"] / max(-(-n // every), 1):.1f} ms')
  print(f'wrote {out}/drive.diorama and footage.mp4; play it with: python -m dioramaweb view {out}')
  return out
