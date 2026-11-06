"""
Copyright (c) 2026-, Ernie Chu.

This file is part of diorama and is licensed under the MIT License.

The drive DioramaKit's parity tests hold the app to the page with: a
drive.diorama made, as dioramaweb's run makes one, from a drive's driving-model
outputs (drive_outputs.npz: frame, outputs, fields, counts, height) and its
packed narrow frames (stops.npz: packed, frame), with no video. The stop-line
model (its ONNX) sees every other frame that has a narrow frame. The frames
keep the drive's own numbers, so a gap between stretches starts the scene
over, as a seek does.

Made, not committed: the frames come from a personal drive.

    cd web && ../sim/.venv/bin/python tools/parity_drive.py ../models/stops ../models/stopline_pilot.onnx ../models/parity-drive
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent / 'sim')]

from dioramaweb import packets  # noqa: E402
from dioramaweb.stopline import StoplineModels  # noqa: E402

FPS = 20


def main(src: Path, onnx: Path, out: Path) -> None:
  drive = np.load(src / 'drive_outputs.npz')
  stops = np.load(src / 'stops.npz')
  packed = {int(f): p for f, p in zip(stops['frame'], stops['packed'], strict=True)}
  models = StoplineModels(onnx, 'cpu')
  fields = [[str(n), int(c)] for n, c in zip(drive['fields'], drive['counts'], strict=True)]
  meta = {
    'version': 1, 'fps': FPS, 'frame': models.meta['frame'], 'height': float(drive['height']), 'live': False,
    'drive': fields, 'seg': models.meta['seg'], 'source': {'route': src.name},
  }
  out.mkdir(parents=True, exist_ok=True)
  lines = 0
  with open(out / 'drive.diorama', 'wb') as fh:
    packets.write(fh, packets.meta(meta))
    for frame, values in zip(drive['frame'], drive['outputs'], strict=True):
      frame = int(frame)
      packets.write(fh, packets.drive(frame, frame / FPS, 0.0, values.astype(np.float32)))
      if frame % 2 == 0 and frame in packed:
        msgs, _ = models.messages(frame, packed[frame])
        for msg in msgs:
          packets.write(fh, msg)
        lines += msgs[-1][6]  # the L message's line count
  print(f'{len(drive["frame"])} frames, {lines} stop-line sightings: {out / "drive.diorama"}')


if __name__ == '__main__':
  main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
