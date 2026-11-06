"""
Copyright (c) 2026-, Zeph Leggett.

This file comes from jetlink's Diorama (MIT) and is part of diorama, under the
same license: see the LICENSE file in the root directory.

    python -m dioramaweb run <route>--* --model driving.onnx --stopline stopline_pilot.onnx --out drive/
    python -m dioramaweb view drive/
    python -m dioramaweb camera   # set the page's camera in the browser
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(prog='python -m dioramaweb', description='Diorama on a recorded drive, in the browser.')
  sub = ap.add_subparsers(dest='cmd', required=True)

  r = sub.add_parser('run', help='run a recorded drive through the driving model and the stop-line model')
  r.add_argument('segments', type=Path, nargs='+',
                 help="the drive's segment folders (fcamera.hevc, ecamera.hevc and rlog.zst, whose calibration is used), or a folder of them")
  r.add_argument('--model', required=True, help='the driving model ONNX')
  r.add_argument('--stopline', type=Path, required=True, help="the stop-line model's ONNX (train/export.py's stopline_pilot.onnx)")
  r.add_argument('--out', type=Path, required=True)
  r.add_argument('--start', type=float, default=0.0, help='seconds into the drive')
  r.add_argument('--seconds', type=float, help='how much of it (default: to the end)')
  r.add_argument('--device', choices=('auto', 'cuda', 'coreml', 'cpu'), default='auto')
  r.add_argument('--rhd', action='store_true', help='left-hand traffic (the driver sits on the right)')

  v = sub.add_parser('view', help='play a drive `run` wrote, in the browser')
  v.add_argument('drive', type=Path)
  v.add_argument('--port', type=int, default=8642)
  v.add_argument('--no-browser', action='store_true')
  v.add_argument('--camera', type=Path, help='the camera file `camera` saved (default: ~/.config/diorama/diorama-camera.json)')

  c = sub.add_parser('camera', help="set the page's camera in the browser, over a still scene of its own, and save it to a file")
  c.add_argument('--file', type=Path, help='where it is saved (default: ~/.config/diorama/diorama-camera.json)')
  c.add_argument('--port', type=int, default=8642)
  c.add_argument('--no-browser', action='store_true')
  c.add_argument('--show', action='store_true', help='print the saved camera, and stop')
  c.add_argument('--reset', action='store_true', help="delete the file: the page's own camera again")

  args = ap.parse_args(argv)
  if args.cmd == 'run':
    from dioramaweb.offline import run
    run(args.segments, args.model, args.stopline, args.out, args.start, args.seconds, args.device, rhd=args.rhd)
  elif args.cmd == 'camera':
    from dioramaweb import camera
    from dioramaweb.viewer import serve
    path = args.file or camera.default_path()
    if args.show:
      try:
        found = camera.load(path)
      except ValueError as e:
        raise SystemExit(f'{e}') from None
      print(f'{path}:\n' + json.dumps(found, indent=2) if found is not None else f"no {path}: the page's own camera")
    elif args.reset:
      path.unlink(missing_ok=True)
      print(f"removed {path}: the page's own camera")
    else:
      serve(None, args.port, open_browser=not args.no_browser, camera=path, setup=True)
  else:
    from dioramaweb import camera
    from dioramaweb.viewer import serve
    serve(args.drive, args.port, open_browser=not args.no_browser, camera=args.camera or camera.default_path())
  return 0


if __name__ == '__main__':
  sys.exit(main())
