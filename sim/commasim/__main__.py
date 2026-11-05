"""
Copyright (c) 2026-, Ernie Chu.

This file is part of diorama and is licensed under the MIT License.

The comma, played by a Mac: a recorded drive streamed to the Diorama app as
zoompilot streams a live one. It listens where the comma does (port 5599), and
once the phone dials it says hello with `show`, then sends every frame as an
INFER_REQ flagged SHOW: the packed narrow frame and the driving model's heads
(DESIGN.md). It runs the driving model itself, as the comma does.

    uv run python -m commasim ~/Downloads/jetlink/driving_footage --model ~/Downloads/driving_supercombo.onnx

The iOS Simulator's app dials 127.0.0.1; an iPhone dials this Mac's address.
"""
from __future__ import annotations

import argparse
import json
import os
import select
import socket
import struct
import time
from pathlib import Path

import numpy as np

from commasim.driving import DrivingModel
from commasim.outputs import DRIVE_FIELDS
from commasim.route import FPS, Route
from commasim.vendor import protocol as P

SHOW = 1 << 3  # INFER_REQ's flag: shown, not inferred (DESIGN.md)


def send(sock: socket.socket, msg_type: int, seq: int, *parts: bytes) -> None:
  """One message, with the pad byte where it needs one, as jetlink's TcpTransport sends it."""
  length = sum(len(p) for p in parts)
  flags = P.Flag.PADDED if (P.HEADER_SIZE + length) % P.PACKET_MULTIPLE == 0 else 0
  bufs = [P.pack_header(msg_type, seq, length, flags), *parts]
  if flags:
    bufs.append(b'\0')
  sock.sendall(b''.join(bufs))


def recv(sock: socket.socket, timeout: float | None) -> tuple[int, int, bytes] | None:
  """The next message, or None when none came within `timeout`."""
  sock.settimeout(timeout)
  try:
    head = _exact(sock, P.HEADER_SIZE)
  except TimeoutError:
    return None
  finally:
    sock.settimeout(None)
  _, _, msg_type, seq, flags, length, _ = P.unpack_header(head)
  body = _exact(sock, length + (1 if flags & P.Flag.PADDED else 0))
  return msg_type, seq, body[:length]


def _exact(sock: socket.socket, n: int) -> bytes:
  buf = bytearray()
  while len(buf) < n:
    chunk = sock.recv(n - len(buf))
    if not chunk:
      raise ConnectionError('the phone closed the link')
    buf += chunk
  return bytes(buf)


def drain(sock: socket.socket) -> None:
  """Whatever the phone sent (it answers only pings), read and dropped."""
  while select.select([sock], [], [], 0)[0]:
    if recv(sock, 0.1) is None:
      return


def serve(sock: socket.socket, route: Route, model: DrivingModel, start: int, count: int | None, loop: bool) -> None:
  cal = route.calibration
  fields = [name for name in DRIVE_FIELDS if name in model.slices]
  layout = [[name, model.slices[name].stop - model.slices[name].start] for name in fields]
  hello = {'client': {
    'nonce': os.urandom(8).hex(), 'name': 'commasim',
    'camera': {'wide_from_device_euler': list(cal.wide_from_device), 'height': cal.height},
    'show': {'frame': [6, 128, 256], 'outputs': layout},
  }}
  seq = 1
  send(sock, P.Msg.HELLO_REQ, seq, json.dumps(hello).encode())
  while (msg := recv(sock, 5.0)) is not None and msg[0] != P.Msg.HELLO_RESP:
    pass
  if msg is None:
    raise ConnectionError('no hello back from the phone')
  resp = json.loads(msg[2])
  print(f"hello from {resp.get('name', '?')}: {'a viewer' if resp.get('viewer') else 'NOT a viewer'}")
  if not resp.get('viewer'):
    raise ConnectionError('the server is no Diorama viewer')

  while True:
    model.reset()
    t0 = time.monotonic()
    sent = 0
    for k, _narrow_yuv, warped in route.warped(start, count):
      outputs = model.step(warped)
      vector = np.concatenate([outputs[model.slices[name]] for name in fields]).astype('<f4')
      due = t0 + sent / FPS
      if (wait := due - time.monotonic()) > 0:
        time.sleep(wait)
      seq = (seq + 1) & 0xFFFFFFFF
      send(sock, P.Msg.INFER_REQ, seq, struct.pack('<II', k, SHOW), np.ascontiguousarray(warped[0]).tobytes(), vector.tobytes())
      drain(sock)
      sent += 1
      if sent % (5 * FPS) == 0:
        print(f'frame {k}: {sent / (time.monotonic() - t0):.1f} Hz')
    if not loop:
      return
    print('the drive ended; again from the start')


def main() -> None:
  ap = argparse.ArgumentParser(prog='commasim', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('segments', nargs='+', type=Path, help='segment directories, or a directory of them')
  ap.add_argument('--model', required=True, help='the driving model, ONNX (the comma\'s own)')
  ap.add_argument('--device', default='auto', choices=('auto', 'coreml', 'cpu', 'cuda'))
  ap.add_argument('--listen', default='0.0.0.0', help='address to listen on (default all)')
  ap.add_argument('--port', type=int, default=5599)
  ap.add_argument('--start', type=float, default=0.0, help='seconds into the drive')
  ap.add_argument('--seconds', type=float, help='how much of it (default all)')
  ap.add_argument('--once', action='store_true', help='play the drive once, not over and over')
  args = ap.parse_args()

  route = Route.open(args.segments)
  model = DrivingModel(args.model, args.device)
  start = int(round(args.start * FPS))
  count = None if args.seconds is None else int(round(args.seconds * FPS))
  srv = socket.create_server((args.listen, args.port), reuse_port=True)
  print(f'the comma, played by this Mac: listening on {args.listen}:{args.port}')
  while True:
    phone, addr = srv.accept()
    phone.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f'the phone dialed from {addr[0]}')
    try:
      serve(phone, route, model, start, count, loop=not args.once)
    except (ConnectionError, OSError) as e:
      print(f'link lost: {e}')
    finally:
      phone.close()


if __name__ == '__main__':
  main()
