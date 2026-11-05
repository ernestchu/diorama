"""
Copyright (c) 2026-, Ernie Chu.

This file is part of diorama and is licensed under the MIT License.

The link's framing: jetlink's protocol 3 (jetlink/protocol.py, Copyright (c)
2026-, Zeph Leggett, MIT), which zoompilot already speaks to a phone on the
cable, with Diorama's SHOW flag. Every message is a 32-byte header and a
payload; one that ends on a multiple of 512 bytes gets a pad byte, flagged.
"""
from __future__ import annotations

import json
import socket
import struct
from enum import IntEnum

MAGIC = 0x4B4E4C4A  # b'JLNK'
VERSION = 3
HEADER = struct.Struct('<IHHIIIQ4x')  # magic, version, type, seq, flags, length, reserved
PADDED = 1 << 7
PACKET_MULTIPLE = 512
SHOW = 1 << 3  # on INFER_REQ: the frame is shown, not inferred, and nothing is answered
INFER_REQ = struct.Struct('<II')  # frame_id, flags
# the comma's end of the iPhone's cable (jetlink's iOS gadget), and Diorama's port beside jetlink's 5599
HOST, PORT = '192.168.60.1', 5600


class Msg(IntEnum):
  HELLO_REQ = 1
  HELLO_RESP = 2
  ENGINE_REQ = 3
  ENGINE_RESP = 4
  INFER_REQ = 8
  INFER_RESP = 9
  ERROR = 14
  PING = 15
  PONG = 16
  LEAVE = 19


class LinkError(RuntimeError):
  pass


def frame(msg_type: int, seq: int, *parts: bytes) -> list[bytes]:
  """One message as the buffers that go on the wire."""
  length = sum(len(p) for p in parts)
  pad = (HEADER.size + length) % PACKET_MULTIPLE == 0
  out = [HEADER.pack(MAGIC, VERSION, int(msg_type), seq & 0xFFFFFFFF, PADDED if pad else 0, length, 0), *parts]
  if pad:
    out.append(b'\0')
  return out


def send(sock: socket.socket, msg_type: int, seq: int, *parts: bytes) -> None:
  bufs = frame(msg_type, seq, *parts)
  total = sum(len(b) for b in bufs)
  sent = sock.sendmsg(bufs)
  if sent < total:  # a partial vectored write: the rest in one piece
    sock.sendall(b''.join(bufs)[sent:])


def send_json(sock: socket.socket, msg_type: int, seq: int, obj: dict) -> None:
  send(sock, msg_type, seq, json.dumps(obj, separators=(',', ':')).encode())


def recv(sock: socket.socket) -> tuple[int, int, bytes]:
  """The next message: type, seq, payload. LinkError on a broken stream or a closed socket."""
  magic, version, msg_type, seq, flags, length, _ = HEADER.unpack(_exact(sock, HEADER.size))
  if magic != MAGIC or version != VERSION:
    raise LinkError(f'not a protocol {VERSION} peer (magic 0x{magic:08x}, version {version})')
  if length > 8 << 20:
    raise LinkError(f'a {length}-byte message')
  body = _exact(sock, length + (1 if flags & PADDED else 0))
  return msg_type, seq, body[:length]


def _exact(sock: socket.socket, n: int) -> bytes:
  buf = bytearray(n)
  view, got = memoryview(buf), 0
  while got < n:
    k = sock.recv_into(view[got:])
    if not k:
      raise LinkError('the phone closed the link')
    got += k
  return bytes(buf)
