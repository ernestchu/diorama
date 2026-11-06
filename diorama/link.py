"""
Copyright (c) 2026-, Ernie Chu.

This file is part of diorama and is licensed under the MIT License.

The phone's end of the cable, as the comma sees it: listen where the phone
dials, say hello with what is streamed (`show`), and send each frame without
waiting. One phone at a time; a new dial replaces the last.

Nothing here waits on the phone where the caller can see it: show() leaves
the frame in a one-frame mailbox, and a sender thread takes the newest. A
frame the phone has not taken by the time the next comes is dropped.
"""
from __future__ import annotations

import os
import select
import socket
import threading
import time

from diorama import heads, wire


class Viewer:
  def __init__(self, host: str = wire.HOST, port: int = wire.PORT, name: str = 'comma', log=None):
    self.host, self.port, self.name = host, port, name
    self.log = log
    self.camera: dict | None = None  # {height, wide_from_device_euler}: the calibration the hello carries
    self.frames = (6, 128, 256)
    self._lock = threading.Condition()
    self._mail: tuple[int, bytes, bytes] | None = None
    self._sock: socket.socket | None = None
    self._stop = threading.Event()
    self.sent = 0
    threading.Thread(target=self._accept_loop, name='diorama-accept', daemon=True).start()
    threading.Thread(target=self._send_loop, name='diorama-send', daemon=True).start()

  @property
  def watched(self) -> bool:
    """A phone has said hello: the caller can skip the warp while nobody watches."""
    return self._sock is not None

  def show(self, frame_id: int, packed, outputs) -> None:
    """The newest frame for the phone: the packed narrow frame (uint8) and the heads (float32)."""
    with self._lock:
      self._mail = (frame_id, bytes(memoryview(packed).cast('B')), bytes(memoryview(outputs).cast('B')))
      self._lock.notify()

  def close(self) -> None:
    self._stop.set()
    self._drop('closing')
    with self._lock:
      self._lock.notify_all()

  def _say(self, msg: str) -> None:
    if self.log is not None:
      self.log.warning('diorama: ' + msg)

  def _drop(self, why: str) -> None:
    with self._lock:
      sock, self._sock = self._sock, None
    if sock is not None:
      self._say(f'the phone left: {why}')
      try:
        sock.close()
      except OSError:
        pass

  def _accept_loop(self) -> None:
    srv = None
    while not self._stop.is_set():
      if srv is None:
        try:
          # the cable's address exists only while the iOS gadget is up: try again until it is
          # not SO_REUSEPORT: a second dioramad must fail to listen, not split the phone's dials with the first
          srv = socket.create_server((self.host, self.port))
          srv.settimeout(1.0)
          self._say(f'listening on {self.host}:{self.port}')
        except OSError:
          time.sleep(2.0)
          continue
      try:
        phone, addr = srv.accept()
      except TimeoutError:
        continue
      except OSError as e:
        self._say(f'the listener failed ({e}), listening again')
        srv.close()
        srv = None
        continue
      try:
        self._greet(phone)
      except (OSError, wire.LinkError, ValueError) as e:
        self._say(f'no hello from {addr[0]}: {e}')
        phone.close()
        continue
      self._drop('another phone dialed')
      with self._lock:
        self._sock = phone
        self._lock.notify()
      self._say(f'{addr[0]} is watching')

  def _greet(self, phone: socket.socket) -> None:
    import json
    phone.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    phone.settimeout(5.0)
    client = {'nonce': os.urandom(8).hex(), 'name': self.name,
              'show': {'frame': list(self.frames), 'outputs': heads.layout()}}
    if self.camera:
      client['camera'] = self.camera
    wire.send_json(phone, wire.Msg.HELLO_REQ, 1, {'client': client})
    while True:
      msg_type, _, body = wire.recv(phone)
      if msg_type == wire.Msg.HELLO_RESP:
        break
    if not json.loads(body).get('viewer'):
      raise ValueError('it is not a Diorama viewer')
    phone.settimeout(1.0)  # a send that stalls this long is a phone gone

  def _send_loop(self) -> None:
    seq = 1
    while not self._stop.is_set():
      with self._lock:
        while (self._mail is None or self._sock is None) and not self._stop.is_set():
          self._lock.wait()
        if self._stop.is_set():
          return
        (frame_id, packed, outputs), self._mail = self._mail, None
        sock = self._sock
      seq += 1
      try:
        wire.send(sock, wire.Msg.INFER_REQ, seq, wire.INFER_REQ.pack(frame_id & 0xFFFFFFFF, wire.SHOW), packed, outputs)
        self.sent += 1
        self._answer_pings(sock)
      except (OSError, wire.LinkError) as e:
        if sock is self._sock:
          self._drop(str(e) or type(e).__name__)

  @staticmethod
  def _answer_pings(sock: socket.socket) -> None:
    """The phone sends nothing unasked; read whatever it did send so it never backs up."""
    while select.select([sock], [], [], 0)[0]:
      msg_type, seq, _ = wire.recv(sock)
      if msg_type == wire.Msg.PING:
        wire.send(sock, wire.Msg.PONG, seq)
