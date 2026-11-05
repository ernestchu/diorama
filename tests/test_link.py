"""dioramad's link, against a phone that answers as the app does."""
import json
import socket
import time

import numpy as np

from diorama import heads, wire
from diorama.link import Viewer


def test_hello_then_frames():
  viewer = Viewer(host='127.0.0.1', port=56000, name='test-comma')
  viewer.camera = {'height': 1.37, 'wide_from_device_euler': [0.0, 0.02, -0.01]}
  for _ in range(50):
    try:
      phone = socket.create_connection(('127.0.0.1', 56000), timeout=5)
      break
    except OSError:
      time.sleep(0.1)
  t, seq, body = wire.recv(phone)
  assert t == wire.Msg.HELLO_REQ
  client = json.loads(body)['client']
  assert client['show'] == {'frame': [6, 128, 256], 'outputs': heads.layout()}
  assert client['camera']['height'] == 1.37
  wire.send_json(phone, wire.Msg.HELLO_RESP, seq, {'viewer': True, 'name': 'test-phone'})
  for _ in range(50):
    if viewer.watched:
      break
    time.sleep(0.02)
  packed = np.arange(6 * 128 * 256, dtype=np.uint32).astype(np.uint8).reshape(6, 128, 256)
  outputs = np.linspace(-1, 1, heads.COUNT, dtype=np.float32)
  viewer.show(42, packed, outputs)
  t, _, body = wire.recv(phone)
  assert t == wire.Msg.INFER_REQ
  frame_id, flags = wire.INFER_REQ.unpack_from(body)
  assert (frame_id, flags) == (42, wire.SHOW)
  assert body[8:8 + packed.size] == packed.tobytes()
  np.testing.assert_array_equal(np.frombuffer(body[8 + packed.size:], np.float32), outputs)
  # a ping is answered
  wire.send(phone, wire.Msg.PING, 7)
  viewer.show(43, packed, outputs)
  seen = set()
  for _ in range(2):
    seen.add(wire.recv(phone)[0])
  assert wire.Msg.PONG in seen
  viewer.close()
  phone.close()
