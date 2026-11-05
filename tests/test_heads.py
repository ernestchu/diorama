"""The heads rebuilt from modelV2 parse back to what the model said, through openpilot's own parse (commasim.outputs)."""
from types import SimpleNamespace as NS

import numpy as np

from commasim.outputs import parse
from diorama import heads


def slices():
  out, at = {}, 0
  for name, n in heads.FIELDS:
    out[name] = slice(at, at + n)
    at += n
  return out


def model_v2(p):
  """What openpilot's fill_model_msg makes of the parsed heads, as far as heads.encode reads it."""
  plan, std = p['plan'], p['plan_stds']
  def xyz(c, with_std):
    return NS(x=plan[:, c], y=plan[:, c + 1], z=plan[:, c + 2], **({'xStd': std[:, c], 'yStd': std[:, c + 1], 'zStd': std[:, c + 2]} if with_std else {}))
  line = lambda a, i: NS(y=a[i, :, 0], z=a[i, :, 1])
  return NS(
    position=xyz(0, True), velocity=xyz(3, False), acceleration=xyz(6, False), orientation=xyz(9, False), orientationRate=xyz(12, False),
    laneLines=[line(p['lane_lines'], i) for i in range(4)], laneLineStds=p['lane_lines_stds'][:, 0, 0],
    laneLineProbs=p['lane_lines_prob'][1::2],
    roadEdges=[line(p['road_edges'], i) for i in range(2)], roadEdgeStds=p['road_edges_stds'][:, 0, 0],
    leadsV3=[NS(x=l[:, 0], y=l[:, 1], v=l[:, 2], a=l[:, 3], xStd=s[:, 0], yStd=s[:, 1], vStd=s[:, 2], aStd=s[:, 3], prob=pr)
             for l, s, pr in zip(p['lead'], p['lead_stds'], p['lead_prob'], strict=True)])


def test_encode_parses_back():
  rng = np.random.default_rng(0)
  raw = rng.normal(0, 1, heads.COUNT).astype(np.float32)
  s = slices()
  want = parse(raw, s)
  odo = NS(trans=want['pose'][:3], rot=want['pose'][3:], transStd=want['pose_stds'][:3], rotStd=want['pose_stds'][3:])
  got = parse(heads.encode(model_v2(want), odo), s)
  close = lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
  close(got['plan'], want['plan'])
  close(got['plan_stds'][:, :3], want['plan_stds'][:, :3])
  close(got['lane_lines'], want['lane_lines'])
  close(got['lane_lines_stds'][:, 0, 0], want['lane_lines_stds'][:, 0, 0])
  close(got['lane_lines_prob'][1::2], want['lane_lines_prob'][1::2])
  close(got['road_edges'], want['road_edges'])
  close(got['road_edges_stds'][:, 0, 0], want['road_edges_stds'][:, 0, 0])
  close(got['lead'], want['lead'])
  close(got['lead_stds'], want['lead_stds'])
  close(got['lead_prob'], want['lead_prob'])
  close(got['pose'], want['pose'])
  close(got['pose_stds'], want['pose_stds'])


def test_layout_is_the_phones():
  assert heads.layout() == [['plan', 990], ['lane_lines', 528], ['lane_lines_prob', 8], ['road_edges', 264], ['lead', 144],
                            ['lead_prob', 3], ['pose', 12]]
