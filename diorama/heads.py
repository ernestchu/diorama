"""
Copyright (c) 2026-, Ernie Chu.

This file is part of diorama and is licensed under the MIT License.

The driving model's heads the phone draws from, rebuilt from what modeld
publishes (modelV2 and cameraOdometry) in the raw form the phone parses as
openpilot's Parser does: means then log stds for the mixture heads, logits for
the probabilities. So the phone reads them exactly as it reads a model's own
output vector, whichever model the comma runs.

What modelV2 does not carry (the velocity's and acceleration's stds, a lane
line's std past its first point) is sent as 1, which the phone's drawing does
not read.
"""
from __future__ import annotations

import numpy as np

IDX_N, PLAN_WIDTH, LEAD_TRAJ_LEN, LEAD_N, LEAD_WIDTH = 33, 15, 6, 3, 4
# the heads, in the order a frame carries them, with their float counts
FIELDS = (('plan', 2 * IDX_N * PLAN_WIDTH), ('lane_lines', 2 * 4 * IDX_N * 2), ('lane_lines_prob', 8),
          ('road_edges', 2 * 2 * IDX_N * 2), ('lead', 2 * LEAD_N * LEAD_TRAJ_LEN * LEAD_WIDTH), ('lead_prob', LEAD_N),
          ('pose', 12))
COUNT = sum(n for _, n in FIELDS)
_EPS = 1e-6


def _log(std) -> np.ndarray:
  return np.log(np.maximum(np.asarray(std, np.float64), _EPS))


def _logit(p) -> np.ndarray:
  p = np.clip(np.asarray(p, np.float64), _EPS, 1 - _EPS)
  return np.log(p / (1 - p))


def _vals(xs, n: int, fill: float = 0.0) -> np.ndarray:
  """A capnp list (or anything iterable) as n floats, padded with `fill`."""
  a = np.full(n, fill, np.float64)
  v = np.asarray(list(xs), np.float64)[:n]
  a[:v.size] = v
  return a


def encode(model, odometry) -> np.ndarray:
  """modelV2 and cameraOdometry (or anything with their fields) -> the heads, float32, COUNT long."""
  plan = np.zeros((IDX_N, PLAN_WIDTH))
  plan_std = np.ones((IDX_N, PLAN_WIDTH))
  for col, name in ((0, 'position'), (3, 'velocity'), (6, 'acceleration'), (9, 'orientation'), (12, 'orientationRate')):
    xyz = getattr(model, name)
    for j, axis in enumerate('xyz'):
      plan[:, col + j] = _vals(getattr(xyz, axis), IDX_N)
      std = getattr(xyz, axis + 'Std', None)
      if std is not None and len(std):
        plan_std[:, col + j] = _vals(std, IDX_N, 1.0)

  def lines(items, stds, n):
    mean = np.zeros((n, IDX_N, 2))
    std = np.ones((n, IDX_N, 2))
    for i in range(n):
      line = items[i]
      mean[i, :, 0] = _vals(line.y, IDX_N)
      mean[i, :, 1] = _vals(line.z, IDX_N)
      if i < len(stds):
        std[i, :, :] = stds[i]
    return mean, std

  lane, lane_std = lines(model.laneLines, list(model.laneLineStds), 4)
  edge, edge_std = lines(model.roadEdges, list(model.roadEdgeStds), 2)
  lane_prob = np.zeros(8)
  probs = _vals(model.laneLineProbs, 4)
  lane_prob[1::2] = _logit(probs)
  lane_prob[0::2] = _logit(1 - probs)

  lead = np.zeros((LEAD_N, LEAD_TRAJ_LEN, LEAD_WIDTH))
  lead_std = np.ones((LEAD_N, LEAD_TRAJ_LEN, LEAD_WIDTH))
  lead_prob = np.full(LEAD_N, _logit(0.0))
  for i, l in enumerate(list(model.leadsV3)[:LEAD_N]):
    for j, name in enumerate('xyva'):
      lead[i, :, j] = _vals(getattr(l, name), LEAD_TRAJ_LEN)
      std = getattr(l, name + 'Std', None)
      if std is not None and len(std):
        lead_std[i, :, j] = _vals(std, LEAD_TRAJ_LEN, 1.0)
    lead_prob[i] = _logit(l.prob)

  pose = np.concatenate([_vals(odometry.trans, 3), _vals(odometry.rot, 3)])
  pose_std = np.concatenate([_vals(odometry.transStd, 3, 1.0), _vals(odometry.rotStd, 3, 1.0)])

  out = np.concatenate([
    plan.ravel(), _log(plan_std).ravel(),
    lane.ravel(), _log(lane_std).ravel(), lane_prob,
    edge.ravel(), _log(edge_std).ravel(),
    lead.ravel(), _log(lead_std).ravel(), lead_prob,
    pose, _log(pose_std),
  ]).astype(np.float32)
  assert out.size == COUNT
  return out


def layout() -> list[list]:
  """The hello's `show.outputs`."""
  return [[name, n] for name, n in FIELDS]
