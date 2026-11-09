"""
Copyright (c) 2026-, Ernie Chu.

This file is part of diorama and is licensed under the MIT License.

What the comma 4's display draws for made-up model outputs, for the parity test of the app's drawing of the same
(DioramaKit/Tests/CommaUIParityTests.swift). The display's own ModelRenderer runs here, the copy in
commasim/vendor/comma_ui, its draw calls recorded with the rest of openpilot stubbed out, as engaged with openpilot
longitudinal control; and a software rasteriser paints them as its shader does.

    uv run python comma_ui_parity.py          # write the fixtures
    uv run python comma_ui_parity.py --check  # fail where they differ from what the pinned file draws
"""
import dataclasses
import enum
import hashlib
import importlib.util
import json
import sys
import types
import zlib
from collections import namedtuple
from pathlib import Path

import numpy as np

PINNED_COMMIT = '2ef7f8522f35a91e017b102673ec31fb9f1cebe7'
PINNED_SHA256 = '9aebf9a7076a718feb5bf0cfce8e0e2d7ba387c9460fe2569296dc02183771b3'
HERE = Path(__file__).parent
SOURCE = HERE / 'commasim/vendor/comma_ui/model_renderer.py'
FIXTURES = HERE / '../DioramaKit/Tests/Fixtures/comma_ui'

W, H, F, CX, CY = 512, 256, 910.0, 256.0, 47.6
HEIGHT = 1.22
N = 33
X_IDXS = 192 * (np.arange(N) / 32) ** 2
T_IDXS = 10 * (np.arange(N) / 32) ** 2
VIEW_FROM_DEVICE = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
RASTERED = ('cruise', 'braking', 'lead_close', 'curve_left')

Color = namedtuple('Color', 'r g b a')
Rectangle = namedtuple('Rectangle', 'x y width height')
draws = []


class Status(enum.Enum):
  DISENGAGED = 'disengaged'
  ENGAGED = 'engaged'
  OVERRIDE = 'override'
  LAT_ONLY = 'lat_only'
  LONG_ONLY = 'long_only'


@dataclasses.dataclass
class Gradient:
  start: tuple
  end: tuple
  colors: list
  stops: list


def draw_polygon(rect, points, color=None, gradient=None):
  if len(points) >= 3:
    draws.append((np.array(points, dtype=np.float32), color, None if gradient is None else (list(gradient.colors), list(gradient.stops))))


class FirstOrderFilter:
  def __init__(self, x0, rc, dt, initialized=True):
    self.x, self.dt, self.initialized = x0, dt, initialized
    self.alpha = dt / (rc + dt)

  def update(self, x):
    self.x = (1 - self.alpha) * self.x + self.alpha * x if self.initialized else x
    self.initialized = True
    return self.x


class Bag:
  def __init__(self, **kw):
    self.__dict__.update(kw)


class Blendful:
  def update(self, *a):
    pass

  def draw(self, *a):
    pass


class Sm(dict):
  recv_frame = types.MappingProxyType({'extrinsicsCalibration': 1, 'modelV2': 1})
  updated = types.MappingProxyType({'carParams': False, 'modelV2': True, 'radarState': True})
  valid = types.MappingProxyType({'radarState': True})


UI_STATE = Bag(status=Status.ENGAGED, torque_utilization=0.0, started_frame=0, rainbow_path=False, sm=None, params=Bag(get=lambda *a, **k: 0.0))


class Widget:
  pass


class ModelRendererSP:
  def __init__(self):
    self.rainbow_path = None
    self.blendful = Blendful()


def load():
  """The file itself, unchanged; only what it imports is made up."""
  if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != PINNED_SHA256:
    sys.exit(f'{SOURCE} is not the file of zoompilot {PINNED_COMMIT}')
  E2E = 1
  log = Bag(LongitudinalPlan=Bag(LongitudinalPlanSource=Bag(e2e=E2E)))
  modules = {
    'pyray': dict(Color=Color, Rectangle=Rectangle),
    'openpilot': {}, 'openpilot.common': {}, 'openpilot.selfdrive': {}, 'openpilot.selfdrive.controls': {}, 'openpilot.selfdrive.locationd': {},
    'openpilot.selfdrive.ui': {}, 'openpilot.selfdrive.ui.mici': {}, 'openpilot.selfdrive.ui.sunnypilot': {}, 'openpilot.selfdrive.ui.sunnypilot.mici': {},
    'openpilot.selfdrive.ui.sunnypilot.mici.onroad': {}, 'openpilot.system': {}, 'openpilot.system.ui': {}, 'openpilot.system.ui.lib': {},
    'opendbc': {}, 'opendbc.car': {},
    'openpilot.cereal': dict(log=log, messaging=Bag()),
    'opendbc.car.structs': dict(car=Bag()),
    'openpilot.common.params': dict(Params=lambda: Bag(get=lambda *a, **k: None)),
    'openpilot.common.filter_simple': dict(FirstOrderFilter=FirstOrderFilter),
    'openpilot.selfdrive.controls.radard': dict(RADAR_TO_CAMERA=1.52),
    'openpilot.selfdrive.locationd.calibrationd': dict(HEIGHT_INIT=np.array([HEIGHT])),
    'openpilot.selfdrive.ui.ui_state': dict(ui_state=UI_STATE, UIStatus=Status),
    'openpilot.selfdrive.ui.mici.onroad': dict(blend_colors=lambda a, b, f: a),
    'openpilot.system.ui.lib.application': dict(gui_app=Bag(target_fps=60)),
    'openpilot.system.ui.lib.shader_polygon': dict(draw_polygon=draw_polygon, Gradient=Gradient),
    'openpilot.system.ui.widgets': dict(Widget=Widget),
    'openpilot.selfdrive.ui.sunnypilot.mici.onroad.model_renderer': dict(LANE_LINE_COLORS_SP={}, ModelRendererSP=ModelRendererSP),
  }
  for name, attrs in modules.items():
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
  spec = importlib.util.spec_from_file_location('comma_model_renderer', SOURCE)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module, E2E


def f32(a):
  return np.asarray(a, dtype=np.float32)


def heads(rng, v=20.0, a=0.0, bend=0.0, probs=(0.8, 0.95, 0.95, 0.7), edge_stds=(0.2, 0.3), leads=(), hill=0.0, jitter=0.0):
  """Parsed heads, in openpilot's shapes, of a road and a plan that follow one curve."""
  x = np.maximum(v * T_IDXS + 0.5 * a * T_IDXS ** 2, 0.0)
  plan = np.zeros((N, 15))
  plan[:, 0], plan[:, 1], plan[:, 2] = x, bend * x ** 2, -hill * x ** 2
  plan[:, 6] = a * np.cos(T_IDXS / 3) + jitter * rng.normal(size=N)
  lane = np.zeros((4, N, 2))
  for i, off in enumerate((-5.4, -1.8, 1.8, 5.4)):
    lane[i, :, 0] = off + bend * X_IDXS ** 2 + jitter * rng.normal(size=N) * 0.05
    lane[i, :, 1] = HEIGHT - hill * X_IDXS ** 2
  edge = np.zeros((2, N, 2))
  for i, off in enumerate((-7.2, 7.2)):
    edge[i, :, 0] = off + bend * X_IDXS ** 2
    edge[i, :, 1] = HEIGHT - hill * X_IDXS ** 2
  stds = np.zeros((2, N, 2))
  stds[0, 0, 0], stds[1, 0, 0] = edge_stds
  prob = np.zeros(8)
  prob[1::2] = probs
  lead = np.zeros((3, 6, 4))
  lead_prob = np.zeros(3)
  for i, (p, d, y) in enumerate(leads):
    lead_prob[i], lead[i, 0, 0], lead[i, 0, 1] = p, d, y
  return dict(plan=f32(plan), lane_lines=f32(lane), road_edges=f32(edge), road_edges_stds=f32(stds), lane_lines_prob=f32(prob), lead=f32(lead), lead_prob=f32(lead_prob))


def cases():
  rng = np.random.default_rng(20261109)
  out = {
    'cruise': heads(rng),
    'curve_left': heads(rng, bend=-0.0012),
    'curve_right': heads(rng, v=12.0, bend=0.003, a=-0.4),
    'braking': heads(rng, a=-2.5),
    'accelerating': heads(rng, v=8.0, a=1.6),
    'stopped': heads(rng, v=0.0),
    'nearly_stopped': heads(rng, v=0.4, a=0.1),
    'lead_close': heads(rng, v=9.0, leads=[(0.9, 14.0, 0.4)]),
    'lead_far': heads(rng, v=30.0, leads=[(0.9, 80.0, -1.0)]),
    'lead_past_draw_distance': heads(rng, v=30.0, leads=[(0.9, 120.0, 0.0)]),
    'lead_unsure': heads(rng, leads=[(0.4, 30.0, 0.0)]),
    'two_leads': heads(rng, leads=[(0.9, 25.0, 0.2), (0.8, 50.0, -0.5)]),
    'two_leads_one_car': heads(rng, leads=[(0.9, 25.0, 0.2), (0.8, 26.5, 0.3)]),
    'lanes_unsure': heads(rng, probs=(0.05, 0.2, 0.1, 0.0), edge_stds=(0.05, 1.4)),
    'hill': heads(rng, v=25.0, hill=-0.002),
    'crest': heads(rng, v=25.0, hill=0.004),
  }
  for i in range(6):
    p = rng.uniform(0, 1, 4)
    out[f'random_{i}'] = heads(
      rng, v=rng.uniform(0, 35), a=rng.uniform(-3, 2), bend=rng.uniform(-0.003, 0.003), probs=tuple(p), edge_stds=tuple(rng.uniform(0, 1.2, 2)),
      leads=[(rng.uniform(0, 1), rng.uniform(5, 110), rng.uniform(-3, 3)), (rng.uniform(0, 1), rng.uniform(5, 110), rng.uniform(-3, 3))],
      hill=rng.uniform(-0.002, 0.003), jitter=0.4)
  return out


def xyz(x, y, z):
  return Bag(x=list(x), y=list(y), z=list(z))


def model_v2(h):
  """As modeld fills it (fill_model_msg.py)."""
  plan = h['plan']
  return Bag(
    position=xyz(plan[:, 0], plan[:, 1], plan[:, 2]),
    acceleration=xyz(plan[:, 6], plan[:, 7], plan[:, 8]),
    laneLines=[xyz(f32(X_IDXS), h['lane_lines'][i, :, 0], h['lane_lines'][i, :, 1]) for i in range(4)],
    laneLineProbs=list(h['lane_lines_prob'][1::2]),
    roadEdges=[xyz(f32(X_IDXS), h['road_edges'][i, :, 0], h['road_edges'][i, :, 1]) for i in range(2)],
    roadEdgeStds=list(h['road_edges_stds'][:, 0, 0]),
    leadsV3=[Bag(prob=float(h['lead_prob'][i]), x=list(h['lead'][i, :, 0]), y=list(h['lead'][i, :, 1])) for i in range(3)],
  )


def vision_lead(h, i):
  """radard's RadarState from the model's lead, on a car without radar."""
  return Bag(present=bool(h['lead_prob'][i] > 0.5), dRel=float(h['lead'][i, 0, 0]) - 1.52, yRel=float(-h['lead'][i, 0, 1]))


def draw_case(module, e2e, h, transform):
  renderer = module.ModelRenderer()
  renderer._longitudinal_control = True
  renderer.set_transform(transform)
  rect = Rectangle(0.0, 0.0, float(W), float(H))
  renderer._rect = rect
  UI_STATE.sm = Sm(
    modelV2=model_v2(h), extrinsicsCalibration=Bag(height=[HEIGHT]), carParams=Bag(openpilotLongitudinalControl=True),
    radarState=Bag(leadOne=vision_lead(h, 0), leadTwo=vision_lead(h, 1)), longitudinalPlan=Bag(longitudinalPlanSource=e2e),
    selfdriveState=Bag(enabled=True, engageable=True), carState=Bag(brakePressed=False))
  renderer._render(rect)
  # the fade is settled: each picture is one frame of the model's, not 60 of the display's
  for lead in renderer._lead_vehicles:
    lead.fade_filter.x = 0.8 if lead.fade_filter.x > 0 else 0.0
  draws.clear()
  renderer._render(rect)
  return [d for d in draws if d[1] is None or d[1].a > 0]  # nothing shows of the rest


def lead_sequence(module, e2e, transform):
  """A lead's bar over model frames, each rendered 3 times as a 60 Hz display does: it comes, moves, jumps, goes, and two come."""
  base = heads(np.random.default_rng(1))
  renderer = module.ModelRenderer()
  renderer._longitudinal_control = True
  renderer.set_transform(transform)
  rect = Rectangle(0.0, 0.0, float(W), float(H))
  renderer._rect = rect
  frames = []
  for k in range(70):
    leads = []
    if 10 <= k < 30:
      leads = [(0.9, 40.0 - 0.4 * k + 0.5 * np.sin(k), 0.3 * np.sin(k / 3))]
    elif 30 <= k < 40:
      leads = [(0.9, 28.0 + 0.3 * np.sin(k), 4.5)]  # another car, over 3 m to the side: the bar snaps to it
    elif k >= 48:
      leads = [(0.9, 25.0 + 0.5 * np.cos(k), -0.5), (0.8, 60.0 - 0.2 * (k - 48), 1.0)]
    h = dict(base)
    lead, lead_prob = np.zeros((3, 6, 4)), np.zeros(3)
    for i, (p, d, y) in enumerate(leads):
      lead_prob[i], lead[i, 0, 0], lead[i, 0, 1] = p, d, y
    h['lead'], h['lead_prob'] = f32(lead), f32(lead_prob)
    UI_STATE.sm = Sm(
      modelV2=model_v2(h), extrinsicsCalibration=Bag(height=[HEIGHT]), carParams=Bag(openpilotLongitudinalControl=True),
      radarState=Bag(leadOne=vision_lead(h, 0), leadTwo=vision_lead(h, 1)), longitudinalPlan=Bag(longitudinalPlanSource=e2e),
      selfdriveState=Bag(enabled=True, engageable=True), carState=Bag(brakePressed=False))
    for _ in range(3):
      renderer._render(rect)
    bars = [{'points': numbers(v.bar), 'alpha': int(255 * v.fade_filter.x)} for v in renderer._lead_vehicles if len(v.bar) >= 3 and int(255 * v.fade_filter.x) > 0]
    frames.append({'lead': numbers(h['lead']), 'lead_prob': numbers(h['lead_prob']), 'bars': bars})
  return {'base': {k: numbers(v) for k, v in base.items()}, 'frames': frames}


def raster(polygons):
  """The display's shader over mid grey: its triangle strips, each polygon blended once, straight alpha."""
  image = np.full((H, W, 4), 128.0)
  image[..., 3] = 255
  ys, xs = np.mgrid[0:H, 0:W]
  for points, color, gradient in polygons:
    pts = points[:-1] if len(points) % 2 else points
    n = len(pts)
    strip = []
    for i in range(n // 2):
      strip += [pts[i], pts[-i - 1]]
    mask = np.zeros((H, W), bool)
    for a, b, c in zip(strip, strip[1:], strip[2:]):
      area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
      if abs(area) < 1e-9:
        continue
      left, right = int(max(min(a[0], b[0], c[0]), 0)), int(min(max(a[0], b[0], c[0]) + 2, W))
      top, bottom = int(max(min(a[1], b[1], c[1]), 0)), int(min(max(a[1], b[1], c[1]) + 2, H))
      if left >= right or top >= bottom:
        continue
      px, py = xs[top:bottom, left:right] + 0.5, ys[top:bottom, left:right] + 0.5
      w0 = ((b[0] - px) * (c[1] - py) - (b[1] - py) * (c[0] - px)) / area
      w1 = ((c[0] - px) * (a[1] - py) - (c[1] - py) * (a[0] - px)) / area
      mask[top:bottom, left:right] |= (w0 >= 0) & (w1 >= 0) & (1 - w0 - w1 >= 0)
    if gradient is None:
      source = np.broadcast_to(np.array(color, dtype=float), image.shape)
    else:
      colors = np.array(gradient[0], dtype=float)
      stops = np.clip(np.array(gradient[1], dtype=float), 0, 1)
      t = (H - (ys + 0.5)) / H
      source = np.empty(image.shape)
      for c in range(4):
        source[..., c] = np.interp(t, stops, colors[:, c])
    alpha = source[..., 3:4][mask] / 255
    image[..., :3][mask] = source[..., :3][mask] * alpha + image[..., :3][mask] * (1 - alpha)
  return np.rint(image).astype(np.uint8)


def number(v):
  return float(np.format_float_positional(v, unique=True, trim='0'))


def numbers(a):
  return [number(v) for v in np.asarray(a, dtype=np.float32).ravel()]


def main():
  module, e2e = load()
  transform = np.array([[F, 0, CX], [0, F, CY], [0, 0, 1]], dtype=np.float32) @ VIEW_FROM_DEVICE
  out = {'commit': PINNED_COMMIT, 'width': W, 'height': H, 'f': F, 'cx': CX, 'cy': CY, 'road_height': HEIGHT, 'transform': numbers(transform), 'cases': {}}
  rasters = {}
  for name, h in cases().items():
    polygons = draw_case(module, e2e, h, transform)
    out['cases'][name] = {
      'heads': {k: numbers(v) for k, v in h.items()},
      'draws': [
        {'points': numbers(p)} | ({'color': list(c)} if g is None else {'colors': [list(x) for x in g[0]], 'stops': numbers(g[1])})
        for p, c, g in polygons
      ],
    }
    if name in RASTERED:
      rasters[name] = raster(polygons).tobytes()

  out['lead_sequence'] = lead_sequence(module, e2e, transform)
  files = {'golden.json': json.dumps(out, separators=(',', ':')).encode()}
  for name, data in rasters.items():
    deflate = zlib.compressobj(9, zlib.DEFLATED, -15)
    files[f'{name}.rgba.deflate'] = deflate.compress(data) + deflate.flush()
  if '--check' in sys.argv:
    bad = [name for name, data in files.items() if not (FIXTURES / name).exists() or (FIXTURES / name).read_bytes() != data]
    sys.exit(f'differ from the pinned drawing: {bad}' if bad else 0)
  FIXTURES.mkdir(parents=True, exist_ok=True)
  for name, data in files.items():
    (FIXTURES / name).write_bytes(data)
  print({name: len(data) for name, data in files.items()}, {k: len(v['draws']) for k, v in out['cases'].items()})


if __name__ == '__main__':
  main()
