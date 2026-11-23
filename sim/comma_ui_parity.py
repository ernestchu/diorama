"""
Copyright (c) 2026-, Ernie Chu.

This file is part of diorama and is licensed under the MIT License.

What the comma 4's display draws for made-up model outputs, for the parity test of the app's drawing of the same
(DioramaKit/Tests/CommaUIParityTests.swift). The display's own ModelRenderer runs here, the copies in
commasim/vendor/comma_ui (the mici renderer, the base renderer it extends and the status fade it draws through), its
draw calls recorded with the rest of openpilot stubbed out, as engaged with openpilot longitudinal control; and a
software rasteriser paints them as its shader does.

    uv run python comma_ui_parity.py          # write the fixtures
    uv run python comma_ui_parity.py --check  # fail where they differ from what the pinned files draw

PINNED holds each vendored file's path in zoompilot and its sha256; zoompilot's tests read it to fail when the display's
drawing moves on from these copies (openpilot/selfdrive/ui/tests/test_diorama_comma_ui_pin.py there).
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

PINNED_COMMIT = 'f628686b278675501933c8b9aba9efcb43534648'  # zoompilot main
PINNED = {
  'openpilot/selfdrive/ui/mici/onroad/model_renderer.py': '2609f50bdd8f287e2b9812786f0d85962e751b973bf82fa93b3508675d3fb9c5',
  'openpilot/selfdrive/ui/onroad/model_renderer.py': 'c0f3621c9e74ca3abb8941577a72156eb6d4ef88eb3e3f876a29211e396732bc',
  'openpilot/selfdrive/ui/mici/onroad/status.py': 'af335ca9f3c825902a3f479de8a1b137d0f6f8def518ca2507e4a8a307d36e67',
}
HERE = Path(__file__).parent
VENDOR = HERE / 'commasim/vendor/comma_ui'
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
  frame = 1
  updated = types.MappingProxyType({'carParams': False, 'modelV2': True, 'radarState': True})
  valid = types.MappingProxyType({'radarState': True})


# engaged: openpilot steering and on the speed (not MADS lateral-only), the camera offset at its default of 0
UI_STATE = Bag(
  status=Status.ENGAGED, torque_utilization=0.0, rainbow_path=False, cruise_engaged=True, sm=None,
  received_since_started=lambda name: True, params=Bag(get=lambda *a, **k: 0.0))


class Widget:
  pass


class ChevronMetrics:
  def __init__(self):
    pass


class ModelRendererSP:
  """sunnypilot's mixin for the base renderer: steering, so the path's half width is held at 0.9 m."""
  def __init__(self):
    self.rainbow_path = None
    self.chevron_metrics = None
    self.blendful = Blendful()

  _lateral_active = True

  def _get_path_half_width(self):
    return 0.9


class ModelRendererSPMici:
  def __init__(self):
    self.rainbow_path = None
    self.blendful = Blendful()


def load():
  """The files themselves, unchanged; only what they import from the rest of openpilot is made up."""
  for path, sha in PINNED.items():
    if hashlib.sha256((VENDOR / path).read_bytes()).hexdigest() != sha:
      sys.exit(f'{VENDOR / path} is not the file of zoompilot {PINNED_COMMIT}')
  E2E = 1
  log = Bag(LongitudinalPlan=Bag(LongitudinalPlanSource=Bag(e2e=E2E)))

  def no_color(*a):
    raise AssertionError('blend_colors: these cases have no steering torque to turn the lane lines orange')

  modules = {
    'raylib_nb': dict(Color=Color, Rectangle=Rectangle, get_time=lambda: 0.0),
    'openpilot': {}, 'openpilot.common': {}, 'openpilot.selfdrive': {}, 'openpilot.selfdrive.controls': {}, 'openpilot.selfdrive.locationd': {},
    'openpilot.selfdrive.ui': {}, 'openpilot.selfdrive.ui.onroad': {}, 'openpilot.selfdrive.ui.mici': {}, 'openpilot.selfdrive.ui.sunnypilot': {},
    'openpilot.selfdrive.ui.sunnypilot.onroad': {}, 'openpilot.selfdrive.ui.sunnypilot.mici': {}, 'openpilot.selfdrive.ui.sunnypilot.mici.onroad': {},
    'openpilot.system': {}, 'openpilot.system.ui': {}, 'openpilot.system.ui.lib': {},
    'opendbc': {}, 'opendbc.car': {},
    'openpilot.cereal': dict(log=log, messaging=Bag()),
    'opendbc.car.structs': dict(car=Bag()),
    'openpilot.common.params': dict(Params=lambda: Bag(get=lambda *a, **k: None)),
    'openpilot.common.filter_simple': dict(FirstOrderFilter=FirstOrderFilter),
    'openpilot.selfdrive.controls.radard': dict(RADAR_TO_CAMERA=1.52),
    'openpilot.selfdrive.locationd.calibrationd': dict(HEIGHT_INIT=np.array([HEIGHT])),
    'openpilot.selfdrive.ui.ui_state': dict(ui_state=UI_STATE, UIStatus=Status),
    'openpilot.selfdrive.ui.mici.onroad': dict(blend_colors=no_color),
    'openpilot.system.ui.lib.application': dict(gui_app=Bag(target_fps=60)),
    'openpilot.system.ui.lib.polygon': dict(draw_polygon=draw_polygon, Gradient=Gradient),
    'openpilot.system.ui.widgets': dict(Widget=Widget),
    'openpilot.selfdrive.ui.sunnypilot.onroad.model_renderer': dict(ChevronMetrics=ChevronMetrics, ModelRendererSP=ModelRendererSP),
    'openpilot.selfdrive.ui.sunnypilot.mici.onroad.model_renderer': dict(ModelRendererSP=ModelRendererSPMici),
  }
  for name, attrs in modules.items():
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
  # in import order: the status fade and the base renderer under their own names, for the mici renderer to import
  module = None
  for path in ('openpilot/selfdrive/ui/mici/onroad/status.py', 'openpilot/selfdrive/ui/onroad/model_renderer.py',
               'openpilot/selfdrive/ui/mici/onroad/model_renderer.py'):
    name = path.removesuffix('.py').replace('/', '.')
    spec = importlib.util.spec_from_file_location(name, VENDOR / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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
    velocity=xyz(plan[:, 3], plan[:, 4], plan[:, 5]),
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
    selfdriveState=Bag(enabled=True, engageable=True), carState=Bag(brakePressed=False, vEgo=float(h['plan'][0, 3])))
  renderer._render(rect)
  # the fade is settled: each picture is one frame of the model's, not 60 of the display's
  for lead in renderer._lead_bars:
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
      selfdriveState=Bag(enabled=True, engageable=True), carState=Bag(brakePressed=False, vEgo=float(h['plan'][0, 3])))
    for _ in range(3):
      renderer._render(rect)
    bars = [{'points': numbers(v.bar), 'alpha': int(255 * v.fade_filter.x)} for v in renderer._lead_bars if len(v.bar) >= 3 and int(255 * v.fade_filter.x) > 0]
    frames.append({'lead': numbers(h['lead']), 'lead_prob': numbers(h['lead_prob']), 'bars': bars})
  return {'base': {k: numbers(v) for k, v in base.items()}, 'frames': frames}


def stopping(v0, d_stop, n=N):
  """A plan that slows evenly from v0 to a stop d_stop ahead and stays there: position and velocity, in openpilot's shapes."""
  a = v0 ** 2 / (2 * d_stop) if d_stop > 0 else 0.0
  t_stop = v0 / a if a > 0 else 0.0
  t = np.minimum(T_IDXS, t_stop)
  plan = np.zeros((n, 15))
  plan[:, 0] = v0 * t - 0.5 * a * t ** 2
  plan[:, 3] = np.where(T_IDXS < t_stop, v0 - a * T_IDXS, 0.0)
  plan[:, 6] = np.where(T_IDXS < t_stop, -a, 0.0)
  return f32(plan)


class SmOnce(Sm):
  """modelV2 updated on the first of the display's renders of a model frame only, as the display has it."""
  def __init__(self, first, **kw):
    super().__init__(**kw)
    self.updated = types.MappingProxyType({'carParams': False, 'modelV2': first, 'radarState': True})


def stop_sequence(module, e2e, transform):
  """The stop bar over model frames, each rendered 3 times: a stop comes into range and closes in, a lead stops just past it
  and hides it, the lead goes and it comes back, the lane lines fade and its ends go to the 3.6 m lane, and the car stops."""
  base = heads(np.random.default_rng(2))
  renderer = module.ModelRenderer()
  renderer._longitudinal_control = True
  renderer.set_transform(transform)
  rect = Rectangle(0.0, 0.0, float(W), float(H))
  renderer._rect = rect
  frames = []
  v, d = 15.0, 70.0
  for k in range(140):
    v, d = max(v - 0.05, 0.0), max(d - v / 20, 0.0)
    leads = [(0.9, d + 1.52 + 6.0, 0.0)] if 50 <= k < 70 else []
    probs = (0.8, 0.95, 0.95, 0.7) if k < 90 else (0.1, 0.2, 0.3, 0.0)
    h = dict(base)
    h['plan'] = stopping(v, d) if k < 120 else stopping(0.0, 0.0)
    lead, lead_prob = np.zeros((3, 6, 4)), np.zeros(3)
    for i, (p, dl, y) in enumerate(leads):
      lead_prob[i], lead[i, 0, 0], lead[i, 0, 1] = p, dl, y
    h['lead'], h['lead_prob'] = f32(lead), f32(lead_prob)
    prob = np.zeros(8)
    prob[1::2] = probs
    h['lane_lines_prob'] = f32(prob)
    for r in range(3):
      UI_STATE.sm = SmOnce(
        r == 0, modelV2=model_v2(h), extrinsicsCalibration=Bag(height=[HEIGHT]), carParams=Bag(openpilotLongitudinalControl=True),
        radarState=Bag(leadOne=vision_lead(h, 0), leadTwo=vision_lead(h, 1)), longitudinalPlan=Bag(longitudinalPlanSource=e2e),
        selfdriveState=Bag(enabled=True, engageable=True), carState=Bag(brakePressed=False, vEgo=float(h['plan'][0, 3])))
      renderer._render(rect)
    bars = [{'points': numbers(b.bar), 'alpha': int(255 * b.fade_filter.x)} for b in (*renderer._lead_bars, renderer._stop_point)
            if len(b.bar) >= 3 and int(255 * b.fade_filter.x) > 0]
    frames.append({'plan': numbers(h['plan']), 'lead': numbers(h['lead']), 'lead_prob': numbers(h['lead_prob']),
                   'lane_lines_prob': numbers(h['lane_lines_prob']), 'bars': bars, 'stop': renderer._stop_point.visible})
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
  out['stop_sequence'] = stop_sequence(module, e2e, transform)
  shown = [f['stop'] for f in out['stop_sequence']['frames']]
  if not any(shown) or all(shown):
    sys.exit('stop_sequence: the stop bar should come and go')
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
