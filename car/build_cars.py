"""
Diorama's own cars: two original designs, built in Blender from nothing but the numbers in this file. Neither follows
an existing make, model or artist's work; nothing is imported or traced.

- `sedan`: a three-box luxury saloon: long bonnet, cabin set back, a rear window down to a flat boot lid.
- `suv`: a two-box SUV: tall, a high bonnet, the roof carried back to an upright tailgate.

Run it in Blender (5.x) on car/diorama_car_design.blend, from the repo root:

    /Applications/Blender.app/Contents/MacOS/Blender -b car/diorama_car_design.blend -P car/build_cars.py -- \
        --level 4 --export car --save

or from Blender's own Python, with the file open:

    P = 'car/build_cars.py'; exec(compile(open(P).read(), P, 'exec'), {'ARGS': {'level': 4, 'export': 'car', 'save': True}})

It empties the file and builds each car as objects in its own collection (`Sedan`, `SUV`; x forward, y left, z up,
metres), side by side, with a camera, sun and floor in `Stage` and the bodies' control cages, hidden, in
`Construction`. --export DIR writes diorama-assets-sedan.bin and diorama-assets-suv.bin there: each is the asset the
app loads, in the format of car/assets.py's write_library (its one entry named `car`, about the origin, y to the
right); --level is the bodies' subdivision (3 is lighter, 4 is the assets'); --only NAME builds one.

How a car is made:
- the body is one subdivision surface over a cage of 15 rings (DESIGNS, below);
- it becomes a skin 10 mm thick, and everything that opens it is cut through: the glass, the wheel arches, the lamps,
  the doors' and bonnet's and boot's edges. A cut shows the skin's edge, as a seal;
- what fills a cut is the same surface, a few millimetres under: the glass (opaque), the lamps, the handles;
- the wheels, the mirrors and the SUV's arch lips are their own pieces.
Only what is seen from outside is made: there is no interior and no underside.
"""
import json
import math
import struct
import sys
import time

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

A = dict(globals().get('ARGS', {}))
if '--' in sys.argv:
  _a = sys.argv[sys.argv.index('--') + 1:]
  for _k in ('level', 'export', 'only'):
    if '--' + _k in _a:
      A[_k] = _a[_a.index('--' + _k) + 1]
  A['save'] = '--save' in _a
LEVEL = int(A.get('level', 3))
T0 = time.time()
LOG = []


def log(*a):
  LOG.append(f'{time.time() - T0:6.1f}s ' + ' '.join(str(x) for x in a))
  print(LOG[-1])


# ---------------------------------------------------------------- roles: the colours the app draws (car/assets.py)
TAIL = (0.80, 0.10, 0.10)
ROLES = {  # role: (colour, emission); emission -1 marks the paint, which the app colours per car
  'body': ((0.62, 0.62, 0.64), -1.0), 'glass': ((0.33, 0.33, 0.35), 0.0), 'trim': ((0.30, 0.30, 0.32), 0.0),
  'tire': ((0.22, 0.22, 0.23), 0.0), 'rim': ((0.40, 0.40, 0.42), 0.0), 'well': ((0.20, 0.20, 0.21), 0.0), 'seal': ((0.30, 0.30, 0.32), 0.0),
  'bright': ((0.80, 0.81, 0.83), 0.0),
  'tail': (TAIL, 1.0), 'tail_dark': (tuple(c * 0.42 for c in TAIL), 0.5), 'housing': (tuple(c * 0.38 for c in TAIL), 0.6),
  'reflector': (tuple(c * 0.8 for c in TAIL), 0.0), 'head': ((0.88, 0.89, 0.91), 0.5),
}

# ---------------------------------------------------------------- the two designs
# A body is a cage of 15 rings, nose to tail. A ring runs from the underside's centre line, round the side, to the
# roof's centre line (lines 0-10); the other side is its mirror. The body is the cage's subdivision surface.
#   x: station; w: half width; zb: underside; zbelt: belt (the glass's base, the wing's top at the ends);
#   ztop: centre line; wr: the roof edge's half width; dzr: its drop; bow: how far the upper lines bow forward;
#   inset: the belt, in from the widest line; zmid: the flank's widest line; shd: the shoulder, below the belt
SHD = [0.03, 0.035, 0.04, 0.045, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.055, 0.055, 0.05, 0.045, 0.04]
INSET = [0.05, 0.06, 0.075, 0.08, 0.085, 0.085, 0.085, 0.085, 0.085, 0.095, 0.105, 0.10, 0.08, 0.07, 0.05]
DESIGNS = {
  # a three-box saloon: long bonnet, cabin set back, a rear window down to a flat boot lid
  'sedan': {
    'st': {
      'x':    [2.47, 2.37, 2.17, 1.84, 1.46, 1.00, 0.57, 0.14, -0.45, -1.20, -1.58, -1.88, -2.3, -2.4, -2.47],
      'w':    [0.58, 0.79, 0.90, 0.945, 0.95, 0.94, 0.935, 0.93, 0.93, 0.945, 0.96, 0.945, 0.9, 0.82, 0.66],
      'zb':   [0.32, 0.24, 0.18, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.17, 0.19, 0.27, 0.33, 0.43],
      'zbelt': [0.63, 0.72, 0.80, 0.86, 0.905, 0.935, 0.945, 0.955, 0.965, 0.98, 1.00, 1.02, 1.03, 0.97, 0.77],
      'ztop': [0.66, 0.76, 0.86, 0.92, 0.965, 1.01, 1.24, 1.415, 1.455, 1.42, 1.28, 1.10, 1.075, 1.01, 0.8],
      'wr':   [0.34, 0.42, 0.48, 0.50, 0.52, 0.66, 0.68, 0.61, 0.61, 0.60, 0.61, 0.66, 0.66, 0.57, 0.43],
      'dzr':  [0.01, 0.022, 0.035, 0.04, 0.035, 0.02, 0.09, 0.05, 0.045, 0.05, 0.07, 0.03, 0.015, 0.01, 0.01],
      'bow':  [0.0, 0.04, 0.04, 0.03, 0.04, 0.14, 0.11, 0.08, 0.0, -0.04, -0.09, -0.12, 0.0, 0.0, 0.0],
      'zmid': [0.52, 0.56, 0.61, 0.66, 0.70, 0.72, 0.73, 0.74, 0.75, 0.76, 0.78, 0.79, 0.78, 0.74, 0.64],
      'inset': INSET, 'shd': SHD,
    },
    'rings': ((12, 0.35), (5, 0.4), (11, 0.4)),  # creased across: the boot's rear edge, the windscreen's base, the rear window's
    'axles': (1.57, -1.47), 'rw': 0.37, 'ra': 0.42, 'track': 0.805, 'rim': 0.275, 'front_dz': -0.01,
    'glass': ((5.14, 6.93, 8.24, 11.76, 0.07), (7.35, 8.50, 8.80, 11.20, 0.06), (9.12, 10.90, 8.30, 11.70, 0.07)),  # windscreen, sunroof, rear window
    'side': (5.62, 10.32, 6.14, 7.74), 'stop': (9.17, 9.225), 'gate': (11.08, 0.22), 'lips': 0.0,
    'spokes': (10, 0.013, 0.010, False), 'place': 1.7, 'surround': 0.013, 'handles': 'bright', 'fin': 8.85,
  },
  # a two-box SUV: tall, a high bonnet, the roof carried back to an upright tailgate
  'suv': {
    'st': {
      'x':    [2.35, 2.24, 2.05, 1.78, 1.42, 1.05, 0.66, 0.28, -0.50, -1.30, -1.90, -2.18, -2.25, -2.31, -2.35],
      'w':    [0.62, 0.81, 0.92, 0.96, 0.965, 0.955, 0.945, 0.94, 0.94, 0.955, 0.965, 0.945, 0.895, 0.83, 0.68],
      'zb':   [0.40, 0.31, 0.25, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.24, 0.27, 0.32, 0.38, 0.50],
      'zbelt': [0.77, 0.88, 0.97, 1.025, 1.065, 1.09, 1.10, 1.11, 1.12, 1.13, 1.15, 1.15, 1.08, 0.99, 0.83],
      'ztop': [0.80, 0.92, 1.02, 1.08, 1.125, 1.17, 1.44, 1.645, 1.705, 1.68, 1.60, 1.19, 1.11, 1.02, 0.86],
      'wr':   [0.36, 0.44, 0.50, 0.52, 0.54, 0.68, 0.70, 0.63, 0.63, 0.62, 0.60, 0.65, 0.62, 0.55, 0.43],
      'dzr':  [0.01, 0.022, 0.035, 0.04, 0.035, 0.02, 0.10, 0.055, 0.05, 0.05, 0.05, 0.03, 0.015, 0.01, 0.01],
      'bow':  [0.0, 0.04, 0.04, 0.03, 0.04, 0.13, 0.10, 0.07, 0.0, 0.0, -0.02, -0.05, 0.0, 0.0, 0.0],
      'zmid': [0.62, 0.69, 0.75, 0.80, 0.83, 0.85, 0.86, 0.87, 0.88, 0.89, 0.91, 0.91, 0.86, 0.80, 0.7],
      'inset': INSET, 'shd': SHD,
    },
    'rings': ((10, 0.4), (5, 0.4), (12, 0.15)),  # the roof's rear edge, a spoiler; the windscreen's base; the tail's shelf
    'axles': (1.48, -1.42), 'rw': 0.395, 'ra': 0.455, 'track': 0.82, 'rim': 0.285,
    'glass': ((5.14, 6.93, 8.24, 11.76, 0.07), (7.10, 9.50, 8.30, 11.70, 0.07), (10.15, 10.90, 8.35, 11.65, 0.06)),  # windscreen, roof, rear window
    'side': (5.62, 10.36, 6.14, 7.74), 'stop': (10.18, 10.225), 'gate': (9.82, 0.22), 'lips': 0.045, 'rails': (7.25, 9.62, 8.09, 8.19),
    'spokes': (5, 0.031, 0.024, True), 'place': -1.7, 'fin': 9.68,
  },
}
ONLY = A.get('only')
NS, NL = 15, 11
ST, D, NAME = None, None, None
AXLES = RW = TRACK = RA = None


def use(name):
  """Make a design the one being built."""
  global ST, D, NAME, AXLES, RW, TRACK, RA
  NAME, D = name, dict(DESIGNS[name])
  D.update(A.get('d', {}).get(name, {}))
  ST = dict(D['st'])
  ST.update(A.get('st', {}).get(name, {}))
  AXLES, RW, TRACK, RA = D['axles'], D['rw'], D['track'], D['ra']


def ring(i):
  """Station i's control points, (x, y, z) for y >= 0."""
  x, w, zb, zt = ST['x'][i], ST['w'][i], ST['zb'][i], ST['ztop'][i]
  wr, zr, bow = ST['wr'][i], zt - ST['dzr'][i], ST['bow'][i]
  zsh = ST['zbelt'][i] - ST['shd'][i]
  l6 = (w - ST['inset'][i], ST['zbelt'][i])
  l8 = (wr, zr)
  l7 = (0.5 * (l6[0] + l8[0]) + 0.02, 0.5 * (l6[1] + l8[1]) + 0.005)
  yz = [
    (0.0, zb), (0.5 * (w - 0.10), zb), (w - 0.10, zb + 0.01), (w - 0.06, zb + 0.13), (w, ST['zmid'][i]),
    (w - 0.015, zsh), l6, l7, l8, (0.5 * wr, zt - 0.2 * ST['dzr'][i]), (0.0, zt),
  ]
  bows = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.75, 1.0]
  return [(x + bow * b, y, z) for (y, z), b in zip(yz, bows)]


CREASES = {'shoulder': 0.22, 'belt': 0.25, 'sill': 0.3, 'roof': 0.15, 'nose': 0.1, 'tail': 0.22, 'flank': 0.3, 'bonnet': 0.3}
CREASES.update(A.get('creases', {}))


def build_cage():
  bm = bmesh.new()
  R, L = [], []  # the +y and -y vertices of each station
  for i in range(NS):
    pts = ring(i)
    r = [bm.verts.new(p) for p in pts]
    l = [r[0]] + [bm.verts.new((x, -y, z)) for x, y, z in pts[1:-1]] + [r[-1]]
    R.append(r)
    L.append(l)
  for i in range(NS - 1):
    for j in range(NL - 1):
      for f in (bm.faces.new((R[i][j], R[i + 1][j], R[i + 1][j + 1], R[i][j + 1])), bm.faces.new((L[i][j], L[i][j + 1], L[i + 1][j + 1], L[i + 1][j]))):
        f.material_index = int(j < 3)
  for i, flip in ((0, False), (NS - 1, True)):  # the caps
    for j in range(NL - 1):
      if j == 0:
        f = [R[i][0], R[i][1], L[i][1]]
      elif j == NL - 2:
        f = [R[i][j], R[i][j + 1], L[i][j]]
      else:
        f = [R[i][j], R[i][j + 1], L[i][j + 1], L[i][j]]
      bm.faces.new(f[::-1] if flip else f).material_index = int(j < 3)
  bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
  cr = bm.edges.layers.float.new('crease_edge')

  def crease(line, value, stations=range(NS - 1)):
    for i in stations:
      for side in (R, L):
        e = bm.edges.get((side[i][line], side[i + 1][line]))
        if e:
          e[cr] = value

  crease(5, CREASES['shoulder'])
  crease(4, CREASES['flank'], range(3, 12))
  crease(9, CREASES['bonnet'], range(1, 5))
  crease(6, CREASES['belt'], range(4, 12))
  crease(2, CREASES['sill'])
  crease(8, CREASES['roof'], range(5, 12))
  for i, v in ((0, CREASES['nose']), (NS - 1, CREASES['tail'])):  # the caps' rims
    for side in (R, L):
      for j in range(NL - 1):
        e = bm.edges.get((side[i][j], side[i][j + 1]))
        if e:
          e[cr] = v
  for st, v in D['rings']:  # the rings creased across the top
    for side in (R, L):
      for j in range(6, NL - 1):
        e = bm.edges.get((side[st][j], side[st][j + 1]))
        if e:
          e[cr] = v
  me = bpy.data.meshes.new('cage')
  bm.to_mesh(me)
  bm.free()
  return me


# ---------------------------------------------------------------- helpers
def material(role):
  m = bpy.data.materials.get(role)
  if m is None:
    m = bpy.data.materials.new(role)
    col, emis = ROLES.get(role, ((1, 0, 1), 0))
    m.diffuse_color = (*col, 1)
    m.use_nodes = True
    b = m.node_tree.nodes.get('Principled BSDF')
    b.inputs['Base Color'].default_value = (*col, 1)
    b.inputs['Roughness'].default_value = 0.4 if role in ('body', 'glass') else 0.65
    if emis > 0:
      b.inputs['Emission Color'].default_value = (*col, 1)
      b.inputs['Emission Strength'].default_value = emis
  return m


def collection(name, hide=False):
  c = bpy.data.collections.get(name)
  if c is None:
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
  c.hide_viewport = c.hide_render = hide
  return c


def link(me, name, roles=(), coll=None, smooth=True):
  """An object of the mesh: in the car's collection, under its name, unless a collection is given."""
  car = NAME.upper() if NAME == 'suv' else NAME.capitalize()
  ob = bpy.data.objects.new(name if coll else f'{car}_{name}', me)
  collection(coll or car).objects.link(ob)
  for r in roles:
    me.materials.append(material(r))
  if smooth and len(me.polygons):
    me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
  return ob


def evaluated(ob, name):
  """A copy of the object's mesh with its modifiers applied."""
  bpy.context.view_layer.update()
  dg = bpy.context.evaluated_depsgraph_get()
  out = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
  out.name = name
  return out


def reset():
  for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob)
  for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights, bpy.data.curves):
    for b in list(coll):
      coll.remove(b)
  for c in list(bpy.data.collections):
    bpy.data.collections.remove(c)


def coords(me):
  co = np.empty(len(me.vertices) * 3)
  me.vertices.foreach_get('co', co)
  return co.reshape(-1, 3)


# ---------------------------------------------------------------- curves and outlines
def cr(pts, u):
  """Catmull-Rom through pts (n, d), at parameters u in [0, n - 1]."""
  pts = np.asarray(pts, float)
  n = len(pts)
  u = np.clip(np.atleast_1d(np.asarray(u, float)), 0, n - 1)
  i = np.minimum(u.astype(int), n - 2)
  t = (u - i)[:, None]
  ext = np.concatenate([[2 * pts[0] - pts[1]], pts, [2 * pts[-1] - pts[-2]]])
  p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
  return 0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3)


def fillet(pts, r, n=8):
  """A closed 2-D polygon with its corners rounded: r is one radius or one per corner (0 leaves a corner as it is)."""
  pts = np.asarray(pts, float)
  m = len(pts)
  rs = np.broadcast_to(np.asarray(r, float), (m,))
  out = []
  for k in range(m):
    p, a, b = pts[k], pts[k - 1], pts[(k + 1) % m]
    da, db = a - p, b - p
    la, lb = np.linalg.norm(da), np.linalg.norm(db)
    cosv = np.clip(np.dot(da, db) / (la * lb), -1, 1)
    if rs[k] <= 0 or cosv < -0.9999:
      out.append(p)
      continue
    half = 0.5 * math.acos(cosv)
    t = min(rs[k] / math.tan(half), 0.45 * la, 0.45 * lb)
    rad = t * math.tan(half)
    ua, ub = da / la, db / lb
    bis = (ua + ub) / np.linalg.norm(ua + ub)
    c = p + bis * (rad / math.sin(half))
    a0, a1 = p + ua * t - c, p + ub * t - c
    ang0, ang1 = math.atan2(a0[1], a0[0]), math.atan2(a1[1], a1[0])
    d = (ang1 - ang0 + math.pi) % (2 * math.pi) - math.pi
    for q in np.linspace(0, 1, n + 1):
      out.append(c + rad * np.array([math.cos(ang0 + d * q), math.sin(ang0 + d * q)]))
  return np.array(out)


def densify(pts, step):
  """A closed polyline with no edge longer than step."""
  pts = np.asarray(pts, float)
  out = []
  for k in range(len(pts)):
    a, b = pts[k], pts[(k + 1) % len(pts)]
    n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
    out += [a + (b - a) * q / n for q in range(n)]
  return np.array(out)


def rrect(y0, y1, z0, z1, r):
  return fillet([(y0, z0), (y1, z0), (y1, z1), (y0, z1)], r)


def ribbon(path, w, r=0.0):
  """A path (2-D, open) as a closed outline w wide, its corners rounded to r."""
  p = np.asarray(path, float)
  if r:
    p = fillet(p, [0] + [r] * (len(p) - 2) + [0])
  q = [p[0]]
  for a, b in zip(p[:-1], p[1:]):
    n = max(1, int(math.ceil(np.linalg.norm(b - a) / 0.02)))
    q += [a + (b - a) * k / n for k in range(1, n + 1)]
  p = np.array(q)
  t = np.gradient(p, axis=0)
  t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
  nrm = np.c_[-t[:, 1], t[:, 0]]
  return np.r_[p + nrm * w / 2, (p - nrm * w / 2)[::-1]]


class Surface:
  """The body's surface: its mesh, for rays and nearest points, and the cage's lines on it, for placing things."""

  def __init__(self, me):
    self.me = me
    self.co = coords(me)
    me.calc_loop_triangles()
    tri = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get('vertices', tri)
    self.tri = tri.reshape(-1, 3)
    vn = np.empty(len(me.vertices) * 3, np.float32)
    me.vertex_normals.foreach_get('vector', vn)
    self.vn = vn.reshape(-1, 3).astype(float)
    self.bvh = BVHTree.FromPolygons([Vector(c) for c in self.co], [tuple(int(v) for v in t) for t in self.tri])
    # the cage's own vertices come first in its subdivision, and lie on the surface
    g = np.zeros((NS, 21, 3))
    for i in range(NS):
      for jj in range(21):
        j = jj if jj <= 10 else 20 - jj
        g[i, jj] = self.co[20 * i + j] if (jj <= 10 or j in (0, 10)) else self.co[20 * i + 11 + (j - 1)]
    self.grid = g

  def at(self, s, t):
    """Points at stations s and lines t (0-10 up the +y side, 10-20 down the other), on the surface."""
    s, t = np.atleast_1d(s).astype(float), np.atleast_1d(t).astype(float)
    rows = np.stack([cr(self.grid[i], t) for i in range(NS)])  # (NS, m, 3)
    p = np.array([cr(rows[:, k], s[k])[0] for k in range(len(s))])
    return self.snap(p)

  def snap(self, p):
    return np.array([self.bvh.find_nearest(Vector(q))[0] for q in p])

  def normal(self, p):
    """The smooth normal at the surface's nearest points."""
    out = np.zeros((len(p), 3))
    for k, q in enumerate(p):
      loc, _, idx, _ = self.bvh.find_nearest(Vector(q))
      a, b, c = self.co[self.tri[idx]]
      v0, v1, v2 = b - a, c - a, np.array(loc) - a
      d00, d01, d11, d20, d21 = v0 @ v0, v0 @ v1, v1 @ v1, v2 @ v0, v2 @ v1
      den = max(d00 * d11 - d01 * d01, 1e-20)
      v = (d11 * d20 - d01 * d21) / den
      w = (d00 * d21 - d01 * d20) / den
      n = (1 - v - w) * self.vn[self.tri[idx][0]] + v * self.vn[self.tri[idx][1]] + w * self.vn[self.tri[idx][2]]
      out[k] = n / max(np.linalg.norm(n), 1e-12)
    return out

  def region(self, s0, s1, t0, t1, r=0.05, step=0.02):
    """A rounded rectangle between stations s0-s1 and lines t0-t1, as a closed outline on the surface."""
    c = self.at([0.5 * (s0 + s1)] * 3 + [0.5 * (s0 + s1) + 0.05], [0.5 * (t0 + t1)] * 2 + [0.5 * (t0 + t1) + 0.05] + [0.5 * (t0 + t1)])
    ds, dt = np.linalg.norm(c[3] - c[0]) / 0.05, np.linalg.norm(c[2] - c[0]) / 0.05  # metres per unit of s and of t
    rs, rt = min(r / ds, 0.45 * (s1 - s0)), min(r / dt, 0.45 * (t1 - t0))
    pts = []
    for cs, ct, a0 in ((s1 - rs, t1 - rt, 0), (s0 + rs, t1 - rt, 90), (s0 + rs, t0 + rt, 180), (s1 - rs, t0 + rt, 270)):
      for a in np.radians(np.linspace(a0, a0 + 90, 9)):
        pts.append((cs + rs * math.cos(a), ct + rt * math.sin(a)))
    pts = np.array(pts)
    pts = densify(pts * (ds, dt), step) / (ds, dt)
    return self.at(pts[:, 0], pts[:, 1])

  def project(self, pts, axis, sign):
    """2-D points cast onto the surface along an axis ('x': points are (y, z); 'y': (x, z); 'z': (x, y)), from the
    sign side."""
    out = []
    for a, b in pts:
      o = {'x': (3.5 * sign, a, b), 'y': (a, 3.5 * sign, b), 'z': (a, b, 3.5 * sign)}[axis]
      d = {'x': (-sign, 0, 0), 'y': (0, -sign, 0), 'z': (0, 0, -sign)}[axis]
      hit = self.bvh.ray_cast(Vector(o), Vector(d))[0]
      if hit is None:
        raise ValueError(f'nothing of the body at {a:.3f}, {b:.3f} along {axis}')
      out.append(hit)
    return np.array(out)

  def offset(self, d, role):
    """The surface moved d along its normals, a solid of one material."""
    me = self.me.copy()
    me.materials.clear()
    me.materials.append(material(role))
    me.vertices.foreach_set('co', (self.co + self.vn * d).ravel())
    me.polygons.foreach_set('material_index', [0] * len(me.polygons))
    me.update()
    return me


class Outline:
  """A closed outline on the body, and the way a cutter goes through it."""

  def __init__(self, pts, normal=None, surf=None, out=0.5, inn=0.06):
    self.p = np.asarray(pts, float)
    n = np.asarray(normal, float) if normal is not None else surf.normal(self.p).mean(axis=0)
    self.n = n / np.linalg.norm(n)
    self.out, self.inn = out, inn

  def grown(self, m):
    """The outline moved m outward, across the cutter's direction."""
    t = np.roll(self.p, -1, axis=0) - np.roll(self.p, 1, axis=0)
    o = np.cross(t, self.n)
    o /= np.maximum(np.linalg.norm(o, axis=1, keepdims=True), 1e-12)
    # which way is out: by the outline's turning about the cutter's direction
    u = np.cross(self.n, (1.0, 0.0, 0.0) if abs(self.n[0]) < 0.9 else (0.0, 1.0, 0.0))
    u /= np.linalg.norm(u)
    v = np.cross(self.n, u)
    x, y = self.p @ u, self.p @ v
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return Outline(self.p + o * m * (1 if area > 0 else -1), self.n, out=self.out, inn=self.inn)

  def mirrored(self):
    return Outline(self.p * (1, -1, 1), self.n * (1, -1, 1), out=self.out, inn=self.inn)


def between(outer, inner):
  """The band between two outlines with matching points (one is the other, grown), as one outline: round the outer,
  across, and back round the inner, with a hair's gap where it would close."""
  end = lambda p: p[-1] + 0.9 * (p[0] - p[-1])
  return Outline(np.r_[outer.p, [end(outer.p), end(inner.p)], inner.p[::-1]], outer.n, out=outer.out, inn=outer.inn)


def prisms(outlines, role='seal', name='cutter'):
  """A mesh of the outlines' cutters: each outline swept along its direction, out of the body and into it."""
  bm = bmesh.new()
  for o in outlines:
    # the outline turning anticlockwise about its direction, so the faces below face outward
    u = np.cross(o.n, (1.0, 0.0, 0.0) if abs(o.n[0]) < 0.9 else (0.0, 1.0, 0.0))
    u /= np.linalg.norm(u)
    v = np.cross(o.n, u)
    x, y = o.p @ u, o.p @ v
    p = o.p if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0 else o.p[::-1]
    top = [bm.verts.new(q) for q in p + o.n * o.out]
    bot = [bm.verts.new(q) for q in p - o.n * o.inn]
    m = len(top)
    bm.faces.new(top)
    bm.faces.new(bot[::-1])
    for k in range(m):
      bm.faces.new((top[k], bot[k], bot[(k + 1) % m], top[(k + 1) % m]))
  me = bpy.data.meshes.new(name)
  bm.to_mesh(me)
  bm.free()
  me.materials.append(material(role))
  return me


def boolean(ob, cutter, operation, solver=None):
  c = link(cutter, cutter.name, coll='Construction', smooth=False)
  mod = ob.modifiers.new('bool', 'BOOLEAN')
  mod.operation = operation
  mod.object = c
  mod.solver = solver or A.get('solver', 'MANIFOLD')
  if hasattr(mod, 'material_mode'):
    mod.material_mode = 'TRANSFER'
  return c


def without(me, roles):
  """The mesh less its faces of these materials (and its loose vertices)."""
  names = [m.name for m in me.materials]
  drop = {i for i, n in enumerate(names) if n in roles}
  bm = bmesh.new()
  bm.from_mesh(me)
  bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.material_index in drop], context='FACES')
  bm.to_mesh(me)
  bm.free()
  return me


def patch(surf, outlines, depth, role, name, grow=0.0):
  """The surface, `depth` along its normals, inside the outlines: a piece that fills a cut, or lies on the body."""
  ob = link(surf.offset(depth, role), name + '_src', coll='Construction')
  tmp = boolean(ob, prisms([o.grown(grow) if grow else o for o in outlines], '__cut'), 'INTERSECT')
  me = without(evaluated(ob, name), {'__cut'})
  bpy.data.objects.remove(ob)
  bpy.data.objects.remove(tmp)
  return me


def circle(cx, cz, r, n=96):
  a = np.linspace(0, 2 * math.pi, n, endpoint=False)
  return np.c_[cx + r * np.cos(a), cz + r * np.sin(a)]


def finish(ob, surf=None, angle=35):
  """Shading: hard edges where faces meet at more than `angle`; and where a face lies along the body's surface, the
  surface's own smooth normals, so a cut leaves no mark in the shading."""
  me = ob.data
  me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
  me.set_sharp_from_angle(angle=math.radians(angle))
  me.update()
  if surf is None:
    return
  n = len(me.loops)
  cn = np.empty(n * 3, np.float32)
  me.corner_normals.foreach_get('vector', cn)
  cn = cn.reshape(-1, 3).astype(float)
  lv = np.empty(n, np.int32)
  me.loops.foreach_get('vertex_index', lv)
  fn = np.empty(len(me.polygons) * 3, np.float32)
  me.polygons.foreach_get('normal', fn)
  ls, lt = np.empty(len(me.polygons), np.int32), np.empty(len(me.polygons), np.int32)
  me.polygons.foreach_get('loop_start', ls)
  me.polygons.foreach_get('loop_total', lt)
  lf = np.repeat(fn.reshape(-1, 3), lt, axis=0)  # each corner's face normal
  sn = surf.normal(coords(me))[lv]
  sn *= np.sign(np.sum(sn * lf, axis=1, keepdims=True))  # a patch may face either way
  along = np.sum(sn * lf, axis=1) > math.cos(math.radians(40))
  cn[along] = sn[along]
  me.normals_split_custom_set([tuple(v) for v in cn])
  me.update()


# ---------------------------------------------------------------- the asset (car/assets.py's format)
def triangles(ob):
  """An object's triangles as corners: positions, normals and roles (y to the right, as the app has it)."""
  me = ob.data
  me.calc_loop_triangles()
  n = len(me.loop_triangles)
  loops = np.empty(n * 3, np.int32)
  me.loop_triangles.foreach_get('loops', loops)
  vidx = np.empty(n * 3, np.int32)
  me.loop_triangles.foreach_get('vertices', vidx)
  mat = np.empty(n, np.int32)
  me.loop_triangles.foreach_get('material_index', mat)
  cn = np.empty(len(me.loops) * 3, np.float32)
  me.corner_normals.foreach_get('vector', cn)
  M = np.array(ob.matrix_world)
  pos = coords(me)[vidx] @ M[:3, :3].T + M[:3, 3]
  nrm = cn.reshape(-1, 3)[loops] @ np.linalg.inv(M[:3, :3])
  nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
  names = [s.material.name if s.material else 'body' for s in ob.material_slots] or ['body']
  roles = np.array(names)[np.repeat(mat, 3)]
  pos[:, 1] *= -1
  nrm[:, 1] *= -1
  return pos, nrm, roles


def write_asset(path, objects, dims, span=4.0, limit=65535):
  pos, nrm, roles = (np.concatenate(a) for a in zip(*[triangles(o) for o in objects]))
  col = np.zeros((len(pos), 3))
  emis = np.zeros(len(pos))
  for r in np.unique(roles):
    col[roles == r], emis[roles == r] = ROLES[r]
  v = np.zeros((len(pos), 16), np.uint8)
  v[:, 0:6] = np.round(np.clip(pos / span, -1, 1) * 32767).astype('<i2').view(np.uint8).reshape(-1, 6)
  v[:, 6:9] = np.round(np.clip(nrm, -1, 1) * 127).astype(np.int8).view(np.uint8)
  v[:, 9] = np.round(np.clip(emis, -1, 1) * 127).astype(np.int8).view(np.uint8)
  v[:, 10:13] = np.round(np.clip(col, 0, 1) * 255).astype(np.uint8)
  tri = v.reshape(-1, 48)
  area = np.linalg.norm(np.cross(pos[1::3] - pos[0::3], pos[2::3] - pos[0::3]), axis=1)
  tri = tri[area > 1e-10]  # no slivers of nothing
  v = tri.reshape(-1, 16)
  verts, idx, entry, v_at, i_at, start = [], [], {'dims': [round(float(d), 3) for d in dims], 'span': span, 'parts': []}, 0, 0, 0
  key = np.ascontiguousarray(v).view([('a', '<u8'), ('b', '<u8')]).ravel()
  while start < len(v):  # parts under 65,536 vertices, for 16-bit indices
    chunk = key[start:start + 18 * limit]
    u, first, inv = np.unique(chunk, return_index=True, return_inverse=True)
    if len(u) > limit:  # up to the triangle where one vertex too many first appears
      chunk = chunk[:3 * (np.sort(first)[limit] // 3)]
      u, inv = np.unique(chunk, return_inverse=True)
    entry['parts'].append({'vertices': len(u), 'indices': len(inv), 'vertexOffset': v_at, 'indexOffset': i_at})
    verts.append(u.view(np.uint8).ravel())
    idx.append(inv.astype('<u2'))
    v_at += len(u)
    i_at += len(inv)
    start += len(chunk)
  head = json.dumps({'car': entry}, separators=(',', ':')).encode()
  head += b' ' * (-len(head) % 4)
  with open(path, 'wb') as fh:
    fh.write(struct.pack('<I', len(head)) + head + np.concatenate(verts).tobytes() + np.concatenate(idx).tobytes())
  return {'vertices': v_at, 'triangles': i_at // 3, 'parts': len(verts), 'bytes': 4 + len(head) + v_at * 16 + i_at * 2}


# ---------------------------------------------------------------- the car
SKIN = 0.010  # the body's thickness, which a cut shows as its edge


def details(surf, name):
  """What is particular to a design at its nose and tail: the outlines cut through the skin, and what fills them
  (outlines, depth under the surface, role, how far the fill runs under the cut's edge)."""
  both = lambda o: [o, o.mirrored()]
  rear = lambda pts, **k: Outline(surf.project(densify(pts, 0.02), 'x', -1), (-1, 0, 0), **k)
  front = lambda pts, **k: Outline(surf.project(densify(pts, 0.02), 'x', 1), (1, 0, 0), **k)
  up = lambda pts: np.asarray(pts, float) + (0, D.get('front_dz', 0.0))  # the lamps follow the nose's height
  if name == 'sedan':
    # the tail: a lamp each side of the boot, a bright strip between them, the plate under it
    house = both(rear(fillet([(0.27, 0.865), (0.77, 0.875), (0.77, 0.955), (0.27, 0.955)], [0.012, 0.035, 0.035, 0.012])))
    lit = both(rear(rrect(0.30, 0.74, 0.922, 0.940, 0.007))) + both(rear(rrect(0.58, 0.74, 0.885, 0.905, 0.007)))
    dark = both(rear(rrect(0.30, 0.55, 0.883, 0.907, 0.008)))
    strip = [rear(rrect(-0.255, 0.255, 0.925, 0.938, 0.004))]
    plate = [rear(rrect(-0.27, 0.27, 0.765, 0.885, 0.02))]
    refl = both(rear(rrect(0.50, 0.74, 0.60, 0.624, 0.010)))
    pipes = both(rear(rrect(0.40, 0.66, 0.46, 0.515, 0.022)))
    tips = [o.grown(0.009) for o in pipes]
    frames = [between(o.grown(-0.004), o.grown(-0.011)) for o in house]  # a lit line round each lamp
    # the nose: a lamp each side of a wide grille, an intake under it and one at each corner
    lamp = both(front(up(fillet([(0.44, 0.655), (0.85, 0.690), (0.85, 0.748), (0.44, 0.716)], [0.012, 0.02, 0.02, 0.012]))))
    drl = both(front(up(fillet([(0.46, 0.700), (0.83, 0.729), (0.83, 0.740), (0.46, 0.711)], 0.004))))
    grille = [front(fillet([(-0.34, 0.505), (0.34, 0.505), (0.39, 0.705), (-0.39, 0.705)], 0.04))]
    bars = [front(rrect(-0.36, 0.36, z, z + 0.012, 0.0)) for z in (0.545, 0.585, 0.625, 0.665)]
    intake = [front(rrect(-0.50, 0.50, 0.375, 0.445, 0.03))] + both(front(fillet([(0.56, 0.50), (0.80, 0.53), (0.80, 0.60), (0.56, 0.57)], 0.02)))
    lens = []
    for yc in (0.52, 0.63, 0.74):  # three lenses under each lamp's strip
      z0 = 0.665 + (yc - 0.44) * 0.085
      lens += both(front(up(rrect(yc - 0.04, yc + 0.04, z0, z0 + 0.02, 0.006))))
    slats = [front(rrect(-0.47, 0.47, z, z + 0.008, 0.0)) for z in (0.398, 0.422)]
    blades = both(front(fillet([(0.585, 0.532), (0.775, 0.556), (0.775, 0.566), (0.585, 0.542)], 0.003)))
    cuts = house + strip + plate + refl + tips + lamp + grille + intake
    fills = [(house, -0.006, 'housing', 0.005), (lit, -0.003, 'tail', 0), (dark, -0.0045, 'tail_dark', 0), (strip, -0.003, 'bright', 0.004),
             (plate, -0.004, 'body', 0.005), (refl, -0.004, 'reflector', 0.005), (pipes, -0.012, 'well', 0.002), ([between(t, p) for t, p in zip(tips, pipes)], -0.002, 'bright', 0), (frames, -0.003, 'tail', 0),
             (lens, -0.003, 'head', 0), (slats, -0.005, 'trim', 0), (blades, -0.004, 'bright', 0),
             (lamp, -0.006, 'trim', 0.005), (drl, -0.003, 'head', 0), (grille + intake, -0.010, 'well', 0.005), (bars, -0.004, 'bright', 0)]
  else:
    # the tail: one band of lamps across the tailgate, under the rear window
    band = [rear(rrect(-0.76, 0.76, 0.975, 1.065, 0.04))]
    lit = [rear(rrect(-0.73, 0.73, 1.030, 1.047, 0.008))] + both(rear(rrect(0.47, 0.71, 0.999, 1.013, 0.006)))
    dark = both(rear(rrect(0.44, 0.73, 0.990, 1.022, 0.012)))
    plate = [rear(rrect(-0.27, 0.27, 0.80, 0.92, 0.02))]
    refl = both(rear(rrect(0.50, 0.74, 0.665, 0.69, 0.010)))
    slot = [rear(rrect(-0.42, 0.42, 0.545, 0.59, 0.02))]
    # the nose: a blade of lamp each side, joined by a thin strip; an intake below, a slot at each corner
    blade = both(front(up(fillet([(0.36, 0.795), (0.84, 0.832), (0.84, 0.878), (0.36, 0.826)], [0.006, 0.014, 0.014, 0.006]))))
    join_ = [front(up(rrect(-0.35, 0.35, 0.803, 0.817, 0.005)))]
    intake = [front(rrect(-0.50, 0.50, 0.435, 0.525, 0.03))] + both(front(rrect(0.71, 0.755, 0.47, 0.70, 0.018)))
    cuts = band + plate + refl + slot + blade + join_ + intake
    fills = [(band, -0.006, 'housing', 0.005), (dark, -0.0045, 'tail_dark', 0), (lit, -0.003, 'tail', 0),
             (plate, -0.004, 'body', 0.005), (refl, -0.004, 'reflector', 0.005), (blade + join_, -0.004, 'head', 0.005),
             (slot + intake, -0.009, 'well', 0.005)]
  return cuts, fills


def build_body(surf):
  """The body: the surface as a skin with the windows, arches and lamps cut out of it."""
  both = lambda o: [o, o.mirrored()]
  flank = lambda pts, **k: Outline(surf.project(densify(pts, 0.02), 'y', 1), (0, 1, 0), **k)
  xs = surf.grid[:, 6, 0]
  belt = lambda x: float(np.interp(x, xs[::-1], surf.grid[::-1, 6, 2]))  # the belt's height at x
  station = lambda x: float(np.interp(x, xs[::-1], np.arange(NS)[::-1]))
  nose, tail, cowl = ST['x'][0], ST['x'][-1], ST['x'][5]
  # the glass: windscreen, roof, rear window, and each side's
  glass = [Outline(surf.region(*g), surf=surf) for g in D['glass']]
  side = Outline(surf.region(*D['side'], 0.045), surf=surf)
  glass += [side, side.mirrored()]
  openings, surround = list(glass), []
  if D.get('surround'):  # a bright frame round each side's glass, let into the skin
    wide = side.grown(D['surround'])
    openings[-2:] = [wide, wide.mirrored()]
    surround = [between(wide, side), between(wide.mirrored(), side.mirrored())]
  stop = Outline(surf.region(*D['stop'], 9.25, 10.75, 0.012), surf=surf)  # the high brake lamp, in the rear window
  # the wheel arches
  arches = []
  for ax in AXLES:
    for sgn in (1, -1):
      c = circle(ax, RW + 0.01, RA)
      arches.append(Outline(np.c_[c[:, 0], np.full(len(c), 1.2 * sgn), c[:, 1]], (0, sgn, 0), out=0.0, inn=0.75))
  lamps, fills = details(surf, NAME)
  # the doors' edges: round the front door and the rear, and the pillar between them; their handles, flush
  zs = ST['zb'][7] + 0.155
  xf, xb = cowl - 0.19, cowl - 1.29
  ph = np.radians(np.linspace(62, -4, 12))
  arch = [(AXLES[1] + (RA + 0.065) * math.cos(a), RW + 0.01 + (RA + 0.065) * math.sin(a)) for a in ph]
  xc = arch[0][0]
  doors = both(flank(ribbon([(xf, belt(xf) + 0.03), (xf + 0.02, 0.5 * (zs + belt(xf))), (xf, zs), (arch[-1][0], zs)], 0.008, 0.05), inn=0.05))
  doors += both(flank(ribbon([(xc + 0.005, belt(xc) + 0.05)] + arch, 0.008), inn=0.05))
  pillar = both(flank(ribbon([(xb + 0.02, belt(xb) + 0.045), (xb, zs + 0.004)], 0.008), inn=0.05))
  zh = belt(xb) - 0.085
  handles = both(flank(rrect(xb + 0.08, xb + 0.23, zh, zh + 0.026, 0.012))) + both(flank(rrect(xc + 0.10, xc + 0.25, zh + 0.012, zh + 0.038, 0.012)))
  flap = [flank(rrect(tail + 0.33, tail + 0.46, belt(tail + 0.4) - 0.24, belt(tail + 0.4) - 0.11, 0.025))]  # the charging flap, on the left
  bars = both(flank(rrect(xb + 0.085, xb + 0.225, zh + 0.007, zh + 0.019, 0.005))) + both(flank(rrect(xc + 0.105, xc + 0.245, zh + 0.019, zh + 0.031, 0.005)))
  lamps = lamps + handles + flap
  # the bonnet's edge, from above; the boot's or tailgate's, along the roof's edges and across
  bonnet = [Outline(surf.project(ribbon([(cowl + 0.03, 0.80), (nose - 0.42, 0.775), (nose - 0.14, 0.60), (nose - 0.14, -0.60), (nose - 0.42, -0.775), (cowl + 0.03, -0.80)], 0.008, 0.12), 'z', 1), (0, 0, 1), inn=0.05)]
  g0, gr = D['gate']
  st = fillet([(12.0, 8.04), (g0, 8.04), (g0, 11.96), (12.0, 11.96)], [0, gr, gr, 0])
  st = np.concatenate([np.linspace(a, b, max(2, int(np.abs(b - a).max() / 0.04)), endpoint=False) for a, b in zip(st[:-1], st[1:])] + [st[-1:]])
  p = surf.at(st[:, 0], st[:, 1])
  o = np.cross(np.gradient(p, axis=0), surf.normal(p))
  o *= 0.004 / np.maximum(np.linalg.norm(o, axis=1, keepdims=True), 1e-12)
  gate = [Outline(np.r_[p + o, (p - o)[::-1]], (0, 0, 1), inn=0.05)]
  # a lip round each arch, down to the sill
  lips = []
  if D['lips']:
    for ax in AXLES:
      ph = np.radians(np.linspace(-13, 193, 60))
      ring_ = [(ax + (RA + D['lips']) * math.cos(a), RW + 0.01 + (RA + D['lips']) * math.sin(a)) for a in ph] + [(ax + (RA - 0.005) * math.cos(a), RW + 0.01 + (RA - 0.005) * math.sin(a)) for a in ph[::-1]]
      lips += both(flank(np.array(ring_)))
  # the pillars over the side glass
  strips = []
  for x0, x1 in ((xb - 0.035, xb + 0.035), (xc - 0.045, xc + 0.02)):
    sx = station(0.5 * (x0 + x1))
    z0, z1 = surf.at([sx], [6.14])[0][2] - 0.012, surf.at([sx], [7.74])[0][2] + 0.012
    strips += both(flank(rrect(x0, x1, z0, z1, 0.0)))
  shell = link(surf.me.copy(), 'shell', coll='Construction')
  for r in ('well', 'well'):
    shell.data.materials.append(material(r))
  sol = shell.modifiers.new('skin', 'SOLIDIFY')
  sol.thickness, sol.offset, sol.material_offset = SKIN, -1.0, 2
  tmp = [boolean(shell, prisms(c, 'seal', 'cut'), 'DIFFERENCE') for c in (openings, arches, lamps, doors, pillar, bonnet, gate)]
  me = evaluated(shell, 'Body')
  log('skin cut', len(me.polygons), 'faces')
  bpy.data.objects.remove(shell)
  for t in tmp:
    bpy.data.objects.remove(t)
  # of the skin's inner side, only what backs a cut stays
  names = [m.name for m in me.materials]
  bm = bmesh.new()
  bm.from_mesh(me)
  seal, well = names.index('seal') if 'seal' in names else -1, {i for i, n in enumerate(names) if n == 'well'}
  from mathutils.kdtree import KDTree
  edge = {v for f in bm.faces if f.material_index == seal for v in f.verts}
  kd = KDTree(len(edge))
  for v in edge:
    kd.insert(v.co, v.index)
  kd.balance()
  drop = [f for f in bm.faces if f.material_index in well and kd.find(f.calc_center_median())[2] > 0.03]
  drop += [f for f in bm.faces if f.material_index not in well and f.normal.z < -0.6 and f.calc_center_median().z < ST['zb'][7] + 0.16]  # the underside: never seen
  bmesh.ops.delete(bm, geom=drop, context='FACES')
  bm.to_mesh(me)
  bm.free()
  body = link(me, 'Body')
  finish(body, surf)
  log('body', len(me.polygons), 'faces')
  g = link(patch(surf, glass, -0.008, 'glass', 'Glass', grow=0.006), 'Glass')
  finish(g, surf)
  fills = fills + [(handles, -0.009, 'well', 0.005), (flap, -0.004, 'body', 0.005), (strips, -0.004, 'trim', 0), ([stop], -0.004, 'tail', 0)]
  if surround:
    fills.append((surround, -0.003, 'bright', 0))
  lm = link(join('Lamps', [(patch(surf, o, d, role, 'f', grow=gr), (0, 0, 0)) for o, d, role, gr in fills]), 'Lamps')
  finish(lm, surf)
  # what stands proud of the body: the arches' lips, the roof's rails
  proud = [(bars, 0.003, 0.012, D.get('handles', 'body'))] + ([(lips, 0.004, 0.012, 'trim')] if lips else [])
  if D.get('rails'):
    rail = Outline(surf.region(*D['rails'], 0.02), surf=surf)
    proud.append(([rail, rail.mirrored()], 0.022, 0.03, 'trim'))
  parts = []
  for outs, lift, thick, role in proud:
    ob = link(patch(surf, outs, lift, role, 'proud'), 'proud', coll='Construction')
    sol = ob.modifiers.new('thick', 'SOLIDIFY')
    sol.thickness, sol.offset = thick, -1.0
    parts.append((evaluated(ob, 'p'), (0, 0, 0)))
    bpy.data.objects.remove(ob)
  if parts:
    finish(link(join(NAME + '_trim', parts), 'Trim'), surf)
  # the wheel wells: a tub behind each arch, as far as the body goes
  bm = bmesh.new()
  for ax in AXLES:
    for sgn in (1, -1):
      c = circle(ax, RW + 0.01, RA + 0.04, 64)
      a = [bm.verts.new((x, 0.50 * sgn, z)) for x, z in c]
      b = [bm.verts.new((x, 1.2 * sgn, z)) for x, z in c]
      fs = [bm.faces.new(a), bm.faces.new(b[::-1])] + [bm.faces.new((a[k], b[k], b[(k + 1) % len(a)], a[(k + 1) % len(a)])) for k in range(len(a))]
      bmesh.ops.recalc_face_normals(bm, faces=fs)
  tubs = bpy.data.meshes.new('Wells')
  bm.to_mesh(tubs)
  bm.free()
  tubs.materials.append(material('well'))
  tub = link(tubs, 'tubs', coll='Construction')
  tmp = boolean(tub, surf.offset(-0.009, '__cut'), 'INTERSECT')
  wells = link(without(evaluated(tub, 'Wells'), {'__cut'}), 'Wells')
  bpy.data.objects.remove(tub)
  bpy.data.objects.remove(tmp)
  finish(wells, angle=50)
  log('glass, lamps and wells')


def lathe(bm, profile, segs=80, mat=0, closed=False):
  """A profile of (radius, y) turned about the y axis."""
  rings = [[bm.verts.new((r * math.cos(a), y, r * math.sin(a))) for r, y in profile] for a in np.linspace(0, 2 * math.pi, segs, endpoint=False)]
  n = len(profile)
  faces = []
  for k in range(segs):
    a, b = rings[k], rings[(k + 1) % segs]
    for i in range(n if closed else n - 1):
      f = bm.faces.new((a[i], a[(i + 1) % n], b[(i + 1) % n], b[i]))
      f.material_index = mat
      faces.append(f)
  return faces


def build_wheel():
  """One wheel about the y axis, its outer face toward +y: the tyre, the rim with five twin spokes, and its back."""
  ro = RW  # the tyre's radius; it is 0.244 wide
  # the tyre: bead, side wall, shoulder, tread with three grooves, and back down
  rr = D['rim']  # the rim's radius: the tyre's bead sits on it
  kr = rr / 0.262
  top = ro - 0.023
  wall = [(rr, 0.100), (rr + 0.2 * (top - rr), 0.113), (rr + 0.5 * (top - rr), 0.120), (rr + 0.8 * (top - rr), 0.121), (top, 0.116)]
  shoulder = [(ro - 0.018 * (1 - math.sin(a)), 0.092 + 0.022 * math.cos(a)) for a in np.radians(np.linspace(10, 90, 7))]
  tread = []
  for g in (0.05, 0.0, -0.05):
    tread += [(ro, g + 0.007), (ro - 0.007, g + 0.005), (ro - 0.007, g - 0.005), (ro, g - 0.007)]
  half = wall + shoulder
  tyre = half + tread + [(r, -y) for r, y in half[::-1]]
  bm = bmesh.new()
  lathe(bm, tyre, mat=0)
  # the rim: its lip and barrel
  lathe(bm, [(kr * r, y) for r, y in ((0.263, 0.098), (0.259, 0.106), (0.250, 0.107), (0.243, 0.098), (0.238, 0.080), (0.236, 0.02))], mat=1)
  # the back of the wheel, dark, and the hub
  lathe(bm, [(kr * 0.236, 0.02), (0.12, 0.02), (0.11, 0.05), (0.002, 0.05)], mat=2)
  lathe(bm, [(kr * 0.205, 0.036), (kr * 0.20, 0.042), (0.075, 0.042)], mat=3)  # the brake disc
  lathe(bm, [(0.060, 0.058), (0.055, 0.070), (0.040, 0.076), (0.02, 0.078), (0.002, 0.078)], mat=1, segs=48)
  bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
  me = bpy.data.meshes.new('Wheel')
  bm.to_mesh(me)
  bm.free()
  for r in ('tire', 'rim', 'well', 'trim'):
    me.materials.append(material(r))
  # the spokes: a dished disc, with five windows and five slots cut through it
  face = lambda r: 0.060 + 0.030 * max(0.0, (r - 0.07) / 0.17) ** 1.4
  rs = np.linspace(0.05, 0.2385, 14)
  disc = [(kr * r if r > 0.06 else r, face(r)) for r in rs] + [(kr * r if r > 0.06 else r, face(r) - 0.028) for r in rs[::-1]]
  bm = bmesh.new()
  lathe(bm, disc, closed=True)
  bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
  dm = bpy.data.meshes.new('spokes')
  bm.to_mesh(dm)
  bm.free()
  dm.materials.append(material('rim'))
  outs = []
  n, hwo, hwi, slotted = D['spokes']
  step = math.radians(360 / n)
  fr = 0.014 if n <= 6 else 0.007
  for k in range(n):
    th = math.radians(90) + step * k
    pts = []
    for outer, (r, hw_) in ((True, (kr * 0.212, hwo)), (False, (0.088 if n <= 6 else 0.105, hwi))):  # a window between two spokes: its outer arc, then its inner
      half = step / 2 - math.asin(hw_ / r)
      arc = [(r, a) for a in np.linspace(-half, half, 9 if outer else 3)]
      pts += arc if outer else arc[::-1]
    xy = np.array([(r * math.cos(th + step / 2 + a), r * math.sin(th + step / 2 + a)) for r, a in pts])
    w = fillet(xy, [fr] + [0] * 7 + [fr, 0.7 * fr, 0, 0.7 * fr])
    outs.append(Outline(np.c_[w[:, 0], np.zeros(len(w)), w[:, 1]], (0, 1, 0), out=0.2, inn=0.1))
    if not slotted:
      continue
    slot = fillet([(0.105, -0.0065), (kr * 0.205, -0.0085), (kr * 0.205, 0.0085), (0.105, 0.0065)], 0.006, 5)  # the slot down a spoke
    xy = np.c_[slot[:, 0] * math.cos(th) - slot[:, 1] * math.sin(th), slot[:, 0] * math.sin(th) + slot[:, 1] * math.cos(th)]
    outs.append(Outline(np.c_[xy[:, 0], np.zeros(len(xy)), xy[:, 1]], (0, 1, 0), out=0.2, inn=0.1))
  ob = link(dm, 'spokes', coll='Construction')
  tmp = boolean(ob, prisms(outs, 'rim', 'cut_spokes'), 'DIFFERENCE')
  spokes = evaluated(ob, 'Spokes')
  bpy.data.objects.remove(ob)
  bpy.data.objects.remove(tmp)
  return join(NAME + '_wheel', [(me, (0, 0, 0)), (spokes, (0, 0, 0))])


def build_wheels():
  me = build_wheel()
  for name, ax, sgn in (('FL', AXLES[0], 1), ('FR', AXLES[0], -1), ('RL', AXLES[1], 1), ('RR', AXLES[1], -1)):
    ob = link(me, 'Wheel_' + name)
    ob.location = (ax, sgn * (TRACK), RW)
    ob.rotation_euler = (0, 0, 0 if sgn > 0 else math.pi)
    finish(ob, angle=32)
  log('wheels', len(me.polygons), 'faces each')


def blob(name, size, role, level=2, shape=None):
  """A box of this size, rounded by subdivision, its control points moved by shape (a function of a unit cube's)."""
  bm = bmesh.new()
  bmesh.ops.create_cube(bm, size=1.0)
  bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=1, use_grid_fill=True)
  for v in bm.verts:
    p = np.array(v.co) * 2  # -1..1
    if shape:
      p = np.array(shape(p), float)
    v.co = p * 0.5 * np.array(size)
  me = bpy.data.meshes.new(name + '_cage')
  bm.to_mesh(me)
  bm.free()
  ob = link(me, name + '_cage', coll='Construction')
  mod = ob.modifiers.new('sub', 'SUBSURF')
  mod.levels = level
  out = evaluated(ob, name)
  bpy.data.objects.remove(ob)
  out.materials.append(material(role))
  return out


def join(name, parts):
  """Meshes (each with its place) as one mesh."""
  bm = bmesh.new()
  mats = []
  for me, loc in parts:
    for m in me.materials:
      if m not in mats:
        mats.append(m)
    n0 = len(bm.faces)
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    new = bm.faces[n0:]
    for f in new:
      f.material_index = mats.index(me.materials[f.material_index]) if len(me.materials) else 0
    for v in {v for f in new for v in f.verts}:
      v.co += Vector(loc)
  out = bpy.data.meshes.new(name)
  bm.to_mesh(out)
  bm.free()
  for m in mats:
    out.materials.append(m)
  return out


def build_mirrors(surf):
  """A door mirror each side, on the front corner of the side glass: a stalk, the housing, and its glass."""
  foot = surf.at([5.95], [6.22])[0]
  house = blob('house', (0.105, 0.205, 0.118), 'body', shape=lambda p: (p[0] * (1 - 0.18 * (p[1] > 0)), p[1], p[2] * (1 - 0.22 * (p[1] > 0))))
  glass = blob('mglass', (0.012, 0.165, 0.086), 'glass', level=2)
  stalk = blob('stalk', (0.07, 0.10, 0.028), 'trim', level=2)
  c = np.array([foot[0] - 0.03, foot[1] + 0.155, foot[2] + 0.055])
  me = join(NAME + '_mirror', [(house, c), (glass, c + (-0.052, 0.0, 0.0)), (stalk, (foot[0] - 0.02, foot[1] + 0.035, foot[2] + 0.012))])
  for sgn in (1, -1):
    ob = link(me, 'Mirror_' + ('L' if sgn > 0 else 'R'))
    ob.scale = (1, sgn, 1)
    finish(ob, angle=40)
  log('mirrors')


def stage():
  """A camera, a sun and a floor to look at the car with; none of it is the asset."""
  sc = bpy.context.scene
  cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
  cam.data.lens = 70
  cam.location = (-9.5, 8.0, 4.2)
  cam.rotation_euler = (Vector((0, 0, 0.6)) - Vector(cam.location)).to_track_quat('-Z', 'Y').to_euler()
  sun = bpy.data.objects.new('Sun', bpy.data.lights.new('Sun', 'SUN'))
  sun.data.energy, sun.data.angle = 3.0, 0.3
  sun.rotation_euler = (math.radians(48), 0, math.radians(200))
  floor = bpy.data.meshes.new('Floor')
  floor.from_pydata([(-40, -40, 0), (40, -40, 0), (40, 40, 0), (-40, 40, 0)], [], [(0, 1, 2, 3)])
  m = bpy.data.materials.new('floor')
  m.diffuse_color = (0.235, 0.24, 0.255, 1)
  m.use_nodes = True
  m.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.235, 0.24, 0.255, 1)
  m.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.8
  floor.materials.append(m)
  for ob in (cam, sun, bpy.data.objects.new('Floor', floor)):
    collection('Stage').objects.link(ob)
  sc.camera = cam
  sc.render.resolution_x, sc.render.resolution_y = 1600, 900
  if sc.world is None:
    sc.world = bpy.data.worlds.new('World')
  sc.world.use_nodes = True
  bg = sc.world.node_tree.nodes.get('Background')
  if bg:
    bg.inputs[0].default_value, bg.inputs[1].default_value = (0.55, 0.58, 0.63, 1), 1.0


def sculpt(me):
  """The surface's sculpting, beyond what the cage gives: the wings flared round each wheel arch, and a hollow along
  the doors above the sill."""
  co = coords(me)
  vn = np.empty(len(co) * 3, np.float32)
  me.vertex_normals.foreach_get('vector', vn)
  ny = np.abs(vn.reshape(-1, 3)[:, 1])
  x, y, z = co.T
  zb = ST['zb'][7]
  flank = np.clip((np.abs(y) - 0.60) / 0.2, 0, 1) * np.clip(ny * 1.4, 0, 1)
  out = np.zeros(len(co))
  for ax in AXLES:
    d = np.hypot(x - ax, z - (RW + 0.01))
    out += D.get('flare', 0.016) * np.exp(-((d - (RA + 0.04)) / 0.08) ** 2) * np.clip((z - zb - 0.03) / 0.1, 0, 1)
  x0, x1 = AXLES[1] + RA + 0.12, AXLES[0] - RA - 0.12
  along = np.clip((x - x0) / 0.25, 0, 1) * np.clip((x1 - x) / 0.25, 0, 1)
  out -= D.get('hollow', 0.012) * along * np.exp(-((z - (zb + 0.31)) / 0.11) ** 2)
  co[:, 1] += np.sign(y) * out * flank
  me.vertices.foreach_set('co', co.ravel())
  me.update()


def build_fin(surf):
  """The aerial's fin, on the roof's centre line."""
  p = surf.at([D['fin']], [10.0])[0]
  me = blob('fin', (0.17, 0.05, 0.06), 'body', shape=lambda q: (q[0] - 0.45 * (q[2] > 0), q[1] * (1 - 0.65 * (q[2] > 0)), q[2]))
  ob = link(me, 'Fin')
  ob.location = (p[0], 0, p[2] + 0.018)
  finish(ob, angle=40)


def main():
  reset()
  out = {}
  for name in DESIGNS:
    if ONLY and name != ONLY:
      continue
    use(name)
    cage = build_cage()
    for r in ('body', 'trim'):
      cage.materials.append(material(r))
    cage_ob = link(cage, name + '_cage', coll='Construction')
    mod = cage_ob.modifiers.new('sub', 'SUBSURF')
    mod.levels = mod.render_levels = LEVEL
    me = evaluated(cage_ob, 'surface')
    sculpt(me)
    surf = Surface(me)
    co = surf.co
    dims = (co[:, 0].max() - co[:, 0].min(), co[:, 1].max() - co[:, 1].min(), co[:, 2].max())  # length, width without the mirrors, height
    log(name, 'surface', np.round(dims, 3), 'x', co[:, 0].min().round(3), co[:, 0].max().round(3), 'faces', len(me.polygons))
    build_body(surf)
    build_wheels()
    build_mirrors(surf)
    build_fin(surf)
    for ob in list(collection('Construction').objects):
      if not ob.name.endswith('_cage'):
        bpy.data.objects.remove(ob)
    bpy.context.view_layer.update()
    out[name] = {'dims': [float(d) for d in dims]}
    objs = [o for o in collection(name.upper() if name == 'suv' else name.capitalize()).objects if o.type == 'MESH']
    if A.get('export'):
      import os
      os.makedirs(os.path.abspath(A['export']), exist_ok=True)
      out[name]['asset'] = write_asset(os.path.join(A['export'], f'diorama-assets-{name}.bin'), objs, dims)
    for ob in objs + [bpy.data.objects[name + '_cage']]:  # side by side in the file; each is built, and exported, about the origin
      ob.location.y += D['place']
  for me in [m for m in bpy.data.meshes if m.users == 0]:
    bpy.data.meshes.remove(me)
  stage()
  collection('Construction', hide=True)
  bpy.context.view_layer.update()
  if A.get('save'):
    bpy.ops.wm.save_mainfile()
  out['log'] = LOG
  return out


try:
  RESULT = main()
except Exception:
  import traceback
  RESULT = {'error': traceback.format_exc(), 'log': LOG}
