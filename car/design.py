"""
Diorama's own car: an original design, built from nothing but the numbers in this file.

Run inside Blender (5.x), from the repo root:

    /Applications/Blender.app/Contents/MacOS/Blender -b car/diorama_car_design.blend -P car/design.py -- \
        [--level 4] [--render DIR] [--export OUT.bin] [--save]

It builds the car in the open .blend (x forward, y left, z up, metres, origin on the ground under the centre),
optionally renders previews, saves the .blend and writes the asset the app loads (see car/assets.py for the format).
"""
import math
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector

ARGS = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []


def arg(name, default=None):
  return ARGS[ARGS.index(name) + 1] if name in ARGS else default


LEVEL = int(arg('--level', 4))

# colours by role, the ones the app draws (car/assets.py)
BODY, GLASS, TRIM, TIRE, RIM = (0.62, 0.62, 0.64), (0.33, 0.33, 0.35), (0.30, 0.30, 0.32), (0.22, 0.22, 0.23), (0.40, 0.40, 0.42)
TAIL, HEAD = (0.80, 0.10, 0.10), (0.88, 0.89, 0.91)
ROLES = {  # role: (colour, emission)
  'body': (BODY, -1.0), 'glass': (GLASS, 0.0), 'trim': (TRIM, 0.0), 'tire': (TIRE, 0.0), 'rim': (RIM, 0.0),
  'tail': (TAIL, 1.0), 'tail_dark': (tuple(c * 0.42 for c in TAIL), 0.5), 'housing': (tuple(c * 0.38 for c in TAIL), 0.6),
  'reflector': (tuple(c * 0.8 for c in TAIL), 0.0), 'head': (HEAD, 0.5),
}

# ---------------------------------------------------------------- the body's cage
# One ring of control points per station, nose to tail. Each ring runs from the underside's centre line, round the
# side, to the roof's centre line (11 lines, L0-L10); the other side is its mirror.
X    = [2.37, 2.32, 2.08, 1.78, 1.44, 1.15, 0.72, 0.28, -0.30, -0.95, -1.46, -1.86, -2.14, -2.32, -2.37]
W    = [0.60, 0.80, 0.915, 0.95, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.945, 0.91, 0.83, 0.64]  # half width
ZB   = [0.34, 0.26, 0.21, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.21, 0.24, 0.30, 0.36, 0.46]  # underside
ZSH  = [0.61, 0.65, 0.72, 0.77, 0.80, 0.82, 0.83, 0.84, 0.85, 0.87, 0.89, 0.91, 0.91, 0.89, 0.85]  # shoulder line
ZBELT = [0.65, 0.70, 0.80, 0.88, 0.94, 0.99, 1.00, 1.01, 1.02, 1.04, 1.07, 1.08, 1.07, 1.02, 0.92]  # belt / wing top
ZTOP = [0.67, 0.73, 0.85, 0.93, 0.99, 1.04, 1.31, 1.565, 1.60, 1.54, 1.39, 1.23, 1.13, 1.09, 0.97]  # centre line
WR   = [0.30, 0.40, 0.46, 0.48, 0.50, 0.66, 0.74, 0.61, 0.62, 0.60, 0.58, 0.60, 0.64, 0.56, 0.40]  # roof edge half width
DZR  = [0.01, 0.01, 0.012, 0.012, 0.012, 0.02, 0.10, 0.055, 0.05, 0.05, 0.05, 0.04, 0.02, 0.01, 0.01]  # its drop
ZMID = 0.50
NL = 11


def ring(i):
  w, zb, zt = W[i], ZB[i], ZTOP[i]
  wr, zr = WR[i], zt - DZR[i]
  l6 = (w - 0.07, ZBELT[i])
  l8 = (wr, zr)
  l7 = (0.5 * (l6[0] + l8[0]) + 0.015, 0.5 * (l6[1] + l8[1]) + 0.004)
  return [
    (0.0, zb), (0.5 * (w - 0.07), zb), (w - 0.07, zb + 0.005), (w - 0.02, zb + 0.12), (w, min(ZMID, 0.5 * (zb + ZSH[i]) + 0.05)),
    (w - 0.012, ZSH[i]), l6, l7, l8, (0.5 * wr, zt - 0.2 * DZR[i]), (0.0, zt),
  ]


def build_cage():
  bm = bmesh.new()
  n = len(X)
  R, L = [], []  # +y and -y vertices per station
  for i in range(n):
    pts = ring(i)
    r = [bm.verts.new((X[i], y, z)) for y, z in pts]
    l = [r[0]] + [bm.verts.new((X[i], -y, z)) for y, z in pts[1:-1]] + [r[-1]]
    R.append(r)
    L.append(l)
  for i in range(n - 1):
    for j in range(NL - 1):
      bm.faces.new((R[i][j], R[i + 1][j], R[i + 1][j + 1], R[i][j + 1]))
      bm.faces.new((L[i][j], L[i][j + 1], L[i + 1][j + 1], L[i + 1][j]))
  for i, flip in ((0, False), (n - 1, True)):  # the caps
    for j in range(NL - 1):
      if j == 0:
        f = [R[i][0], R[i][1], L[i][1]]
      elif j == NL - 2:
        f = [R[i][j], R[i][j + 1], L[i][j]]
      else:
        f = [R[i][j], R[i][j + 1], L[i][j + 1], L[i][j]]
      bm.faces.new(f[::-1] if flip else f)
  bm.normal_update()
  bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
  cr = bm.edges.layers.float.new('crease_edge')

  def crease(line, value, stations=range(n - 1)):
    for i in stations:
      for side in (R, L):
        e = bm.edges.get((side[i][line], side[i + 1][line]))
        if e:
          e[cr] = value

  crease(5, 0.45)  # the shoulder
  crease(2, 0.5)  # the sill's lower edge
  crease(8, 0.25, range(5, 12))  # the roof's edge
  for i, v in ((0, 0.35), (n - 1, 0.5)):  # the caps' rims
    for side in (R, L):
      for j in range(NL - 1):
        e = bm.edges.get((side[i][j], side[i][j + 1]))
        if e:
          e[cr] = v
  for side in (R, L):  # the tail's upper edge, a ducktail
    for j in range(6, NL - 1):
      e = bm.edges.get((side[n - 2][j], side[n - 2][j + 1]))
      if e:
        e[cr] = 0.7
  me = bpy.data.meshes.new('cage')
  bm.to_mesh(me)
  bm.free()
  return me


# ---------------------------------------------------------------- helpers
def material(role):
  m = bpy.data.materials.get(role)
  if m is None:
    m = bpy.data.materials.new(role)
    col, emis = ROLES[role]
    m.diffuse_color = (*col, 1)
    m.use_nodes = True
    b = m.node_tree.nodes.get('Principled BSDF')
    b.inputs['Base Color'].default_value = (*col, 1)
    b.inputs['Roughness'].default_value = 0.45 if role in ('body', 'glass') else 0.7
    if emis > 0:
      b.inputs['Emission Color'].default_value = (*col, 1)
      b.inputs['Emission Strength'].default_value = emis
  return m


def link(me, name, role=None, smooth=True):
  ob = bpy.data.objects.new(name, me)
  bpy.context.scene.collection.objects.link(ob)
  if role:
    me.materials.append(material(role))
  if smooth:
    me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
  return ob


def subdivided(me, level, name):
  ob = link(me, name + '_cage')
  mod = ob.modifiers.new('sub', 'SUBSURF')
  mod.levels = mod.render_levels = level
  dg = bpy.context.evaluated_depsgraph_get()
  out = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
  out.name = name
  bpy.data.objects.remove(ob)
  return out


def reset():
  for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob)
  for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
    for b in list(coll):
      coll.remove(b)


# ---------------------------------------------------------------- previews
def render(folder):
  sc = bpy.context.scene
  sc.render.engine = 'CYCLES'
  sc.cycles.samples = 24
  sc.cycles.use_denoising = True
  sc.render.resolution_x, sc.render.resolution_y = 1280, 720
  world = bpy.data.worlds.new('w') if not sc.world else sc.world
  sc.world = world
  world.use_nodes = True
  world.node_tree.nodes['Background'].inputs[0].default_value = (0.75, 0.78, 0.82, 1)
  world.node_tree.nodes['Background'].inputs[1].default_value = 0.9
  sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN'))
  sun.data.energy = 3.0
  sun.data.angle = 0.2
  sun.rotation_euler = (math.radians(50), 0, math.radians(120))
  sc.collection.objects.link(sun)
  ground = bpy.data.meshes.new('ground')
  ground.from_pydata([(-30, -30, 0), (30, -30, 0), (30, 30, 0), (-30, 30, 0)], [], [(0, 1, 2, 3)])
  g = link(ground, 'ground', smooth=False)
  gm = bpy.data.materials.new('ground')
  gm.diffuse_color = (0.5, 0.5, 0.52, 1)
  ground.materials.append(gm)
  cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
  cam.data.lens = 70
  sc.collection.objects.link(cam)
  sc.camera = cam
  views = {'front34': (7.5, 5.5, 2.3), 'rear34': (-7.5, 5.0, 2.6), 'side': (0, 11, 1.0), 'rear': (-10, 0.01, 1.3),
           'front': (10, 0.01, 1.2), 'top': (0.01, 0, 14), 'chase': (-8.5, 0, 3.6), 'low34': (6.0, -6.5, 0.9)}
  only = arg('--views')
  for name, pos in views.items():
    if only and name not in only.split(','):
      continue
    cam.location = pos
    d = Vector((0, 0, 0.7)) - Vector(pos)
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = f'{folder}/{name}.png'
    bpy.ops.render.render(write_still=True)
  for ob in (sun, g, cam):
    bpy.data.objects.remove(ob)


# ---------------------------------------------------------------- main
def main():
  reset()
  body = subdivided(build_cage(), LEVEL, 'body')
  ob = link(body, 'body', 'body')
  co = np.array([v.co[:] for v in body.vertices])
  print('dims', co.max(0) - co.min(0), 'min', co.min(0), 'max', co.max(0), 'faces', len(body.polygons))
  if arg('--render'):
    render(arg('--render'))
  if '--save' in ARGS:
    bpy.ops.wm.save_mainfile()


main()
