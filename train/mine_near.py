"""Frames with near leads: slow driving and standstills in comma1M, where the car ahead can fill the view.

The labelled frames come from approaches 7-35 m before a stop, so they hold almost no lead closer than a few metres,
and a model trained on them alone misses most of those. For every usable segment (both road cameras, no comma two),
this takes frames below V_MAX, one every STEP frames, at most PER_SEG a segment; renders them as build_c1m.py does;
runs the teacher; and chooses the lead with choose_lead_near. Their stop lines are not labelled ('line_unknown': the
line loss skips them), so they train the lead head only.

near/items.json keeps every mined frame (eval_near.py tests on its test split); near_items.json, what
train_c1m.py --near adds, takes from the training split every lead cut by the frame's bottom and a fixed number of
the rest (TAKE), so the slow frames do not swamp the labelled ones.

  python mine_near.py   -> $STOPLINE_C1M/near/{yuv,rgb}/<id>, near/items.json, near_items.json
"""
import json, os, random
from multiprocessing import Pool
import numpy as np, cv2

from c1m import Segment, segments
from cams import FrameCam
import build_c1m
from build_c1m import split_of, choose_lead_near, render, NEAR_ROW
from paths import C1M

OUT = f'{C1M}/near'
V_MAX, STEP, PER_SEG = 5.0, 30, 8
TAKE = {'tall': 1000, 'mid': 500, 'small': 500, 'none': 1000}  # cut leads: all of them


def size(lead):
  """A lead's size class: cut (by the frame's bottom), tall (>= 120 px), mid (>= 60 px), small, or none."""
  if lead is None:
    return 'none'
  h = lead[3] - lead[1]
  return 'cut' if lead[3] >= NEAR_ROW else 'tall' if h >= 120 else 'mid' if h >= 60 else 'small'


def plan(sid):
  try:
    s = Segment(sid)
  except Exception:
    return None
  if s.w == 1164 or s.wide_video is None:
    return None
  v = s.speed
  pick, last = [], -STEP
  for i in np.flatnonzero(v < V_MAX):
    if i - last >= STEP:
      pick.append(int(i)); last = i
  if not pick:
    return None
  if len(pick) > PER_SEG:
    pick = [pick[k] for k in np.linspace(0, len(pick) - 1, PER_SEG).round().astype(int)]
  return sid, {i: f'{sid}_{i:04d}' for i in pick}, {i: float(v[i]) for i in pick}


if __name__ == '__main__':
  for d in ('yuv', 'rgb'):
    os.makedirs(f'{OUT}/{d}', exist_ok=True)
  build_c1m.OUT = OUT  # render() writes the YUV png to OUT/yuv
  with Pool(32) as p:
    plans = [x for x in p.map(plan, segments(), chunksize=8) if x]
  speed = {sid: sp for sid, _, sp in plans}
  print(len(plans), 'segments', sum(len(w) for _, w, _ in plans), 'frames', flush=True)
  pool = Pool(24)  # fork the decoders before CUDA starts in this process
  from teacher import Teacher
  teacher = Teacher()
  cam = FrameCam()
  items = {}
  with pool:
    for n, out in enumerate(pool.imap_unordered(render, [(sid, w) for sid, w, _ in plans])):
      fids = list(out)
      if not fids:
        continue
      boxes = teacher([cv2.cvtColor(out[f]['rgb'], cv2.COLOR_BGR2RGB) for f in fids])
      for fid, bx in zip(fids, boxes):
        sid, fr = fid.rsplit('_', 1)
        lead, info, amb = choose_lead_near(bx, np.array(out[fid]['path']).reshape(-1, 2), cam)
        cv2.imwrite(f'{OUT}/rgb/{fid}.jpg', out[fid]['rgb'], [cv2.IMWRITE_JPEG_QUALITY, 90])
        items[fid] = {'sid': sid, 'frame': int(fr), 'split': split_of(sid), 'kind': 'near_lead', 'dir': f'{OUT}/yuv',
                      'speed': round(speed[sid][int(fr)], 2), 'lines': [], 'line_unknown': True, 'lead': lead,
                      'ambiguous': amb, 'vehicles': [[round(c / 2, 1) for c in b[:4]] + [round(b[4], 3)] for b in bx],
                      'ground': info}
      if n % 100 == 0:
        print(n, len(items), flush=True)
  json.dump(items, open(f'{OUT}/items.json', 'w'))
  print(len(items), 'frames,', sum(v['lead'] is not None for v in items.values()), 'with a lead,',
        sum(v['ambiguous'] for v in items.values()), 'ambiguous')

  train = sorted(k for k, v in items.items() if v['split'] == 'train' and not v['ambiguous'])
  random.seed(0)
  take = {}
  for c in ('cut', 'tall', 'mid', 'small', 'none'):
    ks = [k for k in train if size(items[k]['lead']) == c]
    if c in TAKE and len(ks) > TAKE[c]:
      ks = random.sample(ks, TAKE[c])
    take.update({k: items[k] for k in ks})
    print(c, len(ks))
  json.dump(take, open(f'{C1M}/near_items.json', 'w'))
  print(len(take), 'training frames -> near_items.json')
