"""Frames with no stop line: just after the car has passed a labelled one, in the intersection.

There the model sees paint across the junction (the far side's stop bars and crosswalks, which face the other way)
and takes it for our line. For each labelled stop (a segment whose frame has a line), the lines are lifted
onto the road from that frame (prop.py) and followed through the later frames; a frame counts once every line is
PAST_MIN..PAST_MAX m behind the camera. One frame is taken near each of AT m past. A stop is skipped if the car stops
again within AHEAD m of travel after passing (another line ahead may be in view). Both splits: the test set's
passed frames measure the false lines there.

  python pass_c1m.py      -> $STOPLINE_C1M/pass/yuv/<id>.png, pass_items.json, pass_check.jpg
"""
import collections, json, os
from multiprocessing import Pool
import numpy as np, cv2

from c1m import Segment
import build_c1m
from build_c1m import split_of, choose_lead
from cams import FrameCam
from prop import Poses
from paths import C1M, LABELS, META

OUT = f'{C1M}/pass'
PAST_MIN, PAST_MAX = 1.0, 18.0  # [m] behind the camera
AT = (2.0, 6.0, 10.0, 15.0)     # [m] past the line, one frame each
AHEAD, STOPPED = 40.0, 0.5      # [m] of travel, [m/s]


def plan(sid, cam, frame, lines):
  """{frame: metres past} after the car has passed the lines labelled in `frame`, or {} to skip."""
  seg = Segment(sid)
  P = Poses(seg, cam)
  g = [P.lift(frame, [l[:2], l[2:]]) for l in lines]
  if any(np.isnan(x).any() for x in g):
    return {}
  g = np.concatenate(g)
  past = np.array([-P.project(j, g)[1].max() for j in range(len(P.pos))])  # the nearest line end's distance behind
  travel = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P.pos, axis=0), axis=1))])
  speed = seg.speed
  ok = [j for j in range(frame + 1, len(past)) if PAST_MIN <= past[j] <= PAST_MAX]
  if not ok:
    return {}
  j0 = ok[0]
  if any(speed[j] < STOPPED and travel[j] - travel[j0] < AHEAD for j in range(j0, len(speed))):
    return {}
  out = {}
  for at in AT:
    j = min(ok, key=lambda j: abs(past[j] - at))
    if abs(past[j] - at) < 1.5:
      out[j] = round(float(past[j]), 1)
  return out


if __name__ == '__main__':
  os.makedirs(f'{OUT}/yuv', exist_ok=True)
  store = json.load(open(f'{LABELS}/store.json'))
  meta = {}
  for f in sorted(os.listdir(META)):
    if f.startswith('batch') and f.endswith('.json'):
      meta.update({m['id']: m for m in json.load(open(f'{META}/{f}'))})
  # per segment, the labelled frame with lines whose nearest line is closest (least lift error)
  best = {}
  for k, v in store.items():
    m = meta[k]
    if v['status'] != 'lines' or not v['lines']: continue
    key = m['sid']
    cam = FrameCam()
    d = min(cam.line_dist(l) for l in v['lines'])
    if key not in best or d < best[key][0]:
      best[key] = (d, m['frame'], v['lines'], m)
  labelled = collections.defaultdict(set)
  for k, v in store.items():
    labelled[meta[k]['sid']].add(meta[k]['frame'])
  plans = {}
  for key, (_, frame, lines, m) in sorted(best.items()):
    p = {j: d for j, d in plan(key, FrameCam(), frame, lines).items() if j not in labelled[key]}
    if p: plans[key] = (p, m)
  print(f'{len(best)} labelled stops; {len(plans)} passed; {sum(len(p) for p, _ in plans.values())} frames', flush=True)

  build_c1m.OUT = OUT
  pool = Pool(16)
  from teacher import Teacher
  teacher = Teacher()
  items, sheet, names, jobs = {}, [], {}, []
  with pool:
    for sid, (p, m) in plans.items():
      want = {j: f'{sid}_{j:04d}x' for j in p}
      names.update({fid: (sid, j) for j, fid in want.items()})
      jobs.append((sid, want))
    for out in pool.imap_unordered(build_c1m.render, jobs):
      fids = list(out)
      boxes = teacher([cv2.cvtColor(out[f]['rgb'], cv2.COLOR_BGR2RGB) for f in fids])
      for fid, bx in zip(fids, boxes):
        key, j = names[fid]
        p, m = plans[key]
        cam = FrameCam()
        lead, _ = choose_lead(bx, np.array(out[fid]['path']), cam)
        items[fid] = {'split': split_of(key), 'sid': key, 'kind': 'passed', 'past': p[j], 'device': m['device'],
                      'status': 'none', 'lines': [], 'lead': lead, 'dist_to_stop': None,
                      'vehicles': [[round(v / 2, 1) for v in b[:4]] + [round(b[4], 3)] for b in bx], 'dir': f'{OUT}/yuv'}
        if len(sheet) < 24 and hash(fid) % 9 == 0:
          sheet.append((fid, out[fid]['rgb'], p[j], lead))
  json.dump(items, open(f'{C1M}/pass_items.json', 'w'))
  tiles = []
  for fid, rgb, past, lead in sheet:
    im = rgb.copy()
    if lead:
      cv2.rectangle(im, (int(2 * lead[0]), int(2 * lead[1])), (int(2 * lead[2]), int(2 * lead[3])), (0, 120, 255), 2)
    cv2.putText(im, f'{fid[:8]}{fid[-6:]} +{past} m', (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    tiles.append(cv2.resize(im, (512, 256)))
  while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
  cv2.imwrite(f'{C1M}/pass_check.jpg', np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]),
              [cv2.IMWRITE_JPEG_QUALITY, 85])
  n = collections.Counter(v['split'] for v in items.values())
  print(len(items), 'passed frames', dict(n), '| with a lead', sum(v['lead'] is not None for v in items.values()))
