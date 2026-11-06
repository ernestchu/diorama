"""Sample frames from comma1M stop events (one near, one far from where the car stopped) plus random
driving frames, rebuilt into openpilot's medmodel frame at 2x (1024x512) for labelling. Only drives from devices the
driving model can run on (with both cameras' video; not comma twos).
Writes <site>/frames/<id>.jpg and a private record of where each came from."""
import json, os, random, sys
import numpy as np, cv2
from multiprocessing import Pool
from c1m import Segment

stops = json.load(open(sys.argv[1])); site = sys.argv[2]; meta_out = sys.argv[3]
n_events, n_random, seed = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
exclude = set(json.load(open(sys.argv[7]))) if len(sys.argv) > 7 else set()  # sids already sampled
rng = random.Random(seed)
os.makedirs(f'{site}/frames', exist_ok=True)
from paths import COMMA1M as data
usable = lambda sid: os.path.exists(f'{data}/{sid}/ecamera.hevc')
WINDOWS = (('near', 7, 12), ('far', 18, 35))  # distances to the stop [m]

queued = json.load(open(os.path.join(os.path.dirname(sys.argv[1]), 'queued.json')))
events = [(sid, e) for sid, r in stops.items() if sid not in exclude and sid in queued and usable(sid) for e in r['events'][:1]]
rng.shuffle(events)
first = [x for x in events if queued[x[0]] is None]  # first in line at the stop, the rest behind other cars
behind = [x for x in events if queued[x[0]] is not None]
k = min(len(first), round(n_events * 0.87))
picks = {}
for sid, e in first[:k] + behind[:n_events - k]:
  d = np.array(e['dist'])
  for kind, lo, hi in WINDOWS:
    idx = np.nonzero((d >= lo) & (d <= hi))[0]
    if len(idx):
      i = int(rng.choice(idx))
      picks.setdefault(sid, []).append((i, kind, round(float(d[i]), 1)))
moving = [sid for sid, r in stops.items() if not r['events'] and r['mean_speed'] > 3 and sid not in exclude and usable(sid)]
rng.shuffle(moving)
for sid in moving[:n_random]:
  picks.setdefault(sid, []).append((rng.randrange(0, 1200), 'random', None))


def render(item):
  sid, frames = item
  s = Segment(sid)
  imgs = s.frames([f for f, *_ in frames])
  out = []
  for f, kind, dist in frames:
    if f not in imgs: continue
    fid = f'{sid}_{f:04d}'
    cv2.imwrite(f'{site}/frames/{fid}.jpg', cv2.cvtColor(s.warp(imgs[f], 2.0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    out.append({'id': fid, 'sid': sid, 'frame': f, 'kind': kind, 'dist_to_stop': dist, 'width': s.w,
                'device': s.device, 'rpy': [float(x) for x in s.rpy]})
  return out


with Pool(24) as p:
  meta = [m for ms in p.map(render, list(picks.items())) for m in ms]
json.dump(meta, open(meta_out, 'w'), indent=1)
print(len(meta), 'frames;', {k: sum(m['kind'] == k for m in meta) for k in ('near', 'far', 'random')})
