"""Mark each stop event first-in-line or queued: is there a vehicle in our lane within 20 m when we stop?
Uses the medmodel frame's flat-road geometry (camera ~1.25 m up)."""
import json, sys, numpy as np, cv2
from multiprocessing import Pool
from c1m import Segment
from vehicles import Vehicles

HGT, FL, CY = 1.25, 2 * 910.0, 2 * 47.6  # at 2x

def stop_frame(item):
  sid, stop = item
  s = Segment(sid)
  f = s.frames([stop]).get(stop)
  return sid, (None if f is None else s.warp(f, 2.0))

def queued(boxes):
  for x1, y1, x2, y2, sc in boxes:
    if y2 <= CY + 5: continue
    d = HGT * FL / (y2 - CY)
    lat = ((x1 + x2) / 2 - 512) * d / FL
    if d < 20 and abs(lat) < 1.6: return round(d, 1)
  return None

if __name__ == '__main__':
  stops = json.load(open(sys.argv[1]))
  items = [(sid, r['events'][0]['stop']) for sid, r in stops.items() if r['events']]
  t = Vehicles(); out = {}
  with Pool(24) as p:
    batch = []
    for sid, img in p.imap_unordered(stop_frame, items, chunksize=4):
      if img is None: continue
      batch.append((sid, img))
      if len(batch) == 16:
        for (s, _), b in zip(batch, t([i for _, i in batch])): out[s] = queued(b)
        batch = []
    if batch:
      for (s, _), b in zip(batch, t([i for _, i in batch])): out[s] = queued(b)
  json.dump(out, open(sys.argv[2], 'w'))
  print(len(out), 'events;', sum(v is None for v in out.values()), 'first in line')
