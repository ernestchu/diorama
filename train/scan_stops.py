"""Find stop events in every comma1M segment: the car slows from >6 m/s to a standstill.
For each, the frame where it stops and the distance driven to that point from every earlier frame."""
import json, sys, numpy as np
from multiprocessing import Pool
from c1m import Segment, segments

DT = 0.05

def scan(sid):
  try:
    s = Segment(sid)
    v = s.speed
  except Exception as e:
    return sid, None
  ev = []
  i = 0
  while i < len(v):
    if v[i] < 0.3 and i >= 60:
      back = v[max(0, i - 200):i]
      if back.max() > 6 and v[i - 60] > 2:
        dist = np.cumsum(v[:i][::-1]) [::-1] * DT  # metres from frame j to the stop
        ev.append({'stop': int(i), 'dist': [round(float(x), 2) for x in dist]})
        while i < len(v) and v[i] < 1.0:
          i += 1
        continue
    i += 1
  return sid, {'w': s.w, 'device': s.device, 'events': ev, 'mean_speed': float(v.mean())}

if __name__ == '__main__':
  with Pool(32) as p:
    res = dict(p.map(scan, segments(), chunksize=8))
  json.dump(res, open(sys.argv[1], 'w'))
  n = sum(len(r['events']) for r in res.values() if r)
  print('segments', len(res), 'bad', sum(r is None for r in res.values()), 'stop events', n,
        'segments with a stop', sum(1 for r in res.values() if r and r['events']))
