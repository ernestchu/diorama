"""Frames with no stop line: driving above FAST (40 mph). A car rarely comes up to a stop line that fast, and one it
passes at speed is not what the stop-line view is for, so these are labelled no line without a look (the odd one
that does hold a line is noise). One frame from each of N random comma1M segments that have both cameras' video
(not comma twos), at a random moment above FAST; the lead is chosen as build_c1m.py does. Both splits, by segment.

  python fast_c1m.py [N]      -> $STOPLINE_C1M/fast/yuv/<id>.png, fast_items.json, fast_check.jpg
"""
import json, os, random, sys
from multiprocessing import Pool
import numpy as np, cv2

from c1m import Segment
import build_c1m
from build_c1m import split_of, choose_lead_near
from cams import FrameCam
from paths import C1M, COMMA1M as DATA, META

OUT = f'{C1M}/fast'
FAST = 40 * 0.44704  # [m/s]
SEED = 0


def pick(sid):
  """A random frame above FAST in the segment, or None."""
  v = Segment(sid).speed
  idx = np.nonzero(v > FAST)[0]
  return (sid, int(random.Random(sid).choice(idx)), round(float(v[idx].max()), 1)) if len(idx) else None


if __name__ == '__main__':
  n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
  os.makedirs(f'{OUT}/yuv', exist_ok=True)
  stops = json.load(open(f'{META}/stops.json'))
  sids = sorted(s for s, r in stops.items() if r['device'] != 1 and os.path.exists(f'{DATA}/{s}/ecamera.hevc'))
  random.Random(SEED).shuffle(sids)
  build_c1m.OUT = OUT
  pool = Pool(16)
  picks = []
  for p in pool.imap(pick, sids, chunksize=8):
    if p: picks.append(p)
    if len(picks) == n: break
  from teacher import Teacher
  teacher = Teacher()
  cam = FrameCam()
  items, sheet = {}, []
  with pool:
    for out in pool.imap_unordered(build_c1m.render, [(sid, {f: f'{sid}_{f:04d}f'}) for sid, f, _ in picks]):
      fids = list(out)
      boxes = teacher([cv2.cvtColor(out[f]['rgb'], cv2.COLOR_BGR2RGB) for f in fids])
      for fid, bx in zip(fids, boxes):
        sid = fid.split('_')[0]
        lead, _, amb = choose_lead_near(bx, np.array(out[fid]['path']), cam)
        items[fid] = {'split': split_of(sid), 'sid': sid, 'kind': 'fast', 'device': stops[sid]['device'], 'status': 'none',
                      'lines': [], 'lead': lead, 'ambiguous': amb, 'dist_to_stop': None,
                      'vehicles': [[round(v / 2, 1) for v in b[:4]] + [round(b[4], 3)] for b in bx], 'dir': f'{OUT}/yuv'}
        if len(sheet) < 24 and hash(fid) % 20 == 0:
          sheet.append((fid, out[fid]['rgb'], lead))
  json.dump(items, open(f'{C1M}/fast_items.json', 'w'))
  tiles = []
  for fid, rgb, lead in sheet:
    im = rgb.copy()
    if lead:
      cv2.rectangle(im, (int(2 * lead[0]), int(2 * lead[1])), (int(2 * lead[2]), int(2 * lead[3])), (0, 120, 255), 2)
    cv2.putText(im, fid[:8] + fid[-6:], (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    tiles.append(cv2.resize(im, (512, 256)))
  while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
  cv2.imwrite(f'{C1M}/fast_check.jpg', np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]),
              [cv2.IMWRITE_JPEG_QUALITY, 85])
  print(len(items), 'fast frames', {s: sum(v['split'] == s for v in items.values()) for s in ('train', 'test')},
        '| with a lead', sum(v['lead'] is not None for v in items.values()))
