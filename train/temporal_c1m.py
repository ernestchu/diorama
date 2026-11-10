"""Windows of consecutive frames around the labelled frames, for temporal consistency (metric and training).

For every hand-labelled frame ($STOPLINE_C1M/items.json) and every test frame above 40 mph (fast_items.json), take
the frames within WIN frames (1 s at 20 fps) before and after it: every frame for test windows (the metric needs the
live rate), every STEP_TRAIN-th for training windows. Each window frame gets a reference: the labelled line carried
into the frame with the car's motion (prop.carry), then judged by the labelling guideline. A line whose middle row
lies outside LINE_ROWS is too far or too near (no line); within MARGIN px of either limit, where the annotator could
go either way, or carried to under half its labelled distance (where carrying errors grow), it is unknown. A frame
labelled no line gives no line only at itself, and above 40 mph (no stop line at that speed by the labelling rules);
else unknown, since a line can come into range within a second. Where windows overlap, the nearest label wins.

  python temporal_c1m.py [N]  -> $STOPLINE_C1M/temporal/yuv/<sid>_<frame>t.png, temporal/items.json (one entry a
                                 frame), temporal/windows.json ({centre id: {split, kind, frames: [id], ref: [...]}}),
                                 temporal/rgb/<id>.jpg (test frames, 512x256, for checking); with N, only the first
                                 N segments, into temporal_check/
"""
import json, os, sys
from collections import Counter
from multiprocessing import Pool
import numpy as np, cv2

from c1m import Segment
import build_c1m
from build_c1m import render
from cams import FrameCam
from prop import Poses, carry, open_ends
from paths import C1M, META

OUT = f'{C1M}/temporal'
WIN, STEP_TRAIN = 20, 2
LINE_ROWS, MARGIN = (67.5, 205.0), 4.0   # model-frame rows (the guideline's rows 135 and 410 of 512)
MIN_DIST_RATIO = 0.5


def line_ref(P, cam, i, j, lab, kind):
  """The reference for frame j from a label at frame i: ('line', [x1, y1, x2, y2]) | ('none', None) | ('unknown', None)."""
  if not lab:
    return ('none', None) if (j == i or kind == 'fast') else ('unknown', None)
  if j == i:
    return 'line', lab[0]
  c = carry(P, i, j, lab[0], open_ends(lab[0]))
  if c is None:
    return 'none', None  # behind the camera or out of view: passed, or not yet in the picture
  mid = (c[1] + c[3]) / 2
  lo, hi = LINE_ROWS
  if mid < lo - MARGIN or mid > hi + MARGIN:
    return 'none', None
  if mid < lo + MARGIN or mid > hi - MARGIN:
    return 'unknown', None
  if cam.line_dist(c) < MIN_DIST_RATIO * cam.line_dist(lab[0]):
    return 'unknown', None
  return 'line', [round(v, 2) for v in c]


def main():
  global OUT
  if len(sys.argv) > 1: OUT = OUT + '_check'
  for d in ('yuv', 'rgb'):
    os.makedirs(f'{OUT}/{d}', exist_ok=True)
  build_c1m.OUT = OUT
  meta = {}
  for f in sorted(os.listdir(META)):
    if f.startswith('batch') and f.endswith('.json'):
      meta.update({m['id']: m for m in json.load(open(f'{META}/{f}'))})
  items = json.load(open(f'{C1M}/items.json'))
  fast = json.load(open(f'{C1M}/fast_items.json'))
  centres = {}
  for k, v in items.items():
    centres[k] = {'sid': v['sid'], 'frame': meta[k]['frame'], 'split': v['split'], 'kind': v['kind'], 'lines': v['lines']}
  for k, v in fast.items():
    if v['split'] == 'test':
      centres[k] = {'sid': v['sid'], 'frame': int(k.split('_')[1][:4]), 'split': 'test', 'kind': 'fast', 'lines': []}
  if len(sys.argv) > 1:  # a quick check on the first n segments, into a separate directory
    keep = sorted({c['sid'] for c in centres.values()})[:int(sys.argv[1])]
    centres = {k: c for k, c in centres.items() if c['sid'] in keep}
  jobs = {}
  for k, c in centres.items():
    step = 1 if c['split'] == 'test' else STEP_TRAIN
    c['frames'] = [j for j in range(c['frame'] - WIN, c['frame'] + WIN + 1, step) if 0 <= j < 1200]
    for j in c['frames']:
      jobs.setdefault(c['sid'], {})[j] = f"{c['sid']}_{j:04d}t"
  print(len(centres), 'windows,', sum(len(w) for w in jobs.values()), 'frames in', len(jobs), 'segments', flush=True)

  cam = FrameCam()
  frames = {}  # id -> {sid, frame, path}
  test_sids = {c['sid'] for c in centres.values() if c['split'] == 'test'}
  done = 0
  with Pool(20) as pool:
    for out in pool.imap_unordered(render, list(jobs.items())):
      for fid, o in out.items():
        sid, j = fid[:-6], int(fid[-5:-1])
        frames[fid] = {'sid': sid, 'frame': j, 'path': np.array(o['path'])[::4].round(2).tolist()}
        if sid in test_sids:
          cv2.imwrite(f'{OUT}/rgb/{fid}.jpg', cv2.resize(o['rgb'], (512, 256), interpolation=cv2.INTER_AREA),
                      [cv2.IMWRITE_JPEG_QUALITY, 88])
      done += len(out)
      if done % 2000 < len(out): print(done, flush=True)

  # lines: carried from each window's label; per frame, the nearest label wins
  windows, best, poses = {}, {}, {}
  for k, c in centres.items():
    P = poses.get(c['sid']) or poses.setdefault(c['sid'], Poses(Segment(c['sid']), cam))
    ref = []
    for j in c['frames']:
      fid = f"{c['sid']}_{j:04d}t"
      if fid not in frames: continue
      r = line_ref(P, cam, c['frame'], j, c['lines'], c['kind'])
      ref.append({'id': fid, 'line': r[0], 'xy': r[1]})
      d = abs(j - c['frame'])
      if fid not in best or d < best[fid][0]: best[fid] = (d, r, c['split'], k)
    windows[k] = {'split': c['split'], 'kind': c['kind'], 'sid': c['sid'], 'frame': c['frame'],
                  'frames': [r['id'] for r in ref], 'ref': ref}
  titems = {}
  for fid, f in frames.items():
    d, r, split, k = best[fid]
    titems[fid] = {'split': split, 'sid': f['sid'], 'kind': 'temporal', 'centre': k, 'offset': f['frame'] - centres[k]['frame'],
                   'lines': [r[1]] if r[0] == 'line' else [], 'line_unknown': r[0] == 'unknown',
                   'status': 'lines' if r[0] == 'line' else 'none', 'dir': f'{OUT}/yuv', 'device': None,
                   'dist_to_stop': None, 'path': f['path']}
  json.dump(titems, open(f'{OUT}/items.json', 'w'))
  json.dump(windows, open(f'{OUT}/windows.json', 'w'))
  print(len(titems), 'frames', Counter((v['split'], 'unknown' if v['line_unknown'] else 'line' if v['lines'] else 'none')
                                     for v in titems.values()))


if __name__ == '__main__':
  main()
