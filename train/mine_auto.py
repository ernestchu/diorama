"""Frames worth labelling automatically: approaches to every stop in the training split's segments.

For each stop event in meta/stops.json (devices with both cameras, training split only, so the test segments stay
unseen), take the frames where the car is DISTS metres from where it stopped. A stop line, if there is one, then lies
a few metres further on, inside the range the hand labels cover (about 7-45 m, rows 135 and below in the 2x frame).
Frames within GAP frames of a sampled frame (meta/batch*.json) are skipped. Each frame is rendered as build_c1m.py
does: the 2x YUV png the model trains on, and a 1024x512 RGB jpg for the vision-language labeller (vlm_label.py).

  python mine_auto.py   -> $STOPLINE_C1M/auto/{yuv,rgb}/<id>, auto/items.json
"""
import glob, json, os
from multiprocessing import Pool
import numpy as np, cv2

import build_c1m
from build_c1m import split_of, render
from paths import C1M, META

OUT = f'{C1M}/auto'
DISTS = (9, 14, 20, 28, 38)  # metres before the stop
GAP = 10                     # frames


def main():
  for d in ('yuv', 'rgb'):
    os.makedirs(f'{OUT}/{d}', exist_ok=True)
  build_c1m.OUT = OUT
  stops = json.load(open(f'{META}/stops.json'))
  labelled = {}
  for f in glob.glob(f'{META}/batch*.json'):
    for m in json.load(open(f)):
      labelled.setdefault(m['sid'], []).append(m['frame'])
  jobs, info = {}, {}
  for sid, v in stops.items():
    if not v or v['w'] == 1164 or split_of(sid) != 'train':
      continue
    for e in v['events']:
      dist = np.array(e['dist'])
      for d in DISTS:
        i = int(np.argmin(np.abs(dist - d)))
        if abs(dist[i] - d) > 2 or any(abs(i - j) <= GAP for j in labelled.get(sid, [])):
          continue
        fid = f'{sid}_{i:04d}a'
        jobs.setdefault(sid, {})[i] = fid
        info[fid] = {'sid': sid, 'frame': i, 'split': 'train', 'kind': 'auto', 'dist_to_stop': round(float(dist[i]), 1),
                     'device': v['device'], 'dir': f'{OUT}/yuv', 'lines': []}
  print(len(info), 'frames from', len(jobs), 'segments', flush=True)
  done = 0
  with Pool(24) as p:
    for out in p.imap_unordered(render, list(jobs.items())):
      for fid, o in out.items():
        cv2.imwrite(f'{OUT}/rgb/{fid}.jpg', o['rgb'], [cv2.IMWRITE_JPEG_QUALITY, 92])
        info[fid]['path'] = np.array(o['path'])[::4].round(2).tolist()
      done += len(out)
      if done % 500 < len(out):
        print(done, flush=True)
  info = {k: v for k, v in info.items() if os.path.exists(f'{OUT}/rgb/{k}.jpg')}
  json.dump(info, open(f'{OUT}/items.json', 'w'))
  print(len(info), 'frames written')


if __name__ == '__main__':
  main()
