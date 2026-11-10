"""Put the VLM-labelled frames that need a person (auto_split.py) on the labelling site.

Each frame is prefilled with the VLM's line (or none); the student's line is the dashed comparison line ('orig'), and
the reason line says who found what. The frames join the sampled set as meta/batch_auto.json, so labels.py merge
accepts them, and go into labels/store.json as unreviewed suggestions.

  python queue_auto.py && python labels.py manifest
"""
import json, os, shutil

from c1m import Segment
from paths import C1M, LABELS, META, SITE

A = f'{C1M}/auto'


def main():
  items = json.load(open(f'{A}/items.json'))
  vlm = json.load(open(f'{A}/vlm.json'))
  stu = json.load(open(f'{A}/student.json'))
  review = json.load(open(f'{A}/to_review.json'))
  store_path = f'{LABELS}/store.json'
  store = json.load(open(store_path)) if os.path.exists(store_path) else {}
  os.makedirs(f'{SITE}/frames', exist_ok=True)
  meta, segs = [], {}
  for k in review:
    v = items[k]
    s = segs.get(v['sid']) or segs.setdefault(v['sid'], Segment(v['sid']))
    meta.append({'id': k, 'sid': v['sid'], 'frame': v['frame'], 'kind': 'auto', 'dist_to_stop': v['dist_to_stop'],
                 'width': s.w, 'device': s.device, 'rpy': [float(x) for x in s.rpy]})
    h = vlm[k]
    sc, *sl = stu[k]
    store[k] = {'status': 'lines' if h else 'none', 'lines': [h] if h else [], 'file': 'auto: vlm',
                'reviewed': False, 'why': f"VLM: {'line' if h else 'none'} · model: {'line' if sc >= 0.3 else 'none'} ({sc:.2f})",
                **({'orig': [round(x, 2) for x in sl]} if sc >= 0.3 else {})}
    shutil.copy(f'{A}/rgb/{k}.jpg', f'{SITE}/frames/{k}.jpg')
  json.dump(meta, open(f'{META}/batch_auto.json', 'w'), indent=1)
  json.dump(store, open(store_path, 'w'), indent=1)
  print(len(meta), 'frames queued;', sum(1 for k in review if vlm[k]), 'prefilled with a line')


if __name__ == '__main__':
  main()
