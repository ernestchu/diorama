"""The labelling round trip.

  labels.py merge FILE...   checks downloaded session files and adds them to labels/store.json
  labels.py manifest        lists every sampled frame not yet labelled in site/manifest.json and
                            removes labelled frames' images from the site; also versions the
                            app.js/style.css links in index.html (run it after editing the site)

store.json maps a frame id to its label: {status: lines|none, lines: [[x1,y1,x2,y2]] (at most one, in the ego lane)} in
medmodel pixels (512x256), plus the session file it came from. A session's unclear counts as none and its hard flag is
dropped. A label with reviewed false is a suggestion still to be checked on the site: the
manifest lists it, and prefill.json carries it with what to check it against. A merged label is reviewed.
"""
import glob, hashlib, json, math, os, re, sys

from paths import LABELS, META, SITE
STORE, EXTRA = f'{LABELS}/store.json', f'{LABELS}/overlays.json'


def load_store():
  return json.load(open(STORE)) if os.path.exists(STORE) else {}


def sampled():
  frames = []
  for f in sorted(glob.glob(f'{META}/batch*.json')):
    frames += [m['id'] for m in json.load(open(f))]
  return frames


def check(label, fid):
  st = label.get('status')
  if st not in ('lines', 'none'):
    raise ValueError(f'{fid}: status {st!r}')
  ls = label.get('lines', [])
  if (st == 'lines') != bool(ls) or len(ls) > 1:
    raise ValueError(f'{fid}: status {st} with {len(ls)} lines')
  for l in ls:
    if len(l) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in l):
      raise ValueError(f'{fid}: bad line {l}')
    if not (-1 <= l[0] <= 513 and -1 <= l[2] <= 513 and -1 <= l[1] <= 257 and -1 <= l[3] <= 257):
      raise ValueError(f'{fid}: line outside the frame {l}')


def merge(files):
  store, known = load_store(), set(sampled())
  for path in files:
    data = json.load(open(path))
    if data.get('format') != 'stopline-labels/1':
      raise SystemExit(f'{path}: not a stop-line labels file')
    new = changed = 0
    for fid, label in data['labels'].items():
      if fid not in known:
        raise SystemExit(f'{path}: {fid} was never sampled')
      if label.get('status') == 'unclear':
        label = {**label, 'status': 'none', 'lines': []}
      check(label, fid)
      entry = {'status': label['status'], 'lines': label['lines'], 't': label.get('t'),
               'file': os.path.basename(path), 'reviewed': True}
      if fid in store:
        old = store[fid]
        changed += (old['status'], old['lines']) != (entry['status'], entry['lines'])
        entry.update({k: old[k] for k in ('orig', 'why') if k in old})
      else:
        new += 1
      store[fid] = entry
    print(f'{path}: {len(data["labels"])} labels, {new} new, {changed} changed')
  os.makedirs(os.path.dirname(STORE), exist_ok=True)
  tmp = STORE + '.tmp'
  json.dump(store, open(tmp, 'w'), indent=1)
  os.replace(tmp, STORE)
  counts = {}
  for v in store.values():
    counts[v['status']] = counts.get(v['status'], 0) + 1
  print('store:', len(store), counts, 'lines:', sum(len(v['lines']) for v in store.values()))


def stamp():
  """Version the script and stylesheet links by content, so a cached old copy is never paired with a new page."""
  page = open(f'{SITE}/index.html').read()
  for name in ('app.js', 'style.css'):
    v = hashlib.sha1(open(f'{SITE}/{name}', 'rb').read()).hexdigest()[:8]
    page = re.sub(rf'"{re.escape(name)}(\?v=\w+)?"', f'"{name}?v={v}"', page)
  open(f'{SITE}/index.html', 'w').write(page)


def manifest():
  store = load_store()
  done = {f for f, v in store.items() if v.get('reviewed', True)}
  pending = [f for f in sampled() if f not in done]
  for f in done:
    p = f'{SITE}/frames/{f}.jpg'
    if os.path.exists(p):
      os.remove(p)
  missing = [f for f in pending if not os.path.exists(f'{SITE}/frames/{f}.jpg')]
  if missing:
    raise SystemExit(f'{len(missing)} pending frames have no image, e.g. {missing[0]}')
  version = hashlib.sha1('\n'.join(pending).encode()).hexdigest()[:12]
  json.dump({'version': version, 'frame_size': [512, 256], 'image_scale': 2, 'frames': pending},
            open(f'{SITE}/manifest.json', 'w'))
  extra = json.load(open(EXTRA)) if os.path.exists(EXTRA) else {}
  prefill = {f: {**{k: store[f][k] for k in ('status', 'lines', 'why', 'orig') if k in store[f]}, **extra.get(f, {})}
             for f in pending if f in store}
  json.dump(prefill, open(f'{SITE}/prefill.json', 'w'))
  stamp()
  print('manifest', version, len(pending), 'frames to check;', len(done), 'done')


if __name__ == '__main__':
  {'merge': lambda: merge(sys.argv[2:]), 'manifest': manifest}[sys.argv[1]]()
