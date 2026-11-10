"""Training data from the labelled comma1M frames.

For every frame in labels/store.json:
  - re-render it in the model frame (openpilot's medmodel) at 2x (1024x512), from the camera's own YUV420 planes
    (no RGB round trip),
    stored as a 3-channel Y, U, V png with U and V upsampled; the loader resamples and subsamples again
  - take the path the car actually drove next from the localizer, in the calibrated ground frame (drawn on the
    check sheet)
Splits are by segment (about 20% test), so no drive is in both.

  python build_c1m.py            -> $STOPLINE_C1M/yuv/<id>.png, c1m/items.json, c1m/check.jpg (a visual check sheet)
"""
import hashlib, json, os
from multiprocessing import Pool
import numpy as np, cv2, av
from scipy.spatial.transform import Rotation

from c1m import Segment, rot_from_euler, MED_W, MED_H
from cams import FrameCam, HGT
from paths import C1M as OUT, LABELS, META

LANE_HALF = 1.6     # half the ego lane, for the path drawn on the check sheet [m]


def split_of(sid):
  return 'test' if int(hashlib.sha1(sid.encode()).hexdigest(), 16) % 5 == 0 else 'train'


def ego_path(seg, i):
  """The car's future positions from frame i, as (forward, right) metres in frame i's calibrated frame."""
  fs = seg.loc['frame_states']
  r_dev = Rotation.from_quat(np.roll(fs[i, 3:7], -1)).as_matrix()  # device -> ECEF; stored w, x, y, z
  calib_from_dev = rot_from_euler(seg.rpy).T
  rel = (fs[i:, 0:3] - fs[i, 0:3]) @ r_dev  # rows: R^T (p_j - p_i), device frame
  xy = (rel @ calib_from_dev.T)[:, :2]
  keep = [0]
  for k in range(1, len(xy)):  # thin to points 0.5 m apart, forward only
    if np.linalg.norm(xy[k] - xy[keep[-1]]) >= 0.5:
      keep.append(k)
  return xy[keep]


def render(job):
  sid, wanted = job
  seg = Segment(sid)
  a2 = seg.camera_from_model(2.0)
  a_uv = np.diag([0.5, 0.5, 1]) @ seg.camera_from_model(1.0)  # model pixel (1x) -> chroma-plane pixel
  out = {}
  with av.open(seg.video) as c:
    for i, fr in enumerate(c.decode(video=0)):
      if i not in wanted: continue
      yuv = fr.to_ndarray(format='yuv420p')
      h, w = fr.height, fr.width
      buf, ch, cw = yuv.ravel(), h // 2, w // 2  # planar: Y (h, w), then U and V (h/2, w/2)
      y = buf[:h * w].reshape(h, w)
      u = buf[h * w:h * w + ch * cw].reshape(ch, cw)
      v = buf[h * w + ch * cw:h * w + 2 * ch * cw].reshape(ch, cw)
      flags = cv2.WARP_INVERSE_MAP | cv2.INTER_AREA
      Y = cv2.warpPerspective(y, a2, (2 * MED_W, 2 * MED_H), flags=flags, borderValue=16)
      U = cv2.warpPerspective(u, a_uv, (MED_W, MED_H), flags=flags, borderValue=128)
      V = cv2.warpPerspective(v, a_uv, (MED_W, MED_H), flags=flags, borderValue=128)
      UV = cv2.resize(np.dstack([U, V]), (2 * MED_W, 2 * MED_H), interpolation=cv2.INTER_LINEAR)
      img = np.dstack([Y, UV])
      fid = wanted[i]
      cv2.imwrite(f'{OUT}/yuv/{fid}.png', img)
      rgb = cv2.cvtColor(seg.warp(fr.to_ndarray(format='rgb24'), 2.0), cv2.COLOR_RGB2BGR)
      out[fid] = {'rgb': rgb, 'path': ego_path(seg, i).tolist()}
      if len(out) == len(wanted): break
  return out


if __name__ == '__main__':
  os.makedirs(f'{OUT}/yuv', exist_ok=True)
  store = json.load(open(f'{LABELS}/store.json'))
  meta = {}
  for f in sorted(os.listdir(META)):
    if f.startswith('batch') and f.endswith('.json'):
      meta.update({m['id']: m for m in json.load(open(f'{META}/{f}'))})
  jobs = {}
  for fid in store:
    m = meta[fid]
    jobs.setdefault(m['sid'], {})[m['frame']] = fid
  pool = Pool(16)
  items = {}
  sheet = []
  with pool:
    for out in pool.imap_unordered(render, list(jobs.items())):
      for fid in out:
        lab, m = store[fid], meta[fid]
        path = np.array(out[fid]['path'])
        items[fid] = {'split': split_of(m['sid']), 'sid': m['sid'], 'kind': m['kind'], 'device': m['device'],
                      'status': lab['status'], 'lines': lab['lines'],
                      'dist_to_stop': m['dist_to_stop'], 'path': path[::4].round(2).tolist()}
        if len(sheet) < 24 and len(sheet) < len(items) and (lab['lines'] or len(sheet) % 3 == 0):
          sheet.append((fid, out[fid]['rgb'], path, lab))
  json.dump(items, open(f'{OUT}/items.json', 'w'))
  # a check sheet: the drawn path (green) and the labelled lines (cyan)
  tiles = []
  for fid, rgb, path, lab in sheet:
    im = rgb.copy()
    cam = FrameCam()
    for side in (-LANE_HALF, LANE_HALF):
      pts = [(x, y + side, HGT) for x, y in path if x > 4]
      if len(pts) > 1:
        cv2.polylines(im, [(2 * cam.pixel(pts)[0]).astype(np.int32)], False, (0, 200, 0), 2)
    for l in lab['lines']:
      cv2.line(im, (int(2 * l[0]), int(2 * l[1])), (int(2 * l[2]), int(2 * l[3])), (255, 255, 0), 2)
    cv2.putText(im, fid[:8] + fid[-5:], (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    tiles.append(cv2.resize(im, (512, 256)))
  while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
  rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
  cv2.imwrite(f'{OUT}/check.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
  n = {s: sum(v['split'] == s for v in items.values()) for s in ('train', 'test')}
  print(len(items), 'frames', n,
        '| test segments', len({v['sid'] for v in items.values() if v['split'] == 'test'}))
