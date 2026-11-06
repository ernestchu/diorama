"""Training data from the labelled comma1M frames.

For every frame in labels/store.json:
  - re-render it in the model frame (openpilot's medmodel) at 2x (1024x512), from the camera's own YUV420 planes
    (no RGB round trip),
    stored as a 3-channel Y, U, V png with U and V upsampled; the loader resamples and subsamples again
  - run the teacher (D-FINE) for vehicle boxes
  - take the path the car actually drove next from the localizer, in the calibrated ground frame, and choose the
    lead: the nearest vehicle whose ground point lies within LANE_HALF of that path and whose box is not
    side-on (crossing traffic stands on the future path at intersections). This stands in for
    openpilot's planned path until the driving model is run on these frames.
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

LANE_HALF = 1.6     # a vehicle whose ground point is this close to the ego path is in the ego lane [m]
MAX_LEAD = 80.0     # [m]
TURN_SLOPE = 0.15   # where the path has turned more than this (offset / distance), traffic there crosses or
                    # comes towards us: e.g. a left turn at an intersection
MAX_ASPECT = 1.9    # wider boxes are side-on vehicles, i.e. crossing traffic, never the lead


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


def path_lateral(path, d):
  """Lateral offset of the path at forward distance d, extended straight along its last heading."""
  fwd = path[:, 0]
  if len(path) >= 2 and fwd.max() >= d:
    k = int(np.argmax(fwd >= d))  # the first crossing: the path can turn round and come back later
    k = min(max(k, 1), len(path) - 1)
    a, b = path[k - 1], path[k]
    t = (d - a[0]) / max(1e-6, b[0] - a[0])
    return a[1] + t * (b[1] - a[1])
  if len(path) >= 2:
    tail = path[max(0, len(path) - 10):]
    h = tail[-1] - tail[0]
    slope = h[1] / h[0] if h[0] > 0.5 else 0.0
    return path[-1, 1] + slope * (d - path[-1, 0])
  return 0.0


def choose_lead(boxes, path, cam, scale=2.0):
  """boxes at `scale` x model-frame resolution -> (lead box in model-frame pixels or None, per-box ground info)."""
  best, info = None, []
  for x1, y1, x2, y2, s in boxes:
    d, lat, _ = cam.ground((x1 + x2) / 2 / scale, y2 / scale)[0]
    if not np.isfinite(d) or d >= HGT * cam.f / 2:  # within 2 px of the horizon or above it
      info.append(None); continue
    p_lat = path_lateral(path, d)
    off = lat - p_lat
    turned = abs(p_lat) > 1.0 + TURN_SLOPE * d
    info.append([round(d, 1), round(off, 2)])
    side_on = (x2 - x1) > MAX_ASPECT * (y2 - y1)
    if d <= MAX_LEAD and abs(off) <= LANE_HALF and not side_on and not turned and (best is None or d < best[0]):
      best = (d, [x1 / scale, y1 / scale, x2 / scale, y2 / scale])
  return (best[1] if best else None), info


NEAR_ROW = 250      # a box reaching this row (of 256) is cut by the frame's bottom: the vehicle is within ~6 m
NEAR_BAND = 20      # ... and is the lead when it covers the centre columns 256 +- this, whatever the path does next
FULL = (500, 240)   # boxes this wide and tall are mostly the teacher's false positives on dark or blurred road


def choose_lead_near(boxes, path, cam, scale=2.0):
  """choose_lead, but a vehicle cut by the frame's bottom across the centre columns is the lead outright: at that range
  it blocks the car whether or not the driven path later turns or pulls round it. Returns (lead, info, ambiguous):
  ambiguous when a full-frame box is the nearest candidate, which the teacher alone can't settle."""
  lead, info = choose_lead(boxes, path, cam, scale)
  near, ambiguous = None, False
  for x1, y1, x2, y2, s in boxes:
    x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
    if y2 < NEAR_ROW or x1 > cam.cx - NEAR_BAND or x2 < cam.cx + NEAR_BAND:
      continue
    if x2 - x1 >= FULL[0] and y2 - y1 >= FULL[1]:
      ambiguous = True; continue
    if (x2 - x1) > MAX_ASPECT * (y2 - y1):
      continue
    if near is None or (x2 - x1) * (y2 - y1) > (near[2] - near[0]) * (near[3] - near[1]):
      near = [x1, y1, x2, y2]
  if near is not None:
    return near, info, False
  return lead, info, ambiguous and lead is None


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
  pool = Pool(16)  # fork the decoders before CUDA starts in this process
  from teacher import Teacher
  teacher = Teacher()
  items = {}
  sheet = []
  with pool:
    for out in pool.imap_unordered(render, list(jobs.items())):
      fids = list(out)
      boxes = teacher([cv2.cvtColor(out[f]['rgb'], cv2.COLOR_BGR2RGB) for f in fids])
      for fid, bx in zip(fids, boxes):
        lab, m = store[fid], meta[fid]
        path = np.array(out[fid]['path'])
        cam = FrameCam()
        lead, info, amb = choose_lead_near(bx, path, cam)
        items[fid] = {'split': split_of(m['sid']), 'sid': m['sid'], 'kind': m['kind'], 'device': m['device'],
                      'status': lab['status'], 'lines': lab['lines'],
                      'dist_to_stop': m['dist_to_stop'], 'lead': lead, 'ambiguous': amb,
                      'vehicles': [[round(v / 2, 1) for v in b[:4]] + [round(b[4], 3)] for b in bx],
                      'ground': info, 'path': path[::4].round(2).tolist()}
        if len(sheet) < 24 and len(sheet) < len(items) and (lead or len(sheet) % 3 == 0):
          sheet.append((fid, out[fid]['rgb'], path, lead, lab))
  json.dump(items, open(f'{OUT}/items.json', 'w'))
  # a check sheet: the drawn path (green), the lead (orange), every teacher box (grey), the labelled lines (cyan)
  tiles = []
  for fid, rgb, path, lead, lab in sheet:
    im = rgb.copy()
    cam = FrameCam()
    for side in (-LANE_HALF, LANE_HALF):
      pts = [(x, y + side, HGT) for x, y in path if x > 4]
      if len(pts) > 1:
        cv2.polylines(im, [(2 * cam.pixel(pts)[0]).astype(np.int32)], False, (0, 200, 0), 2)
    for b in items[fid]['vehicles']:
      cv2.rectangle(im, (int(2 * b[0]), int(2 * b[1])), (int(2 * b[2]), int(2 * b[3])), (160, 160, 160), 1)
    if lead:
      cv2.rectangle(im, (int(2 * lead[0]), int(2 * lead[1])), (int(2 * lead[2]), int(2 * lead[3])), (0, 120, 255), 3)
    for l in lab['lines']:
      cv2.line(im, (int(2 * l[0]), int(2 * l[1])), (int(2 * l[2]), int(2 * l[3])), (255, 255, 0), 2)
    cv2.putText(im, fid[:8] + fid[-5:], (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    tiles.append(cv2.resize(im, (512, 256)))
  while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
  rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
  cv2.imwrite(f'{OUT}/check.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
  n = {s: sum(v['split'] == s for v in items.values()) for s in ('train', 'test')}
  print(len(items), 'frames', n, '| with a lead', sum(v['lead'] is not None for v in items.values()),
        '| test segments', len({v['sid'] for v in items.values() if v['split'] == 'test'}))
