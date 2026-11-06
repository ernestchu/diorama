"""The driving model's lane lines at labelled comma1M frames, on the GPU.

Each segment's frames are run from WARMUP frames before the first wanted one, the way Diorama runs a drive (the model
keeps state), with commasim's driving-model runner (sim/commasim) on onnxruntime-gpu: the CUDA libraries on
LD_LIBRARY_PATH, and DRIVING_MODEL set to the model's ONNX.

  python ego_c1m.py FRAMES.json OUT.json [N]    FRAMES: [frame id]; OUT: {frame id: {lanes: 4 x 33 x (y, z), lane_prob,
                                                edges: 2 x 33 x (y, z)}} in metres (right, down); N: only N segments
"""
import collections, json, multiprocessing as mp, os, sys, time
from pathlib import Path
import numpy as np

from paths import REPO, DRIVING_MODEL as MODEL
WARMUP, GPUS, PER_GPU = 100, 2, 4


def start(queue):
  os.environ['CUDA_VISIBLE_DEVICES'] = str(queue.get())
  global model, route, parse
  sys.path.insert(0, f'{REPO}/sim')
  from commasim.driving import DrivingModel
  from commasim import route
  from commasim.outputs import parse
  model = DrivingModel(MODEL, 'cuda')
  if model.sess.get_providers()[0] != 'CUDAExecutionProvider':
    raise SystemExit(f'driving model is not on the GPU: {model.sess.get_providers()}')


def run(job):
  sid, wanted = job
  from c1m import Segment, FOCAL, WIDE_FOCAL
  seg = Segment(sid)
  f, e = Path(seg.video), Path(seg.wide_video)
  size = route.probe_size(f)
  if route.probe_size(e) != size:
    raise ValueError(f'{sid}: cameras differ in size')
  mn = route.warp_matrix(seg.rpy, FOCAL[seg.w], size, big=False)
  mw = route.warp_matrix(seg.rpy, WIDE_FOCAL[seg.w], size, big=True)
  first, last = max(min(wanted) - WARMUP, 0), max(wanted)
  model.reset()
  out = {}
  for k, (a, b) in enumerate(zip(route.read_yuv(f, size), route.read_yuv(e, size))):
    if k > last:
      break
    if k < first:
      continue
    vector = model.step(np.stack([route.pack(a, mn, size), route.pack(b, mw, size)]))
    if k in wanted:
      p = parse(vector, model.slices, ('lane_lines', 'lane_lines_prob', 'road_edges'))
      out[wanted[k]] = {'lanes': p['lane_lines'].round(3).tolist(), 'lane_prob': p['lane_lines_prob'][1::2].round(3).tolist(),
                        'edges': p['road_edges'].round(3).tolist(), 'warmup': k - first}
  return out


if __name__ == '__main__':
  if not MODEL:
    raise SystemExit('set DRIVING_MODEL to the driving model\'s ONNX')
  frames, out_path = json.load(open(sys.argv[1])), sys.argv[2]
  by = collections.defaultdict(dict)
  for fid in frames:
    sid, f = fid.split('_')
    by[sid][int(f)] = fid
  jobs = sorted(by.items())
  if len(sys.argv) > 3:
    jobs = jobs[:int(sys.argv[3])]
  ctx = mp.get_context('spawn')
  gpus = ctx.Queue()
  for i in range(GPUS * PER_GPU):
    gpus.put(i % GPUS)
  res, t0 = {}, time.time()
  with ctx.Pool(GPUS * PER_GPU, start, (gpus,)) as pool:
    for n, r in enumerate(pool.imap_unordered(run, jobs)):
      res.update(r)
      if n % 50 == 0:
        print(f'{n + 1}/{len(jobs)} segments, {time.time() - t0:.0f} s', flush=True)
  json.dump(res, open(out_path, 'w'))
  print(len(res), 'frames from', len(jobs), 'segments in', round(time.time() - t0), 's')
