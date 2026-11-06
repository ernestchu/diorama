"""Pilot training on the SLSDNet stop-line dataset (+ teacher lead boxes). Two GPUs:

  torchrun --nproc_per_node 2 train.py --out runs/pilot
"""
import argparse, json, math, os, random, time
import numpy as np, cv2, torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from model import Net, decode, STRIDE, END_SCALE, WH_SCALE

from paths import SLSD as DATA
W, H = 512, 256
GW, GH = W // STRIDE, H // STRIDE


def clip_segment(x1, y1, x2, y2):
  """Liang-Barsky: the part of the segment inside the frame, or None."""
  t0, t1, dx, dy = 0.0, 1.0, x2 - x1, y2 - y1
  for p, q in ((-dx, x1), (dx, W - x1), (-dy, y1), (dy, H - y1)):
    if p == 0:
      if q < 0: return None
    else:
      t = q / p
      if p < 0: t0 = max(t0, t)
      else: t1 = min(t1, t)
  if t0 > t1: return None
  return [x1 + t0 * dx, y1 + t0 * dy, x1 + t1 * dx, y1 + t1 * dy]


class Pilot(Dataset):
  def __init__(self, split, train):
    items = json.load(open(f'{DATA}/items.json'))
    self.contrast = {k: v.get('contrast') for k, v in items.items()}
    self.items = [(k, v) for k, v in sorted(items.items()) if v['split'] == split]
    self.train = train

  def __len__(self):
    return len(self.items)

  def geometry(self, v):
    w, h = v['w'], v['h']
    full = min(h, w / 2)
    if not self.train:  # the largest 2:1 window, bottom-aligned and centred
      ch = full; x0, y0 = (w - 2 * ch) / 2, h - ch
      return x0, y0, ch, 0.0
    ch = full * random.uniform(0.55, 1.0)
    for _ in range(10):
      x0, y0 = random.uniform(0, w - 2 * ch), random.uniform(0, h - ch)
      if v['line'] is None or random.random() < 0.2: break
      x1, y1, x2, y2 = v['line']
      if x0 <= x1 and x2 <= x0 + 2 * ch and y0 <= min(y1, y2) and max(y1, y2) <= y0 + ch: break
    return x0, y0, ch, random.uniform(-3, 3)

  def __getitem__(self, i):
    name, v = self.items[i]
    img = cv2.imread(f'{DATA}/yuv/{name}.png', cv2.IMREAD_UNCHANGED)  # Y, U, V
    x0, y0, ch, rot = self.geometry(v)
    s = H / ch
    a = np.array([[s, 0, -s * x0], [0, s, -s * y0], [0, 0, 1]])
    r = cv2.getRotationMatrix2D((W / 2, H / 2), rot, 1.0)
    a = np.vstack([r, [0, 0, 1]]) @ a
    flip = self.train and random.random() < 0.5
    if flip:
      a = np.array([[-1, 0, W], [0, 1, 0], [0, 0, 1]]) @ a
    out = cv2.warpAffine(img, a[:2], (W, H), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    pt = lambda x, y: (a[0, 0] * x + a[0, 1] * y + a[0, 2], a[1, 0] * x + a[1, 1] * y + a[1, 2])

    line, line_ok = None, 1.0
    if v['line'] is not None:
      x1, y1, x2, y2 = v['line']
      (ax, ay), (bx, by) = pt(x1, y1), pt(x2, y2)
      c = clip_segment(ax, ay, bx, by)
      full = math.hypot(bx - ax, by - ay)
      if c is not None and math.hypot(c[2] - c[0], c[3] - c[1]) >= max(8, 0.3 * full):
        line = c if c[0] <= c[2] else [c[2], c[3], c[0], c[1]]
      elif c is not None:
        line_ok = 0.0  # a sliver of a line: neither a line nor no line
    lead, lead_ok = None, 1.0
    if v['lead'] is not None:
      x1, y1, x2, y2 = v['lead']
      ps = np.array([pt(x1, y1), pt(x2, y1), pt(x1, y2), pt(x2, y2)])
      bx1, by1, bx2, by2 = ps[:, 0].min(), ps[:, 1].min(), ps[:, 0].max(), ps[:, 1].max()
      cx1, cy1, cx2, cy2 = max(bx1, 0), max(by1, 0), min(bx2, W), min(by2, H)
      area = (bx2 - bx1) * (by2 - by1)
      vis = max(0, cx2 - cx1) * max(0, cy2 - cy1)
      if vis >= 0.6 * area and cx2 - cx1 >= 4 and cy2 - cy1 >= 4:
        lead = [cx1, cy1, cx2, cy2]
      elif vis > 0:
        lead_ok = 0.0

    y = out[..., 0].astype(np.float32)
    uv = cv2.resize(out[..., 1:], (W // 2, H // 2), interpolation=cv2.INTER_AREA).astype(np.float32)  # 4:2:0
    if self.train:
      f = y / 255
      f = f ** random.uniform(0.6, 1.5)
      f = f * random.uniform(0.7, 1.3) + random.uniform(-0.15, 0.15)
      if random.random() < 0.3:
        f = cv2.GaussianBlur(f, (0, 0), random.uniform(0.3, 1.2))
      f = f + np.random.randn(*f.shape).astype(np.float32) * random.uniform(0, 0.025)
      y = np.clip(f * 255, 0, 255)
      # saturation and white balance
      uv = (uv - 128) * random.uniform(0.6, 1.4) + np.random.uniform(-4, 4, 2).astype(np.float32)
      uv = np.clip(uv + 128 + np.random.randn(*uv.shape).astype(np.float32) * random.uniform(0, 1.5), 0, 255)
    x = torch.from_numpy(y)[None]
    c = torch.from_numpy(np.ascontiguousarray(uv.transpose(2, 0, 1)))
    return x, c, targets(line, lead, line_ok, lead_ok), name


def gaussian(cx, cy, sx, sy):
  ys, xs = np.mgrid[0:GH, 0:GW].astype(np.float32)
  return np.exp(-((xs - cx) ** 2 / (2 * sx * sx) + (ys - cy) ** 2 / (2 * sy * sy)))


def targets(line, lead, line_ok, lead_ok):
  t = {k: np.zeros(s, np.float32) for k, s in (('lheat', (GH, GW)), ('lmask', (GH, GW)), ('lreg', (6,)),
                                                ('cheat', (GH, GW)), ('creg', (4,)))}
  t['lidx'] = np.int64(0); t['cidx'] = np.int64(0)
  t['lhas'] = np.float32(0); t['chas'] = np.float32(0)
  t['lok'] = np.float32(line_ok); t['cok'] = np.float32(lead_ok)
  if line is not None:
    x1, y1, x2, y2 = [v / STRIDE for v in line]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ix, iy = min(GW - 1, int(cx)), min(GH - 1, int(cy))
    t['lheat'] = gaussian(cx, cy, 2.0, 1.2)
    t['lheat'][iy, ix] = 1.0
    t['lidx'] = np.int64(iy * GW + ix)
    t['lreg'] = np.array([cx - ix, cy - iy, (x1 - cx) / END_SCALE, (y1 - cy) / END_SCALE,
                          (x2 - cx) / END_SCALE, (y2 - cy) / END_SCALE], np.float32)
    m = np.zeros((GH * 4, GW * 4), np.uint8)
    cv2.line(m, (int(x1 * 4), int(y1 * 4)), (int(x2 * 4), int(y2 * 4)), 1, 6)
    t['lmask'] = cv2.resize(m.astype(np.float32), (GW, GH), interpolation=cv2.INTER_AREA)
    t['lhas'] = np.float32(1)
  if lead is not None:
    x1, y1, x2, y2 = [v / STRIDE for v in lead]
    cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    ix, iy = min(GW - 1, int(cx)), min(GH - 1, int(cy))
    t['cheat'] = gaussian(cx, cy, max(0.8, bw / 6), max(0.8, bh / 6))
    t['cheat'][iy, ix] = 1.0
    t['cidx'] = np.int64(iy * GW + ix)
    t['creg'] = np.array([cx - ix, cy - iy, bw / WH_SCALE, bh / WH_SCALE], np.float32)
    t['chas'] = np.float32(1)
  return {k: torch.as_tensor(v) for k, v in t.items()}


def focal(logit, gt, ok):
  """CenterNet's penalty-reduced focal loss, per image weighted by ok, normalised by positives."""
  p = torch.sigmoid(logit.float()).clamp(1e-4, 1 - 1e-4)
  pos = (gt >= 1.0).float()
  l = -(pos * (1 - p) ** 2 * torch.log(p) + (1 - pos) * (1 - gt) ** 4 * p ** 2 * torch.log(1 - p))
  l = l.flatten(1).sum(1) * ok
  return l.sum() / pos.flatten(1).sum(1).mul(ok).sum().clamp(min=1)


def at(maps, idx):
  """(N, C, GH, GW) maps -> (N, C) at each image's flat index."""
  return maps.flatten(2).gather(2, idx[:, None, None].expand(-1, maps.shape[1], 1))[..., 0]


def losses(line, lead, t):
  lw = t['lhas'] * t['lok']
  cw = t['chas'] * t['cok']
  lr = at(line[:, 1:7].float(), t['lidx'])
  cr = at(lead[:, 1:5].float(), t['cidx'])
  out = {
    'lheat': focal(line[:, 0], t['lheat'], t['lok']),
    'loff': (F.l1_loss(lr[:, :2], t['lreg'][:, :2], reduction='none').sum(1) * lw).sum() / lw.sum().clamp(min=1),
    'lend': 4 * (F.l1_loss(lr[:, 2:], t['lreg'][:, 2:], reduction='none').sum(1) * lw).sum() / lw.sum().clamp(min=1),
    'lmask': 0.5 * (F.binary_cross_entropy_with_logits(line[:, 7].float(), t['lmask'], reduction='none')
                    .flatten(1).mean(1) * t['lok']).mean() * 10,
    'cheat': focal(lead[:, 0], t['cheat'], t['cok']),
    'coff': (F.l1_loss(cr[:, :2], t['creg'][:, :2], reduction='none').sum(1) * cw).sum() / cw.sum().clamp(min=1),
    'cwh': 2 * (F.l1_loss(cr[:, 2:], t['creg'][:, 2:], reduction='none').sum(1) * cw).sum() / cw.sum().clamp(min=1),
  }
  return sum(out.values()), out


def iou(a, b):
  ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
  u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
  return ix * iy / u if u > 0 else 0


@torch.no_grad()
def evaluate(model, loader, device):
  """Stop line: a hit is the top line with both ends within 20 px (of 512) of the label's.
  Lead: a hit is the box with IoU >= 0.5 against the teacher's lead."""
  model.eval()
  recs = []
  for x, c, t, names in loader:
    line, lead = model(x.to(device), c.to(device))
    for n, d in enumerate(decode(line.float(), lead.float(), line_thresh=0.0, lead_thresh=0.0, max_lines=1)):
      rec = {'name': names[n], 'lok': t['lok'][n].item(), 'cok': t['cok'][n].item()}
      if t['lhas'][n] > 0:
        g = t['lreg'][n].numpy(); idx = t['lidx'][n].item()
        cx, cy = idx % GW + g[0], idx // GW + g[1]
        gl = [(cx + g[2] * END_SCALE) * STRIDE, (cy + g[3] * END_SCALE) * STRIDE,
              (cx + g[4] * END_SCALE) * STRIDE, (cy + g[5] * END_SCALE) * STRIDE]
        rec['gl'] = gl
      if t['chas'][n] > 0:
        g = t['creg'][n].numpy(); idx = t['cidx'][n].item()
        cx, cy, bw, bh = idx % GW + g[0], idx // GW + g[1], g[2] * WH_SCALE, g[3] * WH_SCALE
        rec['gc'] = [(cx - bw / 2) * STRIDE, (cy - bh / 2) * STRIDE, (cx + bw / 2) * STRIDE, (cy + bh / 2) * STRIDE]
      rec['pl'] = d['lines'][0] if d['lines'] else None
      rec['pc'] = d['lead']
      recs.append(rec)
  model.train()
  return recs


def score(recs, subset=None):
  """Precision, recall and F1 at two thresholds. With subset (a set of names), only those images count."""
  res = {}
  if subset is not None:
    recs = [r for r in recs if r['name'] in subset]
  for th in (0.3, 0.5):
    tp = fp = fn = 0; errs = []; rows = []
    for r in recs:
      if not r['lok']: continue
      p = r['pl'] if r['pl'] and r['pl'][4] >= th else None
      g = r.get('gl')
      if p and g:
        e = max(math.hypot(p[0] - g[0], p[1] - g[1]), math.hypot(p[2] - g[2], p[3] - g[3]))
        row = abs((p[1] + p[3]) / 2 - (g[1] + g[3]) / 2)
        if e <= 20: tp += 1; errs.append(e); rows.append(row)
        else: fp += 1; fn += 1
      elif p: fp += 1
      elif g: fn += 1
    pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
    res[f'line@{th}'] = {'P': round(pr, 4), 'R': round(rc, 4), 'F1': round(2 * pr * rc / max(1e-9, pr + rc), 4),
                         'end_err_med': round(float(np.median(errs)), 2) if errs else None,
                         'row_err_med': round(float(np.median(rows)), 2) if rows else None}
    tp = fp = fn = 0; ious = []
    for r in recs:
      if not r['cok']: continue
      p = r['pc'] if r['pc'] and r['pc'][4] >= th else None
      g = r.get('gc')
      if p and g:
        o = iou(p, g)
        if o >= 0.5: tp += 1; ious.append(o)
        else: fp += 1; fn += 1
      elif p: fp += 1
      elif g: fn += 1
    pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
    res[f'lead@{th}'] = {'P': round(pr, 4), 'R': round(rc, 4), 'F1': round(2 * pr * rc / max(1e-9, pr + rc), 4),
                         'iou_mean': round(float(np.mean(ious)), 3) if ious else None}
  return res


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', required=True)
  ap.add_argument('--epochs', type=int, default=150)
  ap.add_argument('--bs', type=int, default=32)
  ap.add_argument('--lr', type=float, default=1e-3)
  ap.add_argument('--color', action='store_true', help='Y + 4:2:0 UV input instead of Y alone')
  ap.add_argument('--seed', type=int, default=0)
  args = ap.parse_args()
  dist.init_process_group('nccl')
  rank, world = dist.get_rank(), dist.get_world_size()
  device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
  torch.cuda.set_device(device)
  s0 = args.seed * 1000 + rank
  torch.manual_seed(s0); random.seed(s0); np.random.seed(s0)
  os.makedirs(args.out, exist_ok=True)

  train = Pilot('train', True)
  sampler = DistributedSampler(train, shuffle=True)
  dl = DataLoader(train, args.bs, sampler=sampler, num_workers=8, drop_last=True, pin_memory=True, persistent_workers=True,
                  worker_init_fn=lambda w: (random.seed(s0 * 100 + w), np.random.seed(s0 * 100 + w)))
  test_set = Pilot('test', False)
  test = DataLoader(test_set, 32, num_workers=8)
  # the quarter of test lines that stand out least from the road in luma: where colour could matter
  cs = {n: test_set.contrast[n]['y'] for n, _ in test_set.items if test_set.contrast[n]}
  low = {n for n, v in cs.items() if v <= np.percentile(list(cs.values()), 25)}
  torch.manual_seed(args.seed)  # the same initial weights on every rank and for both inputs
  model = Net(color=args.color).to(device)
  model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
  ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
  opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
  steps = args.epochs * len(dl)
  sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
  log = open(f'{args.out}/log.jsonl', 'a') if rank == 0 else None
  for ep in range(args.epochs):
    sampler.set_epoch(ep)
    tot = {}
    for x, c, t, _ in dl:
      x, c = x.to(device, non_blocking=True), c.to(device, non_blocking=True)
      t = {k: v.to(device, non_blocking=True) for k, v in t.items()}
      with torch.autocast('cuda', dtype=torch.bfloat16):
        line, lead = ddp(x, c)
      loss, parts = losses(line, lead, t)
      opt.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
      opt.step(); sched.step()
      for k, v in parts.items(): tot[k] = tot.get(k, 0) + v.item() / len(dl)
    if rank == 0 and (ep % 10 == 9 or ep == args.epochs - 1):
      recs = evaluate(model, test, device)
      low_res = score(recs, low)
      rec = {'epoch': ep + 1, 'loss': {k: round(v, 4) for k, v in tot.items()}, **score(recs),
             'low_contrast_line@0.3': low_res['line@0.3'], 'low_contrast_n': len(low)}
      print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
      torch.save(model.state_dict(), f'{args.out}/last.pt')
  dist.destroy_process_group()


if __name__ == '__main__':
  main()
