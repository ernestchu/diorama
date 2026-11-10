"""Training on the labelled comma1M frames (build_c1m.py first). Two GPUs:

  python -m torch.distributed.run --nproc_per_node 2 train_c1m.py --out runs/c1m/color-s0 --color --init runs/ab/color-s0/last.pt

The published model (v5, train.sh): FastViT-MA36 with the D-FINE neck, the FastViT recipe in fp16, from ImageNet:

  python -m torch.distributed.run --nproc_per_node 2 train_c1m.py --out runs/v5/v5-s0 --arch fastvit_ma36 \
    --recipe apple --neck dfine --color --passed --fast --auto --temporal 4000

The frames are already in the narrow camera's calibrated model frame (medmodel), so geometric augmentation stays small
(the row of a line is its distance). A frame has at most one stop line, in the ego lane.
"""
import argparse, hashlib, json, math, os, random
import numpy as np, cv2, torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from model import Net, STRIDE, END_SCALE
from train import clip_segment, gaussian, focal, W, H, GW, GH
from cams import FrameCam

from paths import C1M as DATA
K = 1             # stop lines per frame at most
CAM = FrameCam()


class C1M(Dataset):
  def __init__(self, split, train, frac=1.0, passed=False, fast=False, auto=False, temporal=0, temporal_geo=True,
               temporal_bottom=None):
    items = json.load(open(f'{DATA}/items.json'))
    if passed:  # frames just past a labelled stop line, with none (pass_c1m.py); in both splits
      items.update(json.load(open(f'{DATA}/pass_items.json')))
    if fast:  # frames above 40 mph, with none (fast_c1m.py); in both splits
      items.update(json.load(open(f'{DATA}/fast_items.json')))
    if auto and split == 'train':  # approach frames labelled by the VLM and the model in agreement (auto_items.py)
      items.update(json.load(open(f'{DATA}/auto_items.json')))
    self.items = [(k, v) for k, v in sorted(items.items()) if v['split'] == split]
    if frac < 1.0:  # a fixed, nested share of the drives: every smaller share is inside every larger one
      sids = sorted({v['sid'] for _, v in self.items}, key=lambda s: hashlib.sha1(('frac' + s).encode()).hexdigest())
      keep = set(sids[:max(1, round(frac * len(sids)))])
      self.items = [(k, v) for k, v in self.items if v['sid'] in keep]
    self.train = train
    # temporal windows (temporal_c1m.py): `temporal` of their frames per epoch, drawn afresh for each sample
    self.extra = [(k, v) for k, v in sorted(json.load(open(f'{DATA}/temporal/items.json')).items())
                  if v['split'] == split] if temporal and split == 'train' else []
    self.n_extra = temporal if self.extra else 0
    # temporal_geo False: window frames teach only where a line is (heat map), not its exact ends, offset or mask,
    #   which carry the localizer's error; those come from the hand labels alone
    # temporal_bottom: a carried line whose middle is below this row (near the bumper, where carrying is least
    #   reliable) makes the window frame's line unknown
    self.temporal_geo, self.temporal_bottom = temporal_geo, temporal_bottom

  def __len__(self):
    return len(self.items) + getattr(self, 'n_extra', 0)  # subclasses that set items only have no extras

  def affine(self, cam):
    """2x stored image -> 512x256 network input."""
    a = np.diag([0.5, 0.5, 1.0])
    if not self.train:
      return a, False
    s, rot = random.uniform(0.9, 1.1), random.uniform(-2, 2)
    r = cv2.getRotationMatrix2D((cam.cx, cam.cy), rot, s)  # about the principal point: the horizon stays put
    t = np.array([[1, 0, random.uniform(-10, 10)], [0, 1, random.uniform(-6, 6)], [0, 0, 1]])
    a = t @ np.vstack([r, [0, 0, 1]]) @ a
    flip = random.random() < 0.5
    if flip:
      a = np.array([[-1, 0, W], [0, 1, 0], [0, 0, 1]]) @ a
    return a, flip

  def __getitem__(self, i):
    name, v = self.items[i] if i < len(self.items) else random.choice(self.extra)
    img = cv2.imread(f"{v.get('dir', DATA + '/yuv')}/{name}.png", cv2.IMREAD_UNCHANGED)
    a, _ = self.affine(CAM)
    out = cv2.warpAffine(img, a[:2], (W, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    pt = lambda x, y: (a[0, 0] * 2 * x + a[0, 1] * 2 * y + a[0, 2], a[1, 0] * 2 * x + a[1, 1] * 2 * y + a[1, 2])

    lines, line_ok = [], (0.0 if v.get('line_unknown') else 1.0)
    tmp = v.get('kind') == 'temporal'
    geo = 0.0 if tmp and not getattr(self, 'temporal_geo', True) else 1.0
    tb = getattr(self, 'temporal_bottom', None)
    if tmp and tb is not None and any((l[1] + l[3]) / 2 > tb for l in v['lines']):
      line_ok = 0.0
    for l in v['lines']:
      (ax, ay), (bx, by) = pt(l[0], l[1]), pt(l[2], l[3])
      c = clip_segment(ax, ay, bx, by)
      full = math.hypot(bx - ax, by - ay)
      if c is not None and math.hypot(c[2] - c[0], c[3] - c[1]) >= max(8, 0.3 * full):
        lines.append(c if c[0] <= c[2] else [c[2], c[3], c[0], c[1]])
      elif c is not None:
        line_ok = 0.0

    y = out[..., 0].astype(np.float32)
    uv = cv2.resize(out[..., 1:], (W // 2, H // 2), interpolation=cv2.INTER_AREA).astype(np.float32)
    if self.train:
      f = y / 255
      f = f ** random.uniform(0.6, 1.5)
      f = f * random.uniform(0.7, 1.3) + random.uniform(-0.15, 0.15)
      if random.random() < 0.3:
        f = cv2.GaussianBlur(f, (0, 0), random.uniform(0.3, 1.2))
      f = f + np.random.randn(*f.shape).astype(np.float32) * random.uniform(0, 0.025)
      y = np.clip(f * 255, 0, 255)
      uv = (uv - 128) * random.uniform(0.6, 1.4) + np.random.uniform(-4, 4, 2).astype(np.float32)
      uv = np.clip(uv + 128 + np.random.randn(*uv.shape).astype(np.float32) * random.uniform(0, 1.5), 0, 255)
    x = torch.from_numpy(y)[None]
    c = torch.from_numpy(np.ascontiguousarray(uv.transpose(2, 0, 1)))
    return x, c, targets(lines[:K], line_ok, geo), name


def targets(lines, line_ok, geo=1.0):
  t = {'lheat': np.zeros((GH, GW), np.float32), 'lmask': np.zeros((GH, GW), np.float32),
       'lreg': np.zeros((K, 6), np.float32), 'lidx': np.zeros(K, np.int64), 'lval': np.zeros(K, np.float32),
       'lok': np.float32(line_ok), 'lgeo': np.float32(geo)}
  m = np.zeros((GH * 4, GW * 4), np.uint8)
  for k, line in enumerate(lines):
    x1, y1, x2, y2 = [v / STRIDE for v in line]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ix, iy = min(GW - 1, int(cx)), min(GH - 1, int(cy))
    t['lheat'] = np.maximum(t['lheat'], gaussian(cx, cy, 2.0, 1.2))
    t['lheat'][iy, ix] = 1.0
    t['lidx'][k] = iy * GW + ix
    t['lreg'][k] = [cx - ix, cy - iy, (x1 - cx) / END_SCALE, (y1 - cy) / END_SCALE,
                    (x2 - cx) / END_SCALE, (y2 - cy) / END_SCALE]
    t['lval'][k] = 1
    cv2.line(m, (int(x1 * 4), int(y1 * 4)), (int(x2 * 4), int(y2 * 4)), 1, 6)
  t['lmask'] = cv2.resize(m.astype(np.float32), (GW, GH), interpolation=cv2.INTER_AREA)
  return {k: torch.as_tensor(v) for k, v in t.items()}


def gather(maps, idx):
  """(N, C, GH, GW) maps, (N, K) flat indices -> (N, K, C)."""
  n, c = maps.shape[:2]
  return maps.flatten(2).gather(2, idx[:, None, :].expand(n, c, idx.shape[1])).transpose(1, 2)


def losses(line, t):
  lw = t['lval'] * (t['lok'] * t['lgeo'])[:, None]         # (N, K)
  lr = gather(line[:, 1:7].float(), t['lidx'])             # (N, K, 6)
  l1 = lambda a, b: F.l1_loss(a, b, reduction='none').sum(-1)
  return {
    'lheat': focal(line[:, 0], t['lheat'], t['lok']),
    'loff': (l1(lr[..., :2], t['lreg'][..., :2]) * lw).sum() / lw.sum().clamp(min=1),
    'lend': 4 * (l1(lr[..., 2:], t['lreg'][..., 2:]) * lw).sum() / lw.sum().clamp(min=1),
    'lmask': 5 * (F.binary_cross_entropy_with_logits(line[:, 7].float(), t['lmask'], reduction='none')
                  .flatten(1).mean(1) * t['lok'] * t['lgeo']).mean(),
  }


def decode_all(line):
  """Every line peak (score, 4 ends), in 512x256 pixels, per image."""
  lh = torch.sigmoid(line[:, 0:1].float())
  peaks = lh * (F.max_pool2d(lh, 5, 1, 2) == lh)
  out = []
  for n in range(line.shape[0]):
    s, idx = peaks[n, 0].flatten().topk(K)
    ls = []
    for sc, i in zip(s.tolist(), idx.tolist()):
      y, x = divmod(i, GW)
      cx, cy = x + line[n, 1, y, x].item(), y + line[n, 2, y, x].item()
      e = (line[n, 3:7, y, x].float() * END_SCALE).tolist()
      ls.append([sc, (cx + e[0]) * STRIDE, (cy + e[1]) * STRIDE, (cx + e[2]) * STRIDE, (cy + e[3]) * STRIDE])
    out.append({'lines': ls})
  return out


def dist_at_centre(l, cam):
  """Flat-road distance of a line at the image's centre column (or its nearest end if it doesn't reach it)."""
  return cam.line_dist(l, at=cam.cx)


def evaluate(model, ds, device):
  model.eval()
  recs = []
  dl = DataLoader(ds, 32, num_workers=8)
  with torch.no_grad():
    for x, c, t, names in dl:
      line = model(x.to(device), c.to(device))
      for n, d in enumerate(decode_all(line.cpu())):
        v = dict(ds.items)[names[n]]
        recs.append({'name': names[n], 'kind': v['kind'],
                     'gl': [list(map(float, l)) for l in v['lines']], **d})
  model.train()
  return recs


def score(recs, subset=lambda r: True):
  res = {}
  rs = [r for r in recs if subset(r)]
  for th in (0.3, 0.5):
    tp = npred = ngt = 0; errs, derr = [], []
    for r in rs:
      preds = [p for p in r['lines'] if p[0] >= th]
      npred += len(preds); ngt += len(r['gl'])
      used = set()
      for g in r['gl']:
        best = None
        for j, p in enumerate(preds):
          if j in used: continue
          e = min(max(math.hypot(p[1] - g[0], p[2] - g[1]), math.hypot(p[3] - g[2], p[4] - g[3])),
                  max(math.hypot(p[3] - g[0], p[4] - g[1]), math.hypot(p[1] - g[2], p[2] - g[3])))
          if e <= 20 and (best is None or e < best[0]): best = (e, j)
        if best:
          used.add(best[1]); tp += 1; errs.append(best[0])
          cam = CAM
          dg, dp = dist_at_centre(g, cam), dist_at_centre(preds[best[1]][1:], cam)
          derr.append((abs(dp - dg), abs(dp - dg) / dg))
    pr, rc = tp / max(1, npred), tp / max(1, ngt)
    res[f'line@{th}'] = {'P': round(pr, 4), 'R': round(rc, 4), 'F1': round(2 * pr * rc / max(1e-9, pr + rc), 4),
                         'end_err_med': round(float(np.median(errs)), 2) if errs else None,
                         'dist_err_med_m': round(float(np.median([d[0] for d in derr])), 2) if derr else None,
                         'dist_err_med_rel': round(float(np.median([d[1] for d in derr])), 3) if derr else None,
                         'n_gt': ngt}
  return res


DC_SCALE = 100.0  # FPN channel means are pinned towards 0 in units of this (see main)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', required=True)
  ap.add_argument('--epochs', type=int, default=150)
  ap.add_argument('--bs', type=int, default=32)
  ap.add_argument('--lr', type=float, default=1e-3)
  ap.add_argument('--color', action='store_true')
  ap.add_argument('--init', default=None, help='a checkpoint to start from (else ImageNet)')
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--arch', default='resnet18', help='resnet18, resnet34, or a timm FastViT (e.g. fastvit_ma36)')
  ap.add_argument('--neck', default='fpn', choices=['fpn', 'dfine'], help='dfine: D-FINE hybrid encoder (neck.py)')
  ap.add_argument('--recipe', default='ours', choices=['ours', 'apple'], help='apple: the FastViT recipe (recipe.py)')
  ap.add_argument('--frac', type=float, default=1.0, help='share of the training drives to use')
  ap.add_argument('--passed', action='store_true', help='add the frames just past a labelled stop line, as no line')
  ap.add_argument('--fast', action='store_true', help='add frames above 40 mph, as no line')
  ap.add_argument('--auto', action='store_true', help='add the automatically labelled approach frames')
  ap.add_argument('--temporal', type=int, default=0, help='frames of the temporal windows to draw per epoch')
  ap.add_argument('--temporal-presence', action='store_true', help='window frames: heat map only, no line geometry')
  ap.add_argument('--temporal-bottom', type=float, default=None, help='window frames: line unknown below this row')
  ap.add_argument('--qknorm', action='store_true', help='FastViT: QK-normalized attention (bounded logits, fp16-safe)')
  ap.add_argument('--act-penalty', type=float, default=0.0,
                  help='penalize trunk and neck activations above this magnitude (0: off), to stay in fp16 range')
  ap.add_argument('--act-weight', type=float, default=1.0)
  ap.add_argument('--fp16', action='store_true', help='fp16 autocast (with a gradient scaler) instead of bfloat16')
  args = ap.parse_args()
  apple = args.recipe == 'apple'
  import recipe
  if apple and args.arch.startswith('fastvit'):
    args.qknorm, args.act_penalty = True, recipe.ACT_PENALTY
  if apple:
    args.fp16 = True
  dist.init_process_group('nccl')
  rank = dist.get_rank()
  device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
  torch.cuda.set_device(device)
  s0 = args.seed * 1000 + rank
  torch.manual_seed(s0); random.seed(s0); np.random.seed(s0)
  os.makedirs(args.out, exist_ok=True)

  train = C1M('train', True, args.frac, args.passed, args.fast, args.auto, args.temporal,
              temporal_geo=not args.temporal_presence, temporal_bottom=args.temporal_bottom)
  if rank == 0: print('training frames', len(train), flush=True)
  sampler = DistributedSampler(train, shuffle=True)
  dl = DataLoader(train, args.bs, sampler=sampler, num_workers=8, drop_last=True, pin_memory=True,
                  persistent_workers=True,
                  worker_init_fn=lambda w: (random.seed(s0 * 100 + w), np.random.seed(s0 * 100 + w)))
  test = C1M('test', False, passed=args.passed, fast=args.fast)
  torch.manual_seed(args.seed)
  model = Net(pretrained=args.init is None, color=args.color, arch=args.arch, qknorm=args.qknorm,
              drop_path=recipe.DROP_PATH.get(args.arch, 0.0) if apple else 0.0, neck=args.neck)
  if args.init:
    sd = {k.replace('module.', ''): v for k, v in torch.load(args.init, map_location='cpu').items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.endswith('log_tau') for k in missing), (missing, unexpected)
    if missing and rank == 0: print('new parameters (QK-norm temperatures):', len(missing), flush=True)
  acts = []  # (penalty, max) per hooked output, computed in the hook so no activation is kept twice
  if args.act_penalty > 0:  # every conv / linear output and every block output in stages 3-5 (stage 2 stays small)
    def act_hook(mod, inp, o):
      if mod.training and isinstance(o, torch.Tensor):
        a = o.abs()
        acts.append((torch.relu(a / args.act_penalty - 1).mean(), a.detach().max()))
    for st in (model.c3, model.c4, model.c5):
      for mod in st.modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)) or type(mod).__name__.endswith('Block'):
          mod.register_forward_hook(act_hook)
    # and the neck. The FPN's smoothing conv sums bias-carrying laterals and feeds a BatchNorm, which hides a
    # per-channel offset from the loss, so it drifts until fp16 overflows; a threshold penalty averaged over pixels
    # barely sees it, so each FPN channel's spatial mean is also pinned towards 0 (a direction the loss cannot see).
    def dc_hook(mod, inp, o):
      if mod.training and isinstance(o, torch.Tensor):
        dc = o.float().mean((2, 3))
        acts.append((((dc / DC_SCALE) ** 2).mean(), o.detach().abs().max()))
    if model.dfine:  # D-FINE neck: every conv input is normalized, so only the threshold penalty (convs, linears)
      for mod in model.neck.modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
          mod.register_forward_hook(act_hook)
    else:
      for mod in [*model.lat, model.smooth[0]]:
        mod.register_forward_hook(act_hook)
        mod.register_forward_hook(dc_hook)
  model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.to(device))
  ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
  if apple:
    opt = recipe.optimizer(model, args.lr); sched = recipe.cosine(opt, args.epochs, len(dl))
    ema = recipe.EMA(model, recipe.ema_decay(len(dl)))
  else:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl), pct_start=0.05)
  log = open(f'{args.out}/log.jsonl', 'a') if rank == 0 else None
  scaler = torch.amp.GradScaler('cuda') if args.fp16 else None
  subsets = {'near': lambda r: r['kind'] == 'near', 'far': lambda r: r['kind'] == 'far'}
  # NaN guard: every 50 steps check the weights; keep the last finite state and the losses since, and stop at the
  # first non-finite weight with a dump (nan_debug.pt) to replay from
  step, hist, good = 0, [], None
  for ep in range(args.epochs):
    sampler.set_epoch(ep)
    tot, nonfinite = {}, 0
    for x, c, t, _ in dl:
      x, c = x.to(device, non_blocking=True), c.to(device, non_blocking=True)
      t = {k: v.to(device, non_blocking=True) for k, v in t.items()}
      acts.clear()
      with torch.autocast('cuda', dtype=torch.float16 if args.fp16 else torch.bfloat16):
        line = ddp(x, c)
      parts = losses(line, t)
      if acts:
        parts['act'] = args.act_weight * sum(p.float() for p, _ in acts)
        act_max = torch.stack([mx.float() for _, mx in acts]).max().item()
      loss = sum(parts.values())
      opt.zero_grad(set_to_none=True)
      if apple:  # the FastViT recipe: no gradient clipping
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); ema.update(model)
      elif args.fp16:
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(opt); scaler.update(); sched.step()
      else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step(); sched.step()
      if acts: parts['act_max'] = torch.tensor(act_max)  # logged: the epoch's mean of the per-step largest activation
      lv = {k: v.item() for k, v in parts.items()}
      nonfinite += not all(math.isfinite(v) for v in lv.values())
      hist.append((step, lv, scaler.get_scale() if scaler else None)); hist = hist[-200:]
      step += 1
      if step % 50 == 0:
        if all(torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()):
          good = (step, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        else:
          if rank == 0:
            torch.save({'step': step, 'epoch': ep, 'hist': hist, 'good': good, 'batch': (x.cpu(), c.cpu())},
                       f'{args.out}/nan_debug.pt')
            print(json.dumps({'nan_at_step': step, 'epoch': ep, 'last_good_step': good and good[0],
                              'recent': hist[-60:]}), flush=True)
          raise SystemExit(f'non-finite weights by step {step} (epoch {ep})')
      for k, v in lv.items(): tot[k] = tot.get(k, 0) + v / len(dl)
    if rank == 0 and (ep % 25 == 24 or ep == args.epochs - 1):
      recs = evaluate(ema.m if apple else model, test, device)
      rec = {'epoch': ep + 1, 'loss': {k: round(v, 4) for k, v in tot.items()}, 'nonfinite_steps': nonfinite,
             'grad_scale': scaler.get_scale() if scaler else None, **score(recs)}
      for name, f in subsets.items():
        rec[name] = score(recs, f)['line@0.3']
      print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
      torch.save((ema.m if apple else model).state_dict(), f'{args.out}/last.pt')  # apple: the EMA weights
      if apple: torch.save(model.state_dict(), f'{args.out}/raw.pt')
      if ep == args.epochs - 1:
        json.dump(recs, open(f'{args.out}/test_preds.json', 'w'))
  dist.destroy_process_group()


if __name__ == '__main__':
  main()
