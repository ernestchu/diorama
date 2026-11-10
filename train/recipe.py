"""The FastViT authors' ImageNet recipe (paper Table 15, apple/ml-fastvit train.py), adapted to our two stages:
fp16 mixed precision with a loss scaler and no gradient clipping, AdamW with weight decay 0.05 (not on biases, norm
parameters, layer scales or temperatures), a cosine schedule with 5 warm-up epochs down to 1% of the peak, an EMA of
the weights (0.9995) that is what gets evaluated and saved, and stochastic depth (0.35 for MA36).
Our adaptations: the pretrained trunk trains at TRUNK_LR_MULT of the peak (our batch is 64, theirs 1024), and two fp16
guards: QK-normalized attention and a penalty on trunk activations above ACT_PENALTY.
"""
import copy, math
import torch

WD, WARMUP, MIN_LR, EMA_DECAY, TRUNK_LR_MULT, ACT_PENALTY = 0.05, 5, 0.01, 0.9995, 0.1, 8000.0
DROP_PATH = {'fastvit_ma36': 0.35, 'fastvit_sa36': 0.2, 'fastvit_sa24': 0.1, 'fastvit_sa12': 0.1}


def param_groups(model, lr):
  trunk = {id(p) for name in ('conv1', 'stem', 'c2', 'c3', 'c4', 'c5', 'trunk') if hasattr(model, name)
           for p in getattr(model, name).parameters()}
  groups = {}
  for n, p in model.named_parameters():
    if not p.requires_grad:
      continue
    decay = p.ndim > 1 and not n.endswith('log_tau')
    key = (id(p) in trunk, decay)
    groups.setdefault(key, []).append(p)
  return [{'params': ps, 'lr': lr * (TRUNK_LR_MULT if t else 1.0), 'weight_decay': WD if d else 0.0, 'base_lr': lr * (TRUNK_LR_MULT if t else 1.0)}
          for (t, d), ps in groups.items()]


def optimizer(model, lr):
  return torch.optim.AdamW(param_groups(model, lr), lr=lr)


def cosine(opt, epochs, steps_per_epoch):
  total, warm = epochs * steps_per_epoch, WARMUP * steps_per_epoch
  def f(step):
    if step < warm:
      return (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    return MIN_LR + (1 - MIN_LR) * 0.5 * (1 + math.cos(math.pi * t))
  return torch.optim.lr_scheduler.LambdaLR(opt, f)


def ema_decay(steps_per_epoch):
  """Apple's 0.9995 averages over ~2000 steps, a sliver of their 375k (1250 steps an epoch); our epochs are 25 (SLSDNet)
  or ~220 (comma1M) steps, so average over about one epoch instead, within [0.99, 0.9995]."""
  return min(EMA_DECAY, max(0.99, 1 - 1 / steps_per_epoch))


class EMA:
  def __init__(self, model, decay=EMA_DECAY):
    self.m = copy.deepcopy(model).eval()
    for p in self.m.parameters():
      p.requires_grad_(False)
    self.decay = decay

  @torch.no_grad()
  def update(self, model):
    for e, p in zip(self.m.state_dict().values(), model.state_dict().values()):
      if e.dtype.is_floating_point:
        e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
      else:
        e.copy_(p)


def act_guard(model, threshold=ACT_PENALTY):
  """Hooks on trunk stages 3-5 and the FPN (every conv/linear and block output): returns the list they fill with
  (penalty, max) per output during training steps."""
  acts = []
  def hook(mod, inp, o):
    if mod.training and isinstance(o, torch.Tensor):
      a = o.abs()
      acts.append((torch.relu(a / threshold - 1).mean(), a.detach().max()))
  for st in (model.c3, model.c4, model.c5):
    for mod in st.modules():
      if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)) or type(mod).__name__.endswith('Block'):
        mod.register_forward_hook(hook)
  for mod in ([m for m in model.neck.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))] if model.dfine
              else [*model.lat, model.smooth[0]]):  # the neck (see train_c1m.py)
    mod.register_forward_hook(hook)
  return acts
