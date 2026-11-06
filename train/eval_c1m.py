"""Re-evaluate saved comma1M runs on the current test set (c1m/items.json), e.g. after new labels grew it.

  python eval_c1m.py runs/curve2 runs/curve2-on1392 [passed] [fast]  -> <out>/<run>/test_preds.json for every <runs>/<run>/last.pt
(with passed, the test set takes in pass_c1m.py's frames)
"""
import glob, json, os, sys
import torch
from model import Net
from train_c1m import C1M, evaluate

src, dst = sys.argv[1], sys.argv[2]
device = torch.device('cuda', 0)
test = C1M('test', False, passed='passed' in sys.argv[3:], fast='fast' in sys.argv[3:])
for ck in sorted(glob.glob(f'{src}/*/last.pt')):
  run = os.path.basename(os.path.dirname(ck))
  sd = torch.load(ck, map_location='cpu')
  model = Net(pretrained=False, color='uv.weight' in sd).to(device)
  model.load_state_dict(sd)
  os.makedirs(f'{dst}/{run}', exist_ok=True)
  json.dump(evaluate(model, test, device), open(f'{dst}/{run}/test_preds.json', 'w'))
  print(run, flush=True)
