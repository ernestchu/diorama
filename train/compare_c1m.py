"""Summarise comma1M runs: the last evaluation of each seed, mean and spread per configuration.

  python compare_c1m.py runs/fast [runs/ego ...]    (each holds <config>-s<seed>/log.jsonl)
"""
import glob, json, os, re, sys
import numpy as np

rows = {}
for f in sorted(f for d in sys.argv[1:] for f in glob.glob(f'{d}/*-s[0-9]/log.jsonl')):
  cfg = re.match(r'(.+)-s\d+$', os.path.basename(os.path.dirname(f))).group(1)
  rows.setdefault(cfg, []).append([json.loads(l) for l in open(f)][-1])

keys = [('line@0.3', 'P'), ('line@0.3', 'R'), ('line@0.3', 'F1'), ('line@0.5', 'F1'), ('line@0.3', 'end_err_med'),
        ('line@0.3', 'dist_err_med_m'), ('line@0.3', 'dist_err_med_rel'), ('near', 'F1'), ('far', 'F1'),
        ('lead@0.3', 'F1'), ('lead@0.5', 'F1'), ('lead@0.3', 'iou_mean')]
print(f'{"metric":28s}' + ''.join(f'{c:>24s}' for c in rows))
for k, f in keys:
  cells = []
  for rs in rows.values():
    v = np.array([r[k][f] if r[k][f] is not None else np.nan for r in rs], float)
    cells.append(f'{np.nanmean(v):.3f} ± {np.nanstd(v):.3f} (n={len(v)})')
  print(f'{k + " " + f:28s}' + ''.join(f'{c:>24s}' for c in cells))
