#!/bin/sh
# Pretraining on the SLSDNet stop-line dataset (Y + UV), three seeds: the starting points for train.sh.
# prep.py and prep_color.py first. Two GPUs; run from anywhere with STOPLINE_WORK set (runs go to $STOPLINE_WORK/runs).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
: "${STOPLINE_WORK:?set STOPLINE_WORK}"
export HF_HOME=${HF_HOME:-$STOPLINE_WORK/hf} TORCH_HOME=${TORCH_HOME:-$STOPLINE_WORK/hf/torch}
PY=${PY:-python}
mkdir -p "$STOPLINE_WORK/runs/ab"
for seed in 0 1 2; do
  $PY -m torch.distributed.run --nproc_per_node 2 "$HERE/train.py" --out "$STOPLINE_WORK/runs/ab/color-s$seed" --seed $seed --color \
    > "$STOPLINE_WORK/runs/ab/color-s$seed.log" 2>&1 || echo "FAILED s$seed"
done
