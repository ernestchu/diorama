#!/bin/sh
# The current recipe: the labelled comma1M frames, plus the frames just past each labelled line and frames above
# 40 mph, both as no line, and slow frames with near leads for the lead head (mine_near.py); from the pretrained
# checkpoints (pretrain.sh), three seeds. Builds the data first.
# Two GPUs; run from anywhere with STOPLINE_WORK and COMMA1M set. NAME (default fast) names the runs.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
: "${STOPLINE_WORK:?set STOPLINE_WORK}" "${COMMA1M:?set COMMA1M}"
export HF_HOME=${HF_HOME:-$STOPLINE_WORK/hf} TORCH_HOME=${TORCH_HOME:-$STOPLINE_WORK/hf/torch}
PY=${PY:-python}
NAME=${NAME:-near}
$PY "$HERE/build_c1m.py"
$PY "$HERE/pass_c1m.py"
$PY "$HERE/fast_c1m.py"
$PY "$HERE/mine_near.py"
mkdir -p "$STOPLINE_WORK/runs/$NAME"
for seed in 0 1 2; do
  $PY -m torch.distributed.run --nproc_per_node 2 "$HERE/train_c1m.py" --out "$STOPLINE_WORK/runs/$NAME/$NAME-s$seed" --seed $seed \
    --color --init "$STOPLINE_WORK/runs/ab/color-s$seed/last.pt" --passed --fast --near \
    > "$STOPLINE_WORK/runs/$NAME/$NAME-s$seed.log" 2>&1 || echo "FAILED s$seed"
done
$PY "$HERE/compare_c1m.py" "$STOPLINE_WORK/runs/$NAME"
$PY "$HERE/eval_near.py" "$STOPLINE_WORK/runs/$NAME"/$NAME-s*
