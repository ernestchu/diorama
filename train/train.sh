#!/bin/sh
# The published model (v5): FastViT-MA36 with the D-FINE neck and QK-normalized attention, the FastViT authors'
# recipe in fp16 (recipe.py), straight from ImageNet (no SLSDNet stage). Training frames: the labelled comma1M frames,
# the frames just past each labelled line and above 40 mph (both as no line), the VLM-labelled approach frames on which
# the VLM and a model agree (auto_items.json: label those first, see README), and 4000 frames an epoch from the
# temporal windows around each label. Builds the frames first. SEEDS (default 0) picks the seeds; NAME (default v5)
# names the runs. Two GPUs; run from anywhere with STOPLINE_WORK and COMMA1M set.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
: "${STOPLINE_WORK:?set STOPLINE_WORK}" "${COMMA1M:?set COMMA1M}"
export HF_HOME=${HF_HOME:-$STOPLINE_WORK/hf} TORCH_HOME=${TORCH_HOME:-$STOPLINE_WORK/hf/torch}
PY=${PY:-python}
NAME=${NAME:-v5}
C1M=$(cd "$HERE" && $PY -c 'import paths; print(paths.C1M)')
[ -f "$C1M/auto_items.json" ] || { echo "no $C1M/auto_items.json: label the approach frames first (README)"; exit 1; }
$PY "$HERE/build_c1m.py"
$PY "$HERE/pass_c1m.py"
$PY "$HERE/fast_c1m.py"
$PY "$HERE/temporal_c1m.py"
mkdir -p "$STOPLINE_WORK/runs/$NAME"
for seed in ${SEEDS:-0}; do
  $PY -m torch.distributed.run --nproc_per_node 2 "$HERE/train_c1m.py" --out "$STOPLINE_WORK/runs/$NAME/$NAME-s$seed" \
    --seed $seed --arch fastvit_ma36 --recipe apple --neck dfine --color --passed --fast --auto --temporal 4000 \
    > "$STOPLINE_WORK/runs/$NAME/$NAME-s$seed.log" 2>&1 || echo "FAILED s$seed"
done
$PY "$HERE/compare_c1m.py" "$STOPLINE_WORK/runs/$NAME"
