# Training the stop-line model

The model the iPhone runs (v5): a CenterNet-style stop-line detector on the narrow camera's frame as the comma sends
it, openpilot's medmodel frame at 512x256 (Y, plus U and V at 256x128). It finds at most one stop line, in the ego
lane, as two end points. The trunk is FastViT-MA36 with QK-normalized attention, the neck D-FINE's hybrid encoder
(`neck.py`) down to stride 4 (`model.py`; 43.6 M parameters, 18.1 GMACs), trained with the FastViT authors' recipe in
fp16 (`recipe.py`). `model.py` also builds the earlier ResNet-18/34 FPN models. `export.py` turns a checkpoint into
the Core ML model the app loads and its ONNX twin, each with one output, `line`.

No data, labels or checkpoints are in the repo, apart from the eight worked examples the VLM labeller is shown
(`vlm/shots.json`: frame ids, their lines and why). `paths.py` says where they go; set `STOPLINE_WORK` (and `COMMA1M`,
comma1M's `data/` folder) or keep the defaults under `./work`.

    pip install -r requirements.txt     # Python 3.10+, CUDA; two GPUs for the training scripts
    export STOPLINE_WORK=/path/to/work COMMA1M=/path/to/comma1M/data

## Data

- **comma1M**, with each segment's own calibration (`localizer.safetensors`) for openpilot's warp (`c1m.py`).
  Only devices with both cameras' video (no comma twos).
- **Pretraining**: the SLSDNet stop-line dataset, unzipped into `$SLSD_RAW`.

Your own drives are for trying the model (`web/`), not for training.

## Labels

1. `scan_stops.py $STOPLINE_WORK/meta/stops.json` finds where the car came to a stop in each segment.
   `classify_stops.py` then marks each stop first in line or queued behind another car, with a vehicle detector,
   D-FINE (`vehicles.py`).
2. `export_frames.py` samples frames near and far from those stops, plus random driving, into the labelling site's
   `frames/`, with a record of each in `meta/batch*.json`.
3. Label them on the site (`site/`, hosted separately as a static page). A label is one stop line in the ego lane, or
   none.
4. Download the session's JSON, then run:
   - `labels.py merge FILE` adds it to `labels/store.json`.
   - `labels.py manifest` takes the labelled frames off the site.

`ego_c1m.py` runs the driving model (with `sim/commasim`, on the GPU) at labelled frames, and `egotrim.py` cuts a line
to the ego lane between its lane lines. That is how the earlier all-lane labels were turned into ego-lane suggestions
to check on the site.

### Labels from a vision-language model

The approach frames of the published model are labelled by Claude Haiku given the guideline, and checked by a person
only where it and a model disagree:
1. `mine_auto.py` renders frames 9 to 38 m before every stop in the training split.
2. `vlm_label.py submit`, then `vlm_label.py fetch` once the batches end: Haiku labels them through the Message
   Batches API with `vlm/prompt.md` and the eight worked examples of `vlm/shots.json` (hand-labelled training frames,
   by id). It needs `ANTHROPIC_API_KEY`; the paper's 3467 frames cost $0.37.
3. `auto_split.py CHECKPOINT` runs a model trained without these frames (e.g. `train_c1m.py --passed --fast`) on them and
   sends a frame to review when the model is unsure, when the two disagree on whether there is a line, or when an end
   differs by more than 20 px.
4. `queue_auto.py`, then `labels.py manifest`: the frames to review go to the site, prefilled with Haiku's line.
   Label them and `labels.py merge` the session, as for any other frames; `build_c1m.py` then takes them as hand labels.
5. `auto_items.py` writes the frames on which the two agree, as training frames (`auto_items.json`).

## Train

    sh train.sh         # builds the frames, then trains the published model (seed 0; SEEDS="0 1 2" for more)

`train.sh` builds four sets of frames, then trains with `train_c1m.py --arch fastvit_ma36 --recipe apple --neck dfine
--color --passed --fast --auto --temporal 4000`, from ImageNet:
- `build_c1m.py`: the labelled frames, re-rendered from the camera's own YUV.
- `pass_c1m.py`: frames 2 to 15 m past each labelled line, as no line.
- `fast_c1m.py`: frames above 40 mph, as no line.
- `temporal_c1m.py`: the frames within 1 s of each labelled frame (and of each test frame above 40 mph), each with the
  label carried into it with the car's motion (`prop.py`); training draws 4000 of them an epoch.
It also needs `auto_items.json`, from the labelling steps above.

The FastViT recipe trains in fp16 so that the network stays in fp16's range, as it runs on the Neural Engine: a loss
scaler, an EMA of the weights (what is evaluated and saved), QK-normalized attention, a penalty on activations above
8000, and a check every 50 steps that stops with a dump (`nan_debug.pt`) at the first non-finite weight. Each run
takes about 5 hours on two RTX 4090s.

The SLSDNet stage (`pretrain.sh`, `train.py`; `prep.py` and `prep_color.py` first) gave the earlier ResNet models their
starting point; the published model does as well without it, but `train_c1m.py --init` still takes such a checkpoint.

To read the results:
- `compare_c1m.py runs/<name>` gives the mean and spread over seeds.
- `rowscore_c1m.py` scores lines by their row, which is what sets the distance.
- `eval_c1m.py` re-scores saved checkpoints on the current test set.

The split is by segment: about one segment in five is test, never trained on.

## Export

    python export.py $STOPLINE_WORK/runs/v5/v5-s0/last.pt out/

This writes two files:
- `StoplinePilot.mlpackage`: ML program, fp16. Y is a grayscale image, UV an fp16 tensor; the one output, `line`
  (1, 8, 64, 128), holds the heat map (sigmoid), the centre offset (2), the end points (4) and the mask (sigmoid).
- `stopline_pilot.onnx`: the same model as ONNX.

## Publish

    python publish.py $STOPLINE_WORK/runs/v5/v5-s0/last.pt --note "what changed"

exports the checkpoint and uploads both files, zipped Core ML, to the Hugging Face model repo
[ernestchu/diorama](https://huggingface.co/ernestchu/diorama) with a `manifest.json` (version, note, sha256 of each
file), tagged `v<version>`. The app fetches the latest from there; models are not bundled with it. It needs a
`hf auth login` with write access to the repo.

For development, put the two files in the repo's `models/` (`hf download ernestchu/diorama --local-dir models`, then
unzip the package). `sim/stopline_parity.py` reads them there, and `python -m dioramaweb run --stopline` takes the
ONNX.
