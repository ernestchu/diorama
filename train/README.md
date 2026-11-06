# Training the stop-line model

The model the iPhone runs: a ResNet-18 FPN CenterNet (`model.py`, 11.5 M parameters) on the narrow camera's frame
as the comma sends it, openpilot's medmodel frame at 512x256 (Y, plus U and V at 256x128). It finds at most one stop
line, in the ego lane, as two end points, and the lead vehicle as a box. `export.py` turns a checkpoint into the
Core ML model the app loads and its ONNX twin.

No data, labels or checkpoints are in the repo. `paths.py` says where they go; set `STOPLINE_WORK` (and `COMMA1M`,
comma1M's `data/` folder) or keep the defaults under `./work`.

    pip install -r requirements.txt     # Python 3.10+, CUDA; two GPUs for the training scripts
    export STOPLINE_WORK=/path/to/work COMMA1M=/path/to/comma1M/data

## Data

- **comma1M**, with each segment's own calibration (`localizer.safetensors`) for openpilot's warp (`c1m.py`).
  Only devices with both cameras' video (no comma twos).
- **Pretraining**: the SLSDNet stop-line dataset, unzipped into `$SLSD_RAW`.
- **Leads** come from a detector, D-FINE (`teacher.py`): the nearest vehicle on the path the car went on to drive
  (`build_c1m.py`).

Your own drives are for trying the model (`web/`), not for training.

## Labels

1. `scan_stops.py $STOPLINE_WORK/meta/stops.json` finds where the car came to a stop in each segment.
   `classify_stops.py` then marks each stop first in line or queued behind another car.
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

## Train

    sh pretrain.sh      # prep.py and prep_color.py first: SLSDNet, three seeds
    sh train.sh         # builds the frames, then comma1M from the pretrained weights, three seeds

`train.sh` builds three sets of frames:
- `build_c1m.py`: the labelled frames, re-rendered from the camera's own YUV, with the lead chosen.
- `pass_c1m.py`: frames 2 to 15 m past each labelled line, as no line.
- `fast_c1m.py`: frames above 40 mph, as no line.

It then trains with `train_c1m.py --passed --fast`.

To read the results:
- `compare_c1m.py runs/<name>` gives the mean and spread over seeds.
- `rowscore_c1m.py` scores lines by their row, which is what sets the distance.
- `eval_c1m.py` re-scores saved checkpoints on the current test set.

The split is by segment: about one segment in five is test, never trained on.

## Export

    python export.py $STOPLINE_WORK/runs/fast/fast-s0/last.pt out/

This writes two files:
- `StoplinePilot.mlpackage`: ML program, fp16. Y is a grayscale image, UV an fp16 tensor, and the heat maps go through
  sigmoid in the graph.
- `stopline_pilot.onnx`: the same model as ONNX.

Both go in the repo's `models/`. The app and `sim/stopline_parity.py` read them from there, and
`python -m dioramaweb run --stopline` takes the ONNX.
