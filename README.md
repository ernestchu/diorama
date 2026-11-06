# Diorama

The drive as a live 3-D miniature on an iPhone plugged into a comma running
zoompilot. The comma runs the driving model and streams its narrow camera
frame and the model's outputs over the USB cable; the iPhone runs only the
stop-line model, and draws the plan, lane lines, road edges, lead and stop line.

How it fits together, and the link it speaks: [DESIGN.md](DESIGN.md).

| Folder | What |
| --- | --- |
| `DioramaKit` | Swift package: the scene (`DioramaCore`), the link (`DioramaLink`), Metal drawing, the stop-line model and the feed (`DioramaUI`) |
| `ios` | the iPhone app (xcodegen) |
| `diorama` | `dioramad`, the comma's side: zoompilot pins this repo and runs it |
| `sim` | `commasim`, a Mac playing the comma with a recorded drive |
| `train` | the stop-line model's training: labelling, data, training, export ([train/README.md](train/README.md)) |
| `web` | the drive in a Linux browser, for development ([web/README.md](web/README.md)) |
| `models` | the stop-line model for tests and the Linux path, downloaded from Hugging Face, not committed |

## Set up

The stop-line model is published on Hugging Face, [ernestchu/diorama](https://huggingface.co/ernestchu/diorama)
(`train/publish.py`), with a `manifest.json` giving its version. For the tests and the Linux path:

    hf download ernestchu/diorama --local-dir models && unzip -o models/StoplinePilot.mlpackage.zip -d models

Then the app:

    cd ios && xcodegen && open Diorama.xcodeproj

A build without the model draws the drive with no stop lines.
To make the model, see [train/README.md](train/README.md); `export.py` writes both files.

## Develop on a Mac, without a comma

    cd sim && uv sync
    uv run python -m commasim <drive's segments> --model <driving model .onnx>

`commasim.as_dioramad` plays it through dioramad's own code instead (its warp,
its modelV2 encoding, its link), less openpilot:

    uv run python -m commasim.as_dioramad <drive's segments> --model <driving model .onnx>

Run the app in the iOS Simulator: it dials `127.0.0.1:5600`, where commasim
listens. An iPhone dials the comma on its cable (`192.168.60.1`); set the
Mac's address in the app's Settings to have it watch commasim instead.

## Develop on Linux

Train the stop-line model with `train/`. Try it on a recorded drive in the browser with `web/`: the same driving
model, stop-line model and tracker, drawn by the page Diorama started from. No data, labels or checkpoints are in the
repo.

## Tests

    cd DioramaKit && swift test
    cd sim && uv run pytest ../tests

`stopLineMatchesTheReference` runs the Core ML model against its ONNX twin's
decode on a drive's frames, once `sim/stopline_parity.py` has written them
(it needs `models/stopline_pilot.onnx` beside the Core ML model).

## Credits

The link's framing, and `sim/commasim/vendor`, come from
[jetlink](https://github.com/zoompilot/jetlink) by Zeph Leggett (MIT).
