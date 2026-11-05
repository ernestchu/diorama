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
| `sim` | `commasim`, a Mac playing the comma with a recorded drive |
| `models` | the stop-line model, fetched, not committed |

## Set up

The stop-line model (Core ML), `StoplinePilot.mlpackage`, unzipped into
`models/`. It is not in the repo; ask for a copy.

Then the app:

    cd ios && xcodegen && open Diorama.xcodeproj

A build without the model draws the drive with no stop lines.

## Develop on a Mac, without a comma

    cd sim && uv sync
    uv run python -m commasim <drive's segments> --model <driving model .onnx>

Run the app in the iOS Simulator: it dials `127.0.0.1:5599`, where commasim
listens. An iPhone dials the comma on its cable (`192.168.60.1`); set the
Mac's address in the app's Settings to have it watch commasim instead.

## Tests

    cd DioramaKit && swift test

`stopLineMatchesTheReference` runs the Core ML model against its ONNX twin's
decode on a drive's frames, once `sim/stopline_parity.py` has written them
(it needs `models/stopline_pilot.onnx` beside the Core ML model).

## Credits

The link's framing, and `sim/commasim/vendor`, come from
[jetlink](https://github.com/zoompilot/jetlink) by Zeph Leggett (MIT).
