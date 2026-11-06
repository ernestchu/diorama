# Diorama in a Linux browser

For development: a recorded drive run through the driving model and the stop-line model, then played in the Diorama
page beside the drive's footage. This shows what a new stop-line model does on a drive without a phone. The app does
not use any of it.

- `dioramaweb` reads the drive and runs the driving model with commasim's code (`sim/commasim`). It runs the
  stop-line model on the narrow frame every other frame, as the phone does, then writes `drive.diorama` and
  `footage.mp4`.
- `page/` is the page that plays it, including the stop-line tracker the app's `DioramaStopLines.swift` follows. The
  car comes from `DioramaKit`'s own `diorama-assets.bin`.
- `tools/render_clip.js` screenshots a stretch of the page headlessly (puppeteer-core and Chrome), for a video.

The page and `dioramaweb` come from jetlink's Diorama (MIT).

## Run a drive

You need the sim's dependencies, ffmpeg, and onnxruntime-gpu for CUDA (onnxruntime works on the CPU, slowly).

    cd web
    python -m dioramaweb run <route>--* --model <driving model .onnx> \
        --stopline ../models/stopline_pilot.onnx --out drive/ --start 170 --seconds 20 --device cuda
    python -m dioramaweb view drive/            # http://127.0.0.1:8642/
    python -m dioramaweb camera                 # set each view's camera; saved in ~/.config/diorama/

The segments are a route's folders with `fcamera.hevc`, `ecamera.hevc` and `rlog.zst`. The calibration comes from
the rlog.

## A video

    cd tools && npm install
    node render_clip.js http://127.0.0.1:8642/ frames/ 170 190
    ffmpeg -framerate 20 -i frames/%05d.jpg -c:v libx264 -pix_fmt yuv420p clip.mp4

## Parity with the app

The app's scene (`DioramaScene.swift`) and Metal drawing (`DioramaRenderer.swift`) follow this page. Two hooks
let a test hold the app to it:

- `DioramaEngine.snapshot(state)` is a frame's scene as JSON. `tools/scene.js drive.diorama` writes one per frame,
  and the engine runs as it is in JavaScriptCore (no browser APIs).
- `diorama.html?drive=drive.diorama&still=FRAME&view=road|bird` draws one frame alone, filling the window, with the
  page's own cameras. It then sets `window.dioramaStill` to the snapshot and the title to `still ready`.
  `tools/still.js` saves PNGs and snapshots headlessly.

DioramaKit's tests (`sceneMatchesThePage`, `drawingMatchesThePage`) use both, on a synthetic drive with nothing from
a real one: `tools/synthetic_drive.py` writes it (cruising, a lead, a stop at a stop line, pulling away), and the
page's stills of it are committed beside it. After changing the page's drawing, make them again:

    python tools/synthetic_drive.py ../DioramaKit/Tests/Fixtures/synthetic/drive.diorama
    python -m dioramaweb view ../DioramaKit/Tests/Fixtures/synthetic --no-browser --port 8643 &
    cd tools && CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
      node still.js http://127.0.0.1:8643/ ../../DioramaKit/Tests/Fixtures/synthetic/stills 60,150,220,300,380 road,bird
    rm ../../DioramaKit/Tests/Fixtures/synthetic/stills/*.json
