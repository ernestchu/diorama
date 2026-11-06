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
