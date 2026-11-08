# Diorama Bench

An iPhone app that times stop-line Core ML models without a Mac attached. It
measures each model the same way the r18 vs r34 comparison did, so the numbers
can be compared. The models are for benchmarking only: they are bundled into
this app, and are never published or downloaded by Diorama itself.

## The models

The app bundles `models/bench11/Models`, which git ignores. That folder holds
`models.json` and one folder per model, named as in `models.json`, each
containing its unzipped `.mlpackage`:

    # copy the bench folder (the zips and models.json) from the training machine to models/bench11, then:
    cd models/bench11 && for z in *.zip; do n=${z#StoplinePilot-}; n=${n%.mlpackage.zip}; \
      mkdir -p Models/$n && unzip -oq $z -d Models/$n; done && cp models.json Models/

Check each zip against its `sha256` in `models.json` first.

## Build and install

    cd ios && xcodegen && open Diorama.xcodeproj

Choose the DioramaBench scheme and your iPhone, then Run. The bundle
identifier is your Diorama one with `.bench` appended, from
`Config/Local.xcconfig`. Use a Release build: debug builds are slower on the
CPU side of each step.

## Run

Open Diorama Bench, choose how long each model runs at 20 Hz and how long to
rest before each one, then tap Run All. The phone can be unplugged. Keep the
app in front: iOS suspends apps in the background. The screen stays on by
itself.

For each model, the app:

- waits out the rest, then waits until the phone is back to nominal
  temperature, and records the thermal state at the start;
- times compiling the model, its first load and a cached reload;
- times the prediction alone 200 times after a warm-up, with Core ML set to
  CPU and Neural Engine, to all units, and to CPU only (50 runs). It keeps
  every run's time, and records Core ML's placement of the operations,
  naming any that don't run on the Neural Engine;
- times the app's full step (input packing, prediction, decoding) 200 times
  back to back;
- runs the full step paced at 20 Hz for the chosen time, recording late
  frames, per-minute statistics and the thermal state.

If a model fails, the app records the error and moves on to the next one.

## The report

The app saves `bench-<date>T<time>.json` to its Documents after every model,
so a run that stops early still keeps its results. It's in Files → On My
iPhone → Diorama Bench, and the Share button sends it by AirDrop, Mail or
Messages.

The report includes the device, iOS version, Low Power Mode, the app's
version and git commit, the settings, a log of thermal state changes, and
each model's sha256 and measurements.
