# Diorama for iPhone: design

Diorama draws the drive as a live miniature on an iPhone plugged into a comma.
The comma runs the driving model; the iPhone runs only the stop-line model.

## Who does what

| | comma (zoompilot) | iPhone (this app) |
| --- | --- | --- |
| Driving model | runs it, as it always does | none |
| Narrow camera frame | warps it, streams it | runs the stop-line model on it |
| Driving model's outputs | streams the heads the scene draws from | draws the plan, lane lines, road edges and lead |
| Stop line | | decodes it, tracks it across frames, draws it |

The link is jetlink's protocol 3, the one zoompilot already speaks to an
iPhone over the cable, on the same cable network (jetlink's iOS gadget: the
comma is `192.168.60.1`, CDC-NCM, the phone dials). Diorama listens on its own
port, 5600, beside jetlink's 5599, and adds one flag and two hello keys;
nothing else changes.

## On the comma: dioramad

zoompilot pins this repo as `diorama_repo`, as it pins jetlink, and its
manager runs `diorama.dioramad` onroad while the Diorama setting is on. It is
its own process and never touches modeld:

- It reads the narrow camera through VisionIpc as the UI does, and modelV2,
  cameraOdometry and liveCalibration as any process does.
- While a phone watches, it warps each camera frame on the CPU
  (`diorama/warp.py`): the medmodel frame by nearest neighbour, as openpilot's
  own warp does, then packs it into six planes. Without a phone it only listens.
- It rebuilds the driving model's raw heads from modelV2 and cameraOdometry
  (`diorama/heads.py`), so the phone reads them as it reads a model's own
  output vector, whichever model the comma runs.
- It sends each model frame with its camera frame from a sender thread that
  keeps only the newest (`diorama/link.py`), at nice 10.

The cable's network only exists while jetlink's setting is on iOS, so
Diorama needs that setting too.

## On the wire

- **HELLO_REQ** (comma → phone, once the phone dials): the client JSON gains `show`, the layout of
  what it streams:

      "show": {"frame": [6, 128, 256],
               "outputs": [["plan", 990], ["lane_lines", 528], ...]}

  `frame` is the packed narrow frame (four Y sub-planes, U, V) in openpilot's
  medmodel frame (f 910, cx 256, cy 47.6). `outputs` names the driving
  model's raw heads in order, with their float counts: the heads in
  `DRIVE_FIELDS` (plan, lane_lines, lane_lines_prob, road_edges, lead,
  lead_prob, pose). From dioramad they are always the counts in
  `diorama/heads.py`; commasim sends its model's own. `camera` (calibration) rides along as it already does.
- **HELLO_RESP** (phone → comma): `{"viewer": true, "name": ...}`. A viewer
  never takes the drive: the comma asks it for no engine and never hands
  over; its own model keeps driving.
- **INFER_REQ with `Flag.SHOW` (1 << 3)**: `frame_id, flags`, then the packed
  narrow frame (196,608 bytes of uint8), then the outputs (float32, in the
  hello's order). About 200 KB a frame. The comma sends it without waiting
  (skip if the link is busy) and the phone sends nothing back.
- PING/PONG and LEAVE are unchanged.

The scene's drive packet (`D`) is exactly these outputs, so the phone turns
each frame into a `D` packet and its meta (`M`) from the hello: the Metal
scene is unchanged.

## The comma's cost

jetlink measured a 393 KB frame over this TCP link at about 9 ms of the
comma's CPU; Diorama's frame is about 200 KB. The warp is three gathers into
the camera buffer. All of it is dioramad's, at nice 10 and off modeld's
realtime core; how much it costs on a comma has not been measured yet.

## Developing on a Mac

`sim/commasim` plays the comma: it listens on port 5600, replays a recorded
drive (fcamera.hevc and rlog.zst), warps the narrow frame with the logged
calibration, runs the driving model with onnxruntime, and streams each frame
as above. The iOS Simulator app dials `127.0.0.1` instead of `192.168.60.1`.

## Stop line

`StoplinePilot.mlpackage` (Core ML, FP16): `frame` is Y as a 512x256
grayscale image, `uv` is U and V (1, 2, 128, 256) in float16, both as raw
0–255 values. `line` is (1, 8, 64, 128), at stride 4, with the heat map
already through a sigmoid; it is the model's one output. Decode the single
top line peak (the maximum of its 5x5 neighbourhood) with a score of at least
0.3. The tracker (`StopLines`) holds lines in the world frame, by the
driving model's pose, and draws the nearest one ahead that it has seen at
least 3 times.
