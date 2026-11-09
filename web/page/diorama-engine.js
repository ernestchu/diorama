// Copyright (c) 2026-, Zeph Leggett.
//
// This file comes from jetlink's Diorama (MIT) and is part of diorama, under the same license.
// See the LICENSE file in the root directory for more details.
//
// Diorama's engine: the server's messages (docs/diorama-internals.md) in, the scene out.
// The driving model's outputs give the path, lanes, road edges, lead and ego
// motion; the segmentation gives the road surface, which a road map holds
// around the car. A car is drawn at each of the comma's own lead bars, facing
// along its lane (CommaUI, as CommaUI.swift).
//
// Road frame throughout: x forward, y right, z down, origin at the camera, the
// road at z = height. No browser APIs here: diorama-render.js draws the scene,
// and node runs this file for the tests.
(function (root) {
  'use strict';

  const FPS = 20, DT = 1 / FPS, MPH = 2.23694;
  const IDX_N = 33;
  const X_IDXS = Array.from({ length: IDX_N }, (_, i) => 192 * (i / 32) ** 2);
  const PLAN_WIDTH = 15;
  const LEAD_MHP_N = 2, LEAD_SEL = 3, LEAD_LEN = 6, LEAD_WIDTH = 4;
  const EGO_FRONT = 1.6, EGO_REAR = -3.1;  // the ego car's bumpers, from the camera
  const OCCLUDER = 255;

  // ---------------------------------------------------------------- small math
  const clamp = (x, a, b) => Math.min(Math.max(x, a), b);
  const smoothstep = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
  const sexp = (x) => Math.exp(Math.min(x, 11));
  const sigmoid = (x) => 1 / (1 + sexp(-x));

  function interp(x, xs, ys) {
    if (x <= xs[0]) return ys[0];
    const n = xs.length;
    if (x >= xs[n - 1]) return ys[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
    const t = (x - xs[lo]) / (xs[hi] - xs[lo] || 1);
    return ys[lo] + t * (ys[hi] - ys[lo]);
  }

  // numpy's default (linear) percentile of an unsorted array
  function percentile(arr, q) {
    const a = Float64Array.from(arr).sort();
    if (!a.length) return NaN;
    const r = (q / 100) * (a.length - 1), lo = Math.floor(r), hi = Math.ceil(r);
    return a[lo] + (a[hi] - a[lo]) * (r - lo);
  }

  // 3x3 row-major helpers
  const m3mul = (A, B) => {
    const C = new Float64Array(9);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
    return C;
  };
  const m3vec = (A, v) => [A[0] * v[0] + A[1] * v[1] + A[2] * v[2], A[3] * v[0] + A[4] * v[1] + A[5] * v[2], A[6] * v[0] + A[7] * v[1] + A[8] * v[2]];
  function m3inv(A) {
    const [a, b, c, d, e, f, g, h, i] = A;
    const co = [e * i - f * h, -(d * i - f * g), d * h - e * g];
    const det = a * co[0] + b * co[1] + c * co[2];
    return Float64Array.from([co[0], -(b * i - c * h), b * f - c * e, co[1], a * i - c * g, -(a * f - c * d), co[2], -(a * h - b * g), a * e - b * d]).map((x) => x / det);
  }
  const VIEW_FROM_DEVICE = Float64Array.from([0, 1, 0, 0, 0, 1, 1, 0, 0]);
  // the comma's calibration of the wide camera from a meta's wide frame, [roll, pitch, yaw], or null
  const wideFromDevice = (wide) => (Array.isArray(wide.from_device) && wide.from_device.length === 3 ? wide.from_device : null);
  // openpilot's rot_from_euler: roll about x, then pitch about y, then yaw about z
  function rotFromEuler(r, p, y) {
    const cr = Math.cos(r), sr = Math.sin(r), cp = Math.cos(p), sp = Math.sin(p), cy = Math.cos(y), sy = Math.sin(y);
    return m3mul(Float64Array.from([cy, -sy, 0, sy, cy, 0, 0, 0, 1]), m3mul(Float64Array.from([cp, 0, sp, 0, 1, 0, -sp, 0, cp]), Float64Array.from([1, 0, 0, 0, cr, -sr, 0, sr, cr])));
  }

  // ---------------------------------------------------------------- messages
  // UTF-8 bytes -> string, without TextDecoder (JavaScriptCore, which the app's tests run this in, has none)
  function utf8(u8) {
    let s = '';
    for (let i = 0; i < u8.length;) {
      const c = u8[i++];
      let cp = c;
      if (c >= 0xf0) cp = ((c & 7) << 18) | ((u8[i++] & 63) << 12) | ((u8[i++] & 63) << 6) | (u8[i++] & 63);
      else if (c >= 0xe0) cp = ((c & 15) << 12) | ((u8[i++] & 63) << 6) | (u8[i++] & 63);
      else if (c >= 0xc0) cp = ((c & 31) << 6) | (u8[i++] & 63);
      s += String.fromCodePoint(cp);
    }
    return s;
  }

  function decode(buf) {
    const dv = new DataView(buf.buffer || buf, buf.byteOffset || 0, buf.byteLength);
    const kind = String.fromCharCode(dv.getUint8(0));
    const u8 = new Uint8Array(dv.buffer, dv.byteOffset, dv.byteLength);
    if (kind === 'M') return { kind, meta: JSON.parse(utf8(u8.subarray(1))) };
    if (kind === 'T') return { kind, text: utf8(u8.subarray(1)) };
    if (kind === 'I') {
      const w = dv.getUint16(5, true), h = dv.getUint16(7, true), planes = dv.getUint8(9);
      return { kind, frame: dv.getUint32(1, true), w, h, packed: u8.subarray(10, 10 + w * h * planes) };
    }
    if (kind === 'D') {
      const n = (dv.byteLength - 13) / 4;
      const values = new Float32Array(n);
      for (let i = 0; i < n; i++) values[i] = dv.getFloat32(13 + 4 * i, true);
      return { kind, frame: dv.getUint32(1, true), seconds: dv.getFloat32(5, true), ms: dv.getFloat32(9, true), values };
    }
    if (kind === 'S' || kind === 'W') {  // W: of the wide frame (meta's wide), the same otherwise
      const frame = dv.getUint32(1, true), ms = dv.getFloat32(5, true);
      const w = dv.getUint16(9, true), h = dv.getUint16(11, true), nq = dv.getUint16(13, true);
      let at = 15;
      const label = u8.slice(at, at + nq); at += nq;
      const score = u8.slice(at, at + nq); at += nq;
      const runs = dv.getUint32(at, true); at += 4;
      const qmap = new Uint8Array(w * h);
      let p = 0;
      for (let r = 0; r < runs; r++, at += 2) {
        const v = u8[at], len = u8[at + 1] + 1;
        qmap.fill(v, p, p + len);
        p += len;
      }
      return { kind: 'S', wide: kind === 'W', frame, ms, w, h, label, score, qmap };
    }
    if (kind === 'L') {  // stop lines (diorama/stopline.py): score and two end points each, in the frame's pixels
      const n = dv.getUint8(6), lines = [];
      for (let i = 0; i < n; i++) {
        const at = 7 + 20 * i, f = (j) => dv.getFloat32(at + 4 * j, true);
        lines.push({ score: f(0), x1: f(1), y1: f(2), x2: f(3), y2: f(4) });
      }
      return { kind, frame: dv.getUint32(1, true), wide: dv.getUint8(5) === 1, lines };
    }
    return { kind };
  }

  // A recorded drive: messages one after another, each after its u32 length.
  function* readFile(buf) {
    const dv = new DataView(buf);
    for (let at = 0; at + 4 <= buf.byteLength;) {
      const n = dv.getUint32(at, true);
      yield decode(new Uint8Array(buf, at + 4, n));
      at += 4 + n;
    }
  }

  // ---------------------------------------------------------------- driving model outputs
  // The heads openpilot's Parser makes of them (diorama/outputs.py does the same).
  function parseDrive(values, fields) {
    const raw = {};
    let at = 0;
    for (const [name, n] of fields) { raw[name] = values.subarray(at, at + n); at += n; }
    const mdn = (r) => { const n = r.length / 2; return [Float64Array.from(r.subarray(0, n)), Float64Array.from(r.subarray(n, 2 * n), sexp)]; };
    const out = {};
    for (const name of ['plan', 'lane_lines', 'road_edges', 'pose']) {
      if (raw[name]) [out[name], out[name + '_stds']] = mdn(raw[name]);
    }
    for (const name of ['lane_lines_prob', 'lead_prob']) if (raw[name]) out[name] = Float64Array.from(raw[name], sigmoid);
    if (raw.lead) {
      const r = raw.lead;
      if (r.length !== 2 * LEAD_SEL * LEAD_LEN * LEAD_WIDTH) {
        const per = r.length / LEAD_MHP_N, n = (per - LEAD_SEL) / 2;
        const lead = new Float64Array(LEAD_SEL * n), stds = new Float64Array(LEAD_SEL * n);
        for (let s = 0; s < LEAD_SEL; s++) {
          // the hypothesis the softmax over hypotheses weights most, for this selection
          let best = 0;
          for (let hyp = 1; hyp < LEAD_MHP_N; hyp++) if (r[hyp * per + per - LEAD_SEL + s] > r[best * per + per - LEAD_SEL + s]) best = hyp;
          for (let i = 0; i < n; i++) { lead[s * n + i] = r[best * per + i]; stds[s * n + i] = sexp(r[best * per + n + i]); }
        }
        out.lead = lead; out.lead_stds = stds;
      } else {
        [out.lead, out.lead_stds] = mdn(r);
      }
    }
    return out;
  }
  // accessors on the flat arrays, in openpilot's shapes
  const plan = (p, i, c) => p[i * PLAN_WIDTH + c];  // (33, 15): 0..2 position, 3..5 velocity, 6..8 acceleration
  const lane = (l, line, i, c) => l[(line * IDX_N + i) * 2 + c];  // (4|2, 33, 2): y, z
  const leadAt = (l, sel, t, c) => l[(sel * LEAD_LEN + t) * LEAD_WIDTH + c];  // (3, 6, 4): x, y, v, a

  // Element-wise random walk Kalman filter: R from the model's std, Q = (k std)^2.
  class KF {
    constructor(k) { this.k = k; this.x = null; this.P = null; }
    reset() { this.x = this.P = null; }
    update(z, std) {
      const n = z.length;
      if (this.x === null || this.x.length !== n) {
        this.x = Float64Array.from(z);
        this.P = Float64Array.from(std, (s) => Math.max(s, 1e-3) ** 2);
        return this.x;
      }
      for (let i = 0; i < n; i++) {
        const r = Math.max(std[i], 1e-3) ** 2, P = this.P[i] + this.k * this.k * r, K = P / (P + r);
        this.x[i] += K * (z[i] - this.x[i]);
        this.P[i] = (1 - K) * P;
      }
      return this.x;
    }
  }

  class OutputSmoother {
    constructor(k) {
      this.f = {};
      for (const n of ['plan', 'lane_lines', 'road_edges', 'lane_prob', 'edge_std']) this.f[n] = new KF(k);
    }
    smooth(o) {
      const s = Object.assign({}, o);
      const tenth = (a) => new Float64Array(a.length).fill(0.1);
      s.plan = Float64Array.from(this.f.plan.update(o.plan, o.plan_stds));
      s.lane_lines = Float64Array.from(this.f.lane_lines.update(o.lane_lines, o.lane_lines_stds));
      s.road_edges = Float64Array.from(this.f.road_edges.update(o.road_edges, o.road_edges_stds));
      s.lane_lines_prob = Float64Array.from(this.f.lane_prob.update(o.lane_lines_prob, tenth(o.lane_lines_prob)));
      s.road_edges_stds = Float64Array.from(this.f.edge_std.update(o.road_edges_stds, tenth(o.road_edges_stds)));
      return s;
    }
  }

  // ---------------------------------------------------------------- the camera that saw the frame
  // The driving model's frames are already calibrated: road point -> pixel is K view_from_device, K the narrow
  // frame's (meta.frame) or the wide's (meta.wide). The comma warps the wide camera with the narrow one's
  // calibration, so the wide frame is turned by the wide camera's angles from the device too (rotation, the model's
  // wide_from_device_euler).
  class FrameCamera {
    constructor(meta, f = meta.frame, rotation = null) {
      this.w = f.w; this.h = f.h; this.f = f.f;
      this.height = meta.height;
      const K = Float64Array.from([f.f, 0, f.cx, 0, f.f, f.cy, 0, 0, 1]);
      this.P = m3mul(K, rotation ? m3mul(VIEW_FROM_DEVICE, rotation) : VIEW_FROM_DEVICE);
      this.Pinv = m3inv(this.P);
    }
    project(x, y, z) {
      const P = this.P, w = P[6] * x + P[7] * y + P[8] * z;
      return [(P[0] * x + P[1] * y + P[2] * z) / w, (P[3] * x + P[4] * y + P[5] * z) / w, w];
    }
    // the road point where pixel (u, v)'s ray meets the plane z = gz, or null
    groundHit(u, v, gz) {
      const d = m3vec(this.Pinv, [u, v, 1]);
      if (!(d[2] > 1e-6)) return null;
      const t = gz / d[2];
      return t > 0 ? [d[0] * t, d[1] * t, d[2] * t] : null;
    }
  }

  // ---------------------------------------------------------------- the ego car's path through the world
  // pos_k + R(yaw_k) p_ego, integrated from the model's pose (velocity, yaw rate). Kept for the last few seconds.
  class EgoTrail {
    constructor() { this.poses = new Map(); this.last = null; }
    reset() { this.poses.clear(); this.last = null; }
    add(frame, pose) {
      if (this.last === null) { this.poses.set(frame, [0, 0, 0]); this.last = frame; return; }
      let [x, y, yaw] = this.poses.get(this.last);
      const steps = clamp(frame - this.last, 1, 20);  // a gap in a live feed: assume the car kept going
      for (let i = 0; i < steps; i++) {
        const t0 = pose[0] * DT, t1 = pose[1] * DT, c = Math.cos(yaw), s = Math.sin(yaw);
        x += c * t0 - s * t1; y += s * t0 + c * t1; yaw += pose[5] * DT;
      }
      this.poses.set(frame, [x, y, yaw]);
      this.last = frame;
      for (const k of this.poses.keys()) { if (k < frame - 200) this.poses.delete(k); else break; }
    }
    pose(frame) { return this.poses.get(frame) || null; }
    egoToWorld(p, x, y) { const c = Math.cos(p[2]), s = Math.sin(p[2]); return [p[0] + c * x - s * y, p[1] + s * x + c * y]; }
    worldToEgo(p, wx, wy) { const c = Math.cos(p[2]), s = Math.sin(p[2]), dx = wx - p[0], dy = wy - p[1]; return [c * dx + s * dy, -s * dx + c * dy]; }
  }

  // ---------------------------------------------------------------- segmentation
  // The query map as the road map's classes (0 other, 1 road, 2 lane paint, 3 crosswalk, 4 curb, 5 sidewalk,
  // OCCLUDER under a thing) and the things, each with its pixels.
  function readSeg(msg, meta) {
    const seg = meta.seg, labels = seg.labels;
    const nq = msg.label.length;
    const qClass = new Uint8Array(256), qThing = new Array(256).fill(null);
    for (let q = 0; q < nq; q++) {
      const l = msg.label[q];
      if (l === 255) continue;
      const name = labels[l];
      if (seg.things[name]) { qClass[q] = OCCLUDER; qThing[q] = seg.things[name]; } else qClass[q] = seg.road_class[name] || 0;
    }
    const n = msg.w * msg.h, cls = new Uint8Array(n);
    for (let i = 0; i < n; i++) cls[i] = msg.qmap[i] === 255 ? 0 : qClass[msg.qmap[i]];
    return { w: msg.w, h: msg.h, ds: seg.ds, wide: !!msg.wide, cls, qmap: msg.qmap, qThing };
  }

  // ---------------------------------------------------------------- stop lines
  // The stop-line model's lines, held in the world. The model gives at most one line a frame, the ego lane's, so a
  // sighting whose middle is within STOP_SAME m of a held line's along the road (and STOP_SIDE m across), as the
  // car sees it then, is that line, whatever its angle: the held line's middle, heading and length move 0.3 of the
  // way to it. Held lines that come that close merge, the more seen keeping its place. Only the nearest line ahead
  // that has been seen STOP_SEEN times is drawn. A line is dropped once behind the car, or STOP_KEEP frames after it
  // was last seen unless it is within STOP_NEAR m ahead, below the camera's view over the hood (a car stopped at its
  // line no longer sees it).
  const STOP_SAME = 3, STOP_SIDE = 3, STOP_KEEP = 30, STOP_MAX = 50, STOP_WIDTH = 0.4, STOP_SEEN = 3, STOP_NEAR = 6;
  // Each sighting is first fitted to the ego lane (egoLane, fitToLane): its ends moved, along the line, to where it
  // crosses the lane's two boundaries, longer or shorter. Near the line the camera loses its ends over the hood and the
  // model's line shortens; fitted, it keeps the lane's width. A boundary is the lane line, or the road edge where that
  // lies inside it, as the labels were cut (train/egotrim.py). Whatever the lines' probabilities, as there too: at an
  // intersection the paint ends and the model's confidence in the lines drops, but where it puts them still bounds the
  // ego lane, and that is where the stop line is fitted.
  // The ego lane's two boundaries this frame, left then right, as (x, y) road points near to far; null without them.
  function egoLane(out) {
    if (!out.lane_lines || !out.road_edges) return null;
    return [0, 1].map((side) => X_IDXS.map((x, i) => {
      const ly = lane(out.lane_lines, 1 + side, i, 0), ey = lane(out.road_edges, side, i, 0);
      const inside = side ? ey < ly : ey > ly;  // left: the edge right of the line; right: left of it
      return [x, inside ? ey : ly];
    }));
  }
  // The line A-B (road points, in the frame the boundaries are in) fitted to the lane: [A', B'] on its own extension
  // where it crosses the two boundaries, or null where it does not cross both.
  function fitToLane(A, B, bounds) {
    const dx = B[0] - A[0], dy = B[1] - A[1], dd = dx * dx + dy * dy;
    if (dd < 1e-6) return null;
    const ts = bounds.map((c) => {
      const s = c.map(([x, y]) => (x - A[0]) * -dy + (y - A[1]) * dx);  // which side of the line each point is
      for (let i = 0; i + 1 < c.length; i++) {
        if (s[i] === 0 || s[i] * s[i + 1] < 0) {
          const f = s[i] / (s[i] - s[i + 1]), px = c[i][0] + f * (c[i + 1][0] - c[i][0]), py = c[i][1] + f * (c[i + 1][1] - c[i][1]);
          return ((px - A[0]) * dx + (py - A[1]) * dy) / dd;
        }
      }
      return null;
    });
    if (ts[0] === null || ts[1] === null || ts[0] === ts[1]) return null;
    const t0 = Math.min(ts[0], ts[1]), t1 = Math.max(ts[0], ts[1]);
    return [[A[0] + t0 * dx, A[1] + t0 * dy], [A[0] + t1 * dx, A[1] + t1 * dy]];
  }
  // a line's heading as a half turn: lines have no direction
  const halfTurn = (d) => d - Math.PI * Math.round(d / Math.PI);
  function stopMove(h, m, ang, len, w) {
    h.m = [h.m[0] + w * (m[0] - h.m[0]), h.m[1] + w * (m[1] - h.m[1])];
    h.ang = h.ang + w * halfTurn(ang - h.ang);
    h.len = h.len + w * (len - h.len);
  }
  class StopLines {
    constructor() { this.held = []; }
    // `now`, `bounds`: the car's pose this frame and the ego lane's boundaries in it (egoLane), or bounds null
    observe(k, pose, msg, cam, road, trail, now = null, bounds = null) {
      const same = (h, x, y) => {
        const [hx, hy] = trail.worldToEgo(pose, h.m[0], h.m[1]);
        return Math.abs(hx - x) < STOP_SAME && Math.abs(hy - y) < STOP_SIDE;
      };
      for (const l of msg.lines) {
        const A = cam.groundHit(l.x1, l.y1, road), B = cam.groundHit(l.x2, l.y2, road);
        if (!A || !B || (A[0] + B[0]) / 2 > STOP_MAX) continue;
        let a = trail.egoToWorld(pose, A[0], A[1]), b = trail.egoToWorld(pose, B[0], B[1]);
        if (bounds) {  // the boundaries are in this frame's road frame, the sighting in its own frame's
          const fit = fitToLane(trail.worldToEgo(now, a[0], a[1]), trail.worldToEgo(now, b[0], b[1]), bounds);
          if (fit) [a, b] = fit.map(([fx, fy]) => trail.egoToWorld(now, fx, fy));
        }
        const m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
        const len = Math.hypot(b[0] - a[0], b[1] - a[1]), [x, y] = trail.worldToEgo(pose, m[0], m[1]);
        const h = this.held.filter((h) => same(h, x, y)).sort((p, q) => q.hits - p.hits)[0];
        if (!h) { this.held.push({ m, ang, len, last: k, hits: 1 }); continue; }
        stopMove(h, m, ang, len, 0.3);
        h.last = k; h.hits++;
      }
      this.held.sort((p, q) => q.hits - p.hits);
      for (let i = 0; i < this.held.length; i++) {
        const h = this.held[i], [x, y] = trail.worldToEgo(pose, h.m[0], h.m[1]);
        for (let j = this.held.length - 1; j > i; j--) {
          const g = this.held[j];
          if (!same(g, x, y)) continue;
          stopMove(h, g.m, g.ang, g.len, g.hits / (h.hits + g.hits));
          h.hits += g.hits; h.last = Math.max(h.last, g.last);
          this.held.splice(j, 1);
        }
      }
    }
    // the line to draw now, if any, as a 4-point polygon in the world (as the crosswalk stripes)
    quads(k, pose, trail) {
      const ahead = (h) => trail.worldToEgo(pose, h.m[0], h.m[1])[0];
      this.held = this.held.filter((h) => {
        const x = ahead(h);
        return x > -3 && (k - h.last <= STOP_KEEP || (h.hits >= STOP_SEEN && x < STOP_NEAR));
      });
      const seen = this.held.filter((h) => h.hits >= STOP_SEEN).sort((p, q) => ahead(p) - ahead(q));
      return seen.slice(0, 1).map(({ m, ang, len }) => {
        const ux = Math.cos(ang) * len / 2, uy = Math.sin(ang) * len / 2;
        const nx = -Math.sin(ang) * STOP_WIDTH / 2, ny = Math.cos(ang) * STOP_WIDTH / 2;
        const a = [m[0] - ux, m[1] - uy], b = [m[0] + ux, m[1] + uy];
        return [[a[0] + nx, a[1] + ny], [b[0] + nx, b[1] + ny], [b[0] - nx, b[1] - ny], [a[0] - nx, a[1] - ny]];
      });
    }
  }

  // ---------------------------------------------------------------- road map
  // The road surface around the car in a world window that moves with it: per 0.1 m cell, running averages of the
  // segmentation's road classes, and how much of it has been seen. Two RGBA8 layers, uploaded as textures:
  //   a = (road, lane paint, crosswalk, curb), b = (sidewalk, seen, -, -)
  class RoadMap {
    constructor(cam) {
      this.cam = cam;
      this.RES = 0.1;
      this.N = 1536;  // 153.6 m
      this.OBS_X = [3.0, 40.0]; this.OBS_Y = 25.0;
      this.ALPHA = 0.44;
      this.a = new Uint8Array(this.N * this.N * 4);
      this.b = new Uint8Array(this.N * this.N * 4);
      this.reset();
    }
    reset() {
      this.a.fill(0); this.b.fill(0);
      this.origin = null;  // world (x, y) of cell (0, 0)'s corner; rows run along world x, columns along world y
      this.dirty = null;   // [r0, r1, c0, c1] changed since the renderer last uploaded, or 'all'
      this.crosswalkSeen = 0;
    }
    // Keep the car within 30 m of the window's centre, moving the window (and what it holds) when it is not.
    follow(p) {
      const half = this.N * this.RES / 2;
      if (this.origin === null) { this.origin = [p[0] - half, p[1] - half]; this.dirty = 'all'; return; }
      const cx = this.origin[0] + half, cy = this.origin[1] + half;
      if (Math.abs(p[0] - cx) < 30 && Math.abs(p[1] - cy) < 30) return;
      const dr = Math.round((p[0] - cx) / this.RES), dc = Math.round((p[1] - cy) / this.RES), N = this.N;
      for (const m of [this.a, this.b]) {
        const old = m.slice();
        m.fill(0);
        for (let r = Math.max(0, -dr); r < Math.min(N, N - dr); r++) {
          const c0 = Math.max(0, -dc), c1 = Math.min(N, N - dc);
          if (c1 > c0) m.set(old.subarray(((r + dr) * N + c0 + dc) * 4, ((r + dr) * N + c1 + dc) * 4), (r * N + c0) * 4);
        }
      }
      this.origin = [this.origin[0] + dr * this.RES, this.origin[1] + dc * this.RES];
      this.dirty = 'all';
    }
    // seg as cam saw it: the map's own camera (the narrow frame's) or the wide frame's, whose pixel covers as much
    // road as the narrow's twice as far, so it reaches half as far and weighs as the narrow's would there
    observe(p, seg, zpath, cam = this.cam) {
      this.follow(p);
      const { RES, N } = this, c = Math.cos(p[2]), s = Math.sin(p[2]);
      const fine = cam.f / this.cam.f, farX = this.OBS_X[1] * Math.min(fine, 1);
      // the observation window's cells (3..40 m ahead, 25 m either side) as world cells
      const corners = [[this.OBS_X[0], -this.OBS_Y], [this.OBS_X[0], this.OBS_Y], [farX, -this.OBS_Y], [farX, this.OBS_Y]]
        .map(([x, y]) => [p[0] + c * x - s * y, p[1] + s * x + c * y]);
      const r0 = Math.max(0, Math.floor((Math.min(...corners.map((q) => q[0])) - this.origin[0]) / RES));
      const r1 = Math.min(N, Math.floor((Math.max(...corners.map((q) => q[0])) - this.origin[0]) / RES) + 1);
      const c0 = Math.max(0, Math.floor((Math.min(...corners.map((q) => q[1])) - this.origin[1]) / RES));
      const c1 = Math.min(N, Math.floor((Math.max(...corners.map((q) => q[1])) - this.origin[1]) / RES) + 1);
      if (r1 <= r0 || c1 <= c0) return;
      const P = cam.P, h = cam.height, sw = seg.w, sh = seg.h, ds = seg.ds, W = cam.w, H = cam.h;
      const a = this.a, b = this.b;
      // the road height along the path, tabulated per metre (zpath is the plan's, interpolated)
      const zTab = new Float64Array(42);
      for (let i = 0; i < 42; i++) zTab[i] = zpath(i);
      let crosswalk = 0;
      for (let r = r0; r < r1; r++) {
        const dx = this.origin[0] + (r + 0.5) * RES - p[0];
        for (let col = c0; col < c1; col++) {
          const dy = this.origin[1] + (col + 0.5) * RES - p[1];
          const ex = c * dx + s * dy, ey = -s * dx + c * dy;
          if (ex < this.OBS_X[0] || ex > farX || Math.abs(ey) > this.OBS_Y) continue;
          const i0 = Math.floor(ex), ez = h + zTab[i0] + (zTab[i0 + 1] - zTab[i0]) * (ex - i0);
          const z = Math.max(P[6] * ex + P[7] * ey + P[8] * ez, 1e-3);
          if (z <= 1) continue;
          const u = (P[0] * ex + P[1] * ey + P[2] * ez) / z, v = (P[3] * ex + P[4] * ey + P[5] * ez) / z;
          if (!(u >= 0 && u < W - 1 && v >= 0 && v < H)) continue;
          const k = seg.cls[Math.min(Math.floor(v / ds), sh - 1) * sw + Math.min(Math.floor(u / ds), sw - 1)];
          if (k === OCCLUDER) continue;  // a car or a person covers the road here: no evidence either way
          // near observations are more accurate (one pixel covers less road), so they weigh more
          const alpha = this.ALPHA * clamp(1.4 - ex / fine / 30, 0.25, 1);
          const at = (r * N + col) * 4;
          // + 0.5: the bytes round rather than truncate
          a[at] += alpha * ((k === 1) * 255 - a[at]) + 0.5;
          a[at + 1] += alpha * ((k === 2) * 255 - a[at + 1]) + 0.5;
          a[at + 2] += alpha * ((k === 3) * 255 - a[at + 2]) + 0.5;
          a[at + 3] += alpha * ((k === 4) * 255 - a[at + 3]) + 0.5;
          b[at] += alpha * ((k === 5) * 255 - b[at]) + 0.5;
          b[at + 1] = Math.min(b[at + 1] + alpha * 255 + 0.5, 255);
          if (k === 3) crosswalk++;
        }
      }
      this.crosswalkSeen = crosswalk > 20 ? 40 : Math.max(this.crosswalkSeen - 1, 0);
      if (this.dirty !== 'all') {
        const d = this.dirty;
        this.dirty = d ? [Math.min(d[0], r0), Math.max(d[1], r1), Math.min(d[2], c0), Math.max(d[3], c1)] : [r0, r1, c0, c1];
      }
    }
    // bilinear read of a layer channel at world (wx, wy), 0..1
    sample(layer, ch, wx, wy) {
      const fr = (wx - this.origin[0]) / this.RES - 0.5, fc = (wy - this.origin[1]) / this.RES - 0.5;
      const r = Math.floor(fr), c = Math.floor(fc), N = this.N;
      if (r < 0 || c < 0 || r >= N - 1 || c >= N - 1) return 0;
      const tr = fr - r, tc = fc - c, at = (r * N + c) * 4 + ch;
      return ((layer[at] * (1 - tc) + layer[at + 4] * tc) * (1 - tr) + (layer[at + 4 * N] * (1 - tc) + layer[at + 4 * N + 4] * tc) * tr) / 255;
    }

    // Zebra stripes, as 4-point polygons in world coordinates (fsd_roadmap.RoadMap.crosswalks).
    //
    // Crosswalks at an intersection merge into one region, so the mask is split into straight bands: bands crossing
    // our road and bands crossing a side street. The split runs in a frame aligned with the crosswalk grid itself
    // (edge orientations mod 90 degrees, the axis nearer our lane taken as along), so it still works mid-turn.
    // Every pixel goes to exactly one band (the direction its run is longer in), so the corners where two crossings
    // meet are not striped twice. Bars run along the traffic of the road they cross, at most maxWidth long.
    crosswalks(p, laneHeading, minArea = 2.0, bandLen = 4.5, maxWidth = 4.0) {
      if (!this.crosswalkSeen || this.origin === null) return [];
      const res = 0.2, XMAX = 90, XMIN = -30, YMAX = 30;
      const W2 = Math.round(2 * YMAX / res), H2 = Math.round((XMAX - XMIN) / res);
      const c = Math.cos(p[2]), s = Math.sin(p[2]);
      const k = Math.floor(bandLen / res);
      // the crosswalk raster, ego frame at 0.2 m
      const full = new Uint8Array(W2 * H2);
      let rmin = H2, rmax = -1, cmin = W2, cmax = -1, count = 0;
      for (let row = 0; row < H2; row++) {
        const x = XMAX - (row + 0.5) * res;
        for (let col = 0; col < W2; col++) {
          const y = -YMAX + (col + 0.5) * res, wx = p[0] + c * x - s * y, wy = p[1] + s * x + c * y;
          if (this.sample(this.a, 2, wx, wy) > 0.5 && this.sample(this.b, 1, wx, wy) > 0.5) {
            full[row * W2 + col] = 1; count++;
            rmin = Math.min(rmin, row); rmax = Math.max(rmax, row); cmin = Math.min(cmin, col); cmax = Math.max(cmax, col);
          }
        }
      }
      if (count * res * res < minArea) return [];
      // everything after works in a square crop around it, large enough for the grid turned any way plus a band
      const half = Math.ceil(Math.hypot(rmax - rmin, cmax - cmin) / 2) + k + 4;
      const R0 = Math.max(Math.floor((rmin + rmax) / 2) - half, 0), R1 = Math.min(Math.floor((rmin + rmax) / 2) + half, H2);
      const C0 = Math.max(Math.floor((cmin + cmax) / 2) - half, 0), C1 = Math.min(Math.floor((cmin + cmax) / 2) + half, W2);
      const W = C1 - C0, H = R1 - R0;
      const toXY = (col, row) => [XMAX - (row + R0 + 0.5) * res, -YMAX + (col + C0 + 0.5) * res];
      const toPx = (x, y) => [(y + YMAX) / res - 0.5 - C0, (XMAX - x) / res - 0.5 - R0];
      let mask = new Uint8Array(W * H);
      for (let row = 0; row < H; row++) mask.set(full.subarray((row + R0) * W2 + C0, (row + R0) * W2 + C1), row * W);
      mask = morph(mask, W, H, 3, 3, 'close');
      const mxs = [], mys = [];
      let r0 = H, r1 = 0, q0 = W, q1 = 0;
      for (let row = 0; row < H; row++) for (let col = 0; col < W; col++) if (mask[row * W + col]) {
        const [x, y] = toXY(col, row); mxs.push(x); mys.push(y);
        r0 = Math.min(r0, row); r1 = Math.max(r1, row); q0 = Math.min(q0, col); q1 = Math.max(q1, col);
      }
      const cx = percentile(mxs, 50), cy = percentile(mys, 50);

      // grid orientation from the mask's edges, 4x the angle so the two crossing directions vote together
      r0 = Math.max(r0 - 4, 0); r1 = Math.min(r1 + 5, H); q0 = Math.max(q0 - 4, 0); q1 = Math.min(q1 + 5, W);
      const cw = q1 - q0, ch = r1 - r0, f = blur5(mask, W, r0, r1, q0, q1);
      let zr = 0, zi = 0, sumG = 0;
      for (let row = 1; row < ch - 1; row++) for (let col = 1; col < cw - 1; col++) {
        const at = row * cw + col;
        const gc = (f[at - cw + 1] + 2 * f[at + 1] + f[at + cw + 1]) - (f[at - cw - 1] + 2 * f[at - 1] + f[at + cw - 1]);
        const gr = (f[at + cw - 1] + 2 * f[at + cw] + f[at + cw + 1]) - (f[at - cw - 1] + 2 * f[at - cw] + f[at - cw + 1]);
        if (gc * gc + gr * gr <= 1e-3) continue;
        const gx = -gr, gy = gc, g2 = gx * gx + gy * gy, th = 4 * Math.atan2(gy, gx);  // rows run along -x
        zr += g2 * Math.cos(th); zi += g2 * Math.sin(th); sumG += g2;
      }
      const lane = laneHeading(cx);
      let phi = lane;
      if (Math.hypot(zr, zi) > 0.2 * sumG) {  // a clear grid; otherwise trust the lane direction
        const ang = Math.atan2(zi, zr) / 4;
        phi = lane + ((((ang - lane + Math.PI / 4) % (Math.PI / 2)) + Math.PI / 2) % (Math.PI / 2) - Math.PI / 4);
      }
      const U = [Math.cos(phi), Math.sin(phi)], Wv = [-U[1], U[0]];

      // the aligned frame: turned by -phi about the crosswalk's centre, so U runs along -rows
      const al = new Uint8Array(W * H);
      let ar0 = H, ar1 = 0, ac0 = W, ac1 = 0;
      for (let row = 0; row < H; row++) for (let col = 0; col < W; col++) {
        const [ax, ay] = toXY(col, row), dx = ax - cx, dy = ay - cy;
        const [ec, er] = toPx(cx + U[0] * dx + Wv[0] * dy, cy + U[1] * dx + Wv[1] * dy), ic = Math.round(ec), ir = Math.round(er);
        if (ic >= 0 && ir >= 0 && ic < W && ir < H && mask[ir * W + ic]) {
          al[row * W + col] = 1;
          ar0 = Math.min(ar0, row); ar1 = Math.max(ar1, row + 1); ac0 = Math.min(ac0, col); ac1 = Math.max(ac1, col + 1);
        }
      }
      if (ar1 <= ar0) return [];
      const across = new Uint8Array(W * H), along = new Uint8Array(W * H);
      const hrun = runs(al, W, ar0, ar1, ac0, ac1, true), vrun = runs(al, W, ar0, ar1, ac0, ac1, false);
      for (let row = ar0; row < ar1; row++) for (let col = ac0; col < ac1; col++) {
        const at = row * W + col;
        if (al[at]) (hrun[at] >= vrun[at] ? across : along)[at] = 1;
      }
      const stripes = [];
      for (const [kind, band] of [['across', morph(across, W, H, 1, k, 'open')], ['along', morph(along, W, H, k, 1, 'open')]]) {
        // back to the ego raster
        const back = new Uint8Array(W * H);
        for (let row = 0; row < H; row++) for (let col = 0; col < W; col++) {
          const [x, y] = toXY(col, row), dx = x - cx, dy = y - cy;
          const [ac, arow] = toPx(cx + U[0] * dx + U[1] * dy, cy + Wv[0] * dx + Wv[1] * dy), ic = Math.round(ac), ir = Math.round(arow);
          if (ic >= 0 && ir >= 0 && ic < W && ir < H && band[ir * W + ic]) back[row * W + col] = 1;
        }
        const { labels, comps } = components(back, W, H);
        comps.forEach((pix, ci) => {
          if (pix.length * res * res < minArea) return;
          const pu = [], pw = [];
          for (const at of pix) {
            const [x, y] = toXY(at % W, Math.floor(at / W));
            pu.push(x * U[0] + y * U[1]); pw.push(x * Wv[0] + y * Wv[1]);
          }
          const extU = [percentile(pu, 2), percentile(pu, 98)], extW = [percentile(pw, 2), percentile(pw, 98)];
          const [longAx, lext] = kind === 'across' ? [Wv, extW] : [U, extU];
          let [shortAx, sext, sp] = kind === 'across' ? [U, extU, pu] : [Wv, extW, pw];
          if (sext[1] - sext[0] < 0.8) return;
          if (sext[1] - sext[0] > maxWidth) {
            const m = percentile(sp, 50);
            sext = [clamp(m - maxWidth / 2, sext[0], sext[1]), clamp(m + maxWidth / 2, sext[0], sext[1])];
          }
          const bar = 0.5, gap = 0.6;
          for (let o = lext[0]; o <= lext[1] - bar + 1e-6; o += bar + gap) {
            const pt = (l, sh) => [longAx[0] * l + shortAx[0] * sh, longAx[1] * l + shortAx[1] * sh];
            const quad = [pt(o, sext[0]), pt(o, sext[1]), pt(o + bar, sext[1]), pt(o + bar, sext[0])];
            // only bars whose centre lies on the band
            const [qcol, qrow] = toPx((quad[0][0] + quad[2][0]) / 2, (quad[0][1] + quad[2][1]) / 2), ci0 = Math.round(qcol), ri0 = Math.round(qrow);
            let on = false;
            for (let r = Math.max(ri0 - 2, 0); r < Math.min(ri0 + 3, H) && !on; r++)
              for (let cc = Math.max(ci0 - 2, 0); cc < Math.min(ci0 + 3, W); cc++) if (labels[r * W + cc] === ci + 1) { on = true; break; }
            if (on) stripes.push(quad.map(([x, y]) => [p[0] + c * x - s * y, p[1] + s * x + c * y]));
          }
        });
      }
      return stripes;
    }
  }

  // Binary morphology with a kh x kw rectangle anchored at its centre, as cv2's (the image border never erodes).
  function morph(m, W, H, kh, kw, op) {
    const pass = (src, dilate, horizontal, k) => {
      if (k <= 1) return src;
      const out = new Uint8Array(W * H), lo = -(k >> 1), hi = k - 1 - (k >> 1);
      for (let row = 0; row < H; row++) for (let col = 0; col < W; col++) {
        let v = dilate ? 0 : 1;
        for (let d = lo; d <= hi; d++) {
          const r = horizontal ? row : row + d, c = horizontal ? col + d : col;
          const s = r < 0 || c < 0 || r >= H || c >= W ? (dilate ? 0 : 1) : src[r * W + c];
          if (dilate ? s : !s) { v = dilate ? 1 : 0; break; }
        }
        out[row * W + col] = v;
      }
      return out;
    };
    const erode = (x) => pass(pass(x, false, true, kw), false, false, kh);
    const dilate = (x) => pass(pass(x, true, true, kw), true, false, kh);
    return op === 'open' ? dilate(erode(m)) : erode(dilate(m));
  }

  // 5x5 Gaussian (sigma 1) of a crop of a binary mask, as float
  function blur5(m, W, r0, r1, c0, c1) {
    const k = [0.0545, 0.2442, 0.4026, 0.2442, 0.0545], w = c1 - c0, h = r1 - r0;
    const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
    for (let row = 0; row < h; row++) for (let col = 0; col < w; col++) {
      let s = 0;
      for (let d = -2; d <= 2; d++) { const cc = clamp(col + d, 0, w - 1); s += k[d + 2] * m[(row + r0) * W + cc + c0]; }
      tmp[row * w + col] = s;
    }
    for (let row = 0; row < h; row++) for (let col = 0; col < w; col++) {
      let s = 0;
      for (let d = -2; d <= 2; d++) s += k[d + 2] * tmp[clamp(row + d, 0, h - 1) * w + col];
      out[row * w + col] = s;
    }
    return out;
  }

  // the length of the horizontal (or vertical) run of ones through every pixel of a crop
  function runs(m, W, r0, r1, c0, c1, horizontal) {
    const out = new Int32Array(m.length);
    if (horizontal) {
      for (let row = r0; row < r1; row++) for (let col = c0; col < c1;) {
        if (!m[row * W + col]) { col++; continue; }
        let e = col;
        while (e < c1 && m[row * W + e]) e++;
        for (let i = col; i < e; i++) out[row * W + i] = e - col;
        col = e;
      }
    } else {
      for (let col = c0; col < c1; col++) for (let row = r0; row < r1;) {
        if (!m[row * W + col]) { row++; continue; }
        let e = row;
        while (e < r1 && m[e * W + col]) e++;
        for (let i = row; i < e; i++) out[i * W + col] = e - row;
        row = e;
      }
    }
    return out;
  }

  // 8-connected components: labels (1-based) and each one's pixel indices
  function components(m, W, H) {
    const labels = new Int32Array(W * H), comps = [];
    const stack = [];
    for (let i = 0; i < W * H; i++) {
      if (!m[i] || labels[i]) continue;
      const id = comps.length + 1, pix = [];
      labels[i] = id; stack.push(i);
      while (stack.length) {
        const at = stack.pop(), r = Math.floor(at / W), c = at % W;
        pix.push(at);
        for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
          const rr = r + dr, cc = c + dc;
          if (rr < 0 || cc < 0 || rr >= H || cc >= W) continue;
          const j = rr * W + cc;
          if (m[j] && !labels[j]) { labels[j] = id; stack.push(j); }
        }
      }
      comps.push(pix);
    }
    return { labels, comps };
  }

  // ---------------------------------------------------------------- road view or bird's-eye
  // Bird's-eye while the scene asks for a look around, the road view while cruising. Cues, each a reason shown on
  // the page: stopped; a turn coming (the planned path bending within the next 40 m, and slowing for it or sharply:
  // a sweeping ramp or bend taken at speed is not one); a crosswalk ahead within a few seconds, going slowly or
  // slowing for it (not one driven straight over); a close lead in slow traffic; or simply slow. A cue must hold a moment before the view pans down, the
  // road view comes back only after none for a while and at speed, and either view is held a few seconds at least.
  const VIEW = {
    stopped: 3, slow: 12, roadAbove: 18,  // mph
    turnDeg: 20, turnSlow: 22, turnSharp: 45, turnAhead: 40, turnBelow: 45,  // a bend of turnDeg within turnAhead m
    // that the plan slows below turnSlow mph for, or one of turnSharp degrees; and either only below turnBelow mph
    crossSeconds: 2.5, crossMin: 10, crossBelow: 20, crossSlow: 15,  // a crosswalk within that many seconds (at least
    // m) ahead, below crossBelow mph or with the plan slowing below crossSlow mph before it
    queueLead: 15, queueBelow: 25,  // a lead within that many m in traffic below that many mph
    enter: 0.4, leave: 3.0, hold: 3.0,  // seconds
  };

  class ViewChooser {
    constructor(opts) { this.o = Object.assign({}, VIEW, opts); this.target = null; this.since = 0; this.cueFor = 0; this.calmFor = 0; this.why = ''; }

    // the cue in this frame, or '': v in m/s; px, py the plan, pv its speeds; stripes in the world, pose the ego's;
    // lead or null
    cue(v, px, py, pv, stripes, pose, lead) {
      const o = this.o, mph = v * MPH;
      if (mph < o.stopped) return 'stopped';
      if (mph < o.turnBelow) {
        let j = 1; while (j < px.length - 1 && px[j] < o.turnAhead) j++;
        const h0 = Math.atan2(py[1] - py[0], px[1] - px[0] || 1e-3), h1 = Math.atan2(py[j] - py[j - 1], px[j] - px[j - 1] || 1e-3);
        const bend = Math.abs(Math.atan2(Math.sin(h1 - h0), Math.cos(h1 - h0))) * 180 / Math.PI;
        const slowest = Math.min(...pv.slice(0, j + 1)) * MPH;
        if (px[j] - px[0] > 5 && (bend > o.turnSharp || (bend > o.turnDeg && slowest < o.turnSlow))) return 'turn';
      }
      const reach = Math.max(o.crossMin, v * o.crossSeconds);
      let r = 0; while (r < px.length - 1 && px[r] < reach) r++;
      const slowing = Math.min(...pv.slice(0, r + 1)) * MPH < o.crossSlow;
      if ((mph < o.crossBelow || slowing) && stripes.length && pose) {
        const c = Math.cos(pose[2]), s = Math.sin(pose[2]);
        for (const quad of stripes) {
          for (const [wx, wy] of quad) {
            const x = c * (wx - pose[0]) + s * (wy - pose[1]), y = -s * (wx - pose[0]) + c * (wy - pose[1]);
            if (x > 0 && x < reach && Math.abs(y) < 12) return 'crossing';
          }
        }
      }
      if (lead && lead.x < o.queueLead && mph < o.queueBelow) return 'traffic';
      if (mph < o.slow) return 'slow';
      return '';
    }

    // 1 for bird's-eye, 0 for the road view, after this frame's cue
    step(cue, v) {
      const o = this.o, mph = v * MPH;
      this.since += DT;
      this.cueFor = cue ? this.cueFor + DT : 0;
      this.calmFor = cue ? 0 : this.calmFor + DT;
      if (cue) this.why = cue;
      if (this.target === null) { this.target = cue ? 1 : 0; this.since = 0; return this.target; }
      if (this.since < o.hold) return this.target;
      if (this.target === 0 && this.cueFor >= o.enter) { this.target = 1; this.since = 0; }
      else if (this.target === 1 && this.calmFor >= o.leave && mph > o.roadAbove) { this.target = 0; this.since = 0; this.why = ''; }
      return this.target;
    }
  }

  // Where the plan, having braked, starts to speed up again: the comma's own path turns from red to green there (its
  // hue follows the plan's acceleration). The drawn path ends at it, as at a stop sign the car stops before it goes;
  // once the car has passed it the plan no longer starts by braking, and all of it is drawn. Braking is an acceleration
  // of PLAN_BRAKE or less, so a cruising plan's small ups and downs never cut it. px the plan's distances, pa its
  // accelerations; null where it does not brake and then speed up.
  const PLAN_BRAKE = -0.3;  // m/s^2
  function planEnd(px, pa) {
    let braked = false;
    for (let i = 0; i < px.length; i++) {
      if (pa[i] <= PLAN_BRAKE) braked = true;
      else if (braked && pa[i] >= 0) {
        const f = -pa[i - 1] / (pa[i] - pa[i - 1]);  // where the acceleration crosses zero, between the two
        return px[i - 1] + f * (px[i] - px[i - 1]);
      }
    }
    return null;
  }

  // ---------------------------------------------------------------- the scene, frame by frame
  class Engine {
    constructor(opts = {}) {
      this.opts = Object.assign({ panSeconds: 3, view: {} }, opts);
      this.meta = null;
      this.reset();
    }
    reset() {
      // the camera's height is the comma's word (its calibration), never assumed: without it, nothing is placed
      this.cam = this.meta && this.meta.frame && this.meta.height > 0 ? new FrameCamera(this.meta) : null;
      // the wide frame's, turned by the wide camera's angles from the device as the comma calibrated them
      // (wide.from_device): only where the meta has both. Without, the wide frame's segmentations are not used.
      const fixed = this.meta && this.meta.wide && wideFromDevice(this.meta.wide);
      this.wideCam = fixed && this.cam ? new FrameCamera(this.meta, this.meta.wide, rotFromEuler(...fixed)) : null;
      this.smoother = new OutputSmoother(0.3);
      this.trail = new EgoTrail();
      this.roadmap = this.cam ? new RoadMap(this.cam) : null;
      this.pendingSeg = []; this.lastSeg = null;
      this.pendingLines = []; this.stoplines = new StopLines();
      this.leadBars = new CommaLeadBars(); this.leadBraking = {};
      this.stripes = []; this.stripesFrame = -1;
      this.speed = null; this.view = null; this.chooser = new ViewChooser(this.opts.view);
      this.frame = -1; this.state = null;
      this.timing = { model: [], seg: [], segFrames: [], lag: [] };
    }
    onMeta(meta) { this.meta = meta; this.reset(); }

    push(msg) {
      if (msg.kind === 'M') this.onMeta(msg.meta);
      else if (!this.meta) return;
      else if (msg.kind === 'D') this.step(msg);
      else if (msg.kind === 'S') {
        // the narrow frame's in the road view, the wide frame's in bird's-eye; the other is not used (unless the
        // drive has only the narrow frame's: diorama/stopline.py's narrow_only)
        if (!this.meta.seg.narrow_only && !!msg.wide !== (this.chooser.target === 1)) return;
        this.pendingSeg.push(msg); keepLast(this.timing.seg, msg.ms, 50);
      } else if (msg.kind === 'L') {
        this.pendingLines.push(msg);  // both cameras': held in the world, they agree whichever the view
      }
    }

    // One 20 Hz frame of the driving model's outputs, and the newest segmentation that has arrived since the last.
    step(msg) {
      if (!this.meta.drive || !this.meta.drive.length) return null;  // a live server's meta before its first frame
      const k = msg.frame;
      if (this.frame >= 0 && (k <= this.frame || k > this.frame + 40)) this.reset();  // a seek, a restart, or a long gap
      const raw = parseDrive(msg.values, this.meta.drive);
      const out = this.smoother.smooth(raw);
      keepLast(this.timing.model, msg.ms, 50);
      this.trail.add(k, raw.pose);
      this.frame = k;

      const px = [], pz = [], py = [], pv = [];
      for (let i = 0; i < IDX_N; i++) {
        px.push(plan(out.plan, i, 0)); py.push(plan(out.plan, i, 1)); pz.push(plan(out.plan, i, 2)); pv.push(plan(out.plan, i, 3));
      }
      const moving = px[IDX_N - 1] > px[0] + 1;
      const zpath = moving ? (x) => interp(x, px, pz) : () => 0;
      const laneY = X_IDXS.map((_, i) => (lane(out.lane_lines, 1, i, 0) + lane(out.lane_lines, 2, i, 0)) / 2);
      const slopes = laneY.slice(1).map((y, i) => (y - laneY[i]) / (X_IDXS[i + 1] - X_IDXS[i]));
      const laneHeading = (x) => Math.atan(interp(x, X_IDXS.slice(1), slopes));

      // the newest segmentation, into the road map from the pose it was seen at
      const segMsg = this.pendingSeg.length ? this.pendingSeg[this.pendingSeg.length - 1] : null;
      this.pendingSeg = [];
      const segCam = segMsg && (segMsg.wide ? this.wideCam : this.cam);
      if (segMsg && this.trail.pose(segMsg.frame) && segCam) {
        keepLast(this.timing.segFrames, segMsg.frame, 20);
        keepLast(this.timing.lag, k - segMsg.frame, 20);
        const seg = readSeg(segMsg, this.meta);
        this.roadmap.observe(this.trail.pose(segMsg.frame), seg, zpath, segCam);
        this.lastSeg = { frame: segMsg.frame, seg };
      }

      // crosswalk stripes, re-cut as the map fills in
      const pNow = this.trail.pose(k);
      if (segMsg && k - this.stripesFrame >= 6) {
        this.stripes = this.roadmap.crosswalks(pNow, laneHeading);
        this.stripesFrame = k;
      }
      // the stop-line model's lines, fitted to the ego lane, drawn with the stripes
      const bounds = this.pendingLines.length ? egoLane(out) : null;
      for (const m of this.pendingLines) {
        const cam = m.wide ? this.wideCam : this.cam, seen = this.trail.pose(m.frame);
        if (cam && seen) this.stoplines.observe(k, seen, m, cam, this.cam.height, this.trail, pNow, bounds);
      }
      this.pendingLines = [];
      const stopLine = this.stoplines.quads(k, pNow, this.trail)[0] || null;
      const stripes = stopLine ? this.stripes.concat([stopLine]) : this.stripes;

      // The leads: a car wherever the comma's display puts a lead bar, smoothed and faded as the display does, its rear
      // at the bar's far edge, facing along the ego lane there. Each braking when the model's predicted acceleration
      // for it says so, as last known while its bar fades out. `lead`, the first of them, is what the HUD and the view go by.
      const model = commaModel(raw);
      if (model && this.meta.frame) {
        const f = this.meta.frame;
        this.leadBars.advance(model, commaTransform(f), f.w, f.h, this.meta.height || COMMA.HEIGHT, COMMA.RENDERS);
      }
      let lead = null;
      const leads = [];
      for (const bar of this.leadBars.leads) {
        if (bar.index < raw.lead_prob.length && raw.lead_prob[bar.index] > COMMA.PROB) this.leadBraking[bar.index] = leadAt(raw.lead, bar.index, 0, 3) < -0.8;
        const car = { x: bar.x, y: bar.y, heading: laneHeading(bar.x), braking: !!this.leadBraking[bar.index] };
        leads.push(car);
        if (bar.index === 0) lead = car;
      }

      // road view or bird's-eye (ViewChooser), panning between the two
      const v = plan(out.plan, 0, 3);
      this.speed = this.speed === null ? v : 0.9 * this.speed + 0.1 * v;
      const target = this.chooser.step(this.chooser.cue(this.speed, px, py, pv, stripes, pNow, lead), this.speed);
      const stepS = DT / this.opts.panSeconds;
      this.view = this.view === null ? target : clamp(this.view + Math.sign(target - this.view) * stepS, Math.min(this.view, target), Math.max(this.view, target));

      this.state = {
        frame: k, seconds: msg.seconds, out, px, py, pz, pv, zpath, lead, leads,
        planEnd: planEnd(px, Array.from({ length: IDX_N }, (_, i) => plan(out.plan, i, 6))),
        pose: pNow, stripes, stopLine, view: smoothstep(0, 1, this.view), viewWhy: target ? this.chooser.why : '',
        hud: { v: this.speed, a: plan(out.plan, 0, 6), lead: lead ? lead.x : null },
      };
      return this.state;
    }
  }

  // ---- the comma 4 display's drawing of the driving model, as CommaUI.swift ports it from zoompilot's model_renderer.py
  // and CommaUIParityTests holds both to it: engaged, with openpilot longitudinal, no torque, the model's own leads.
  const COMMA = { CLIP: 500, MIN_DRAW: 10, MAX_DRAW: 100, BAR_LENGTH: 12, BAR_WIDTH: 1.8, RADAR: 1.52, HEIGHT: 1.22, PROB: 0.5, OPACITY: 0.8, RENDERS: 3 };

  // modelV2's fields the display reads, as modeld fills them; null without all of them
  function commaModel(o) {
    if (!o.plan || o.plan.length < IDX_N * PLAN_WIDTH || !o.lane_lines || o.lane_lines.length < 4 * IDX_N * 2 || !o.road_edges || o.road_edges.length < 2 * IDX_N * 2
      || !o.road_edges_stds || o.road_edges_stds.length < 2 * IDX_N * 2 || !o.lane_lines_prob || o.lane_lines_prob.length < 8
      || !o.lead_prob || o.lead_prob.length < 2 || !o.lead || o.lead.length < 2 * LEAD_LEN * LEAD_WIDTH) return null;
    const range = (f) => Array.from({ length: IDX_N }, (_, i) => f(i));
    const column = (c) => range((i) => plan(o.plan, i, c));
    const line = (l, i) => ({ x: X_IDXS, y: range((j) => lane(l, i, j, 0)), z: range((j) => lane(l, i, j, 1)) });
    return {
      position: { x: column(0), y: column(1), z: column(2) }, accelerationX: column(6),
      laneLines: [0, 1, 2, 3].map((i) => line(o.lane_lines, i)), laneLineProbs: [0, 1, 2, 3].map((i) => o.lane_lines_prob[2 * i + 1]),
      roadEdges: [0, 1].map((i) => line(o.road_edges, i)), roadEdgeStds: [0, 1].map((i) => lane(o.road_edges_stds, i, 0, 0)),
      leads: [0, 1].map((i) => ({ prob: o.lead_prob[i], x: leadAt(o.lead, i, 0, 0), y: leadAt(o.lead, i, 0, 1) })),
    };
  }
  // road point to pixel in the frame the models see: already calibrated, so K view_from_device
  const commaTransform = (f) => m3mul([f.f, 0, f.cx, 0, f.f, f.cy, 0, 0, 1], VIEW_FROM_DEVICE);

  function commaProject(T, x, y, z) { return [T[0] * x + T[1] * y + T[2] * z, T[3] * x + T[4] * y + T[5] * z, T[6] * x + T[7] * y + T[8] * z]; }

  function commaRibbon(T, W, H, line, yOff, zOff, maxIdx, allowInvert) {
    const m = COMMA.CLIP, inClip = (p) => p[0] >= -m && p[0] <= W + m && p[1] >= -m && p[1] <= H + m;
    let left = [], right = [];
    for (let i = 0; i <= Math.min(maxIdx, line.x.length - 1); i++) {
      if (!(line.x[i] >= 0)) continue;
      const l = commaProject(T, line.x[i], line.y[i] - yOff, line.z[i] + zOff), r = commaProject(T, line.x[i], line.y[i] + yOff, line.z[i] + zOff);
      if (Math.abs(l[2]) < 1e-6 || Math.abs(r[2]) < 1e-6) continue;
      const a = [l[0] / l[2], l[1] / l[2]], b = [r[0] / r[2], r[1] / r[2]];
      if (!inClip(a) || !inClip(b)) continue;
      left.push(a); right.push(b);
    }
    // a hill's far side comes back down the picture: only points higher up than all before them stay
    if (!allowInvert && left.length > 1) {
      let lowest = Infinity;
      const keep = [];
      left.forEach((p, i) => { lowest = Math.min(lowest, p[1]); if (p[1] === lowest) keep.push(i); });
      left = keep.map((i) => left[i]); right = keep.map((i) => right[i]);
    }
    return left.concat(right.reverse());
  }

  function commaLeadBar(T, lane_, d, yRel, roadHeight) {
    const xs = [d, d - Math.min(6, 0.25 * d)], at = interp(d, lane_.x, lane_.y);
    const ys = xs.map((x) => interp(x, lane_.x, lane_.y) - at - yRel), zs = xs.map((x) => interp(x, lane_.x, lane_.z) + roadHeight), half = COMMA.BAR_WIDTH / 2;
    const bar = [[xs[0], ys[0] + half, zs[0]], [xs[1], ys[1] + half, zs[1]], [xs[1], ys[1] - half, zs[1]], [xs[0], ys[0] - half, zs[0]]].map(([x, y, z]) => {
      const p = commaProject(T, x, y, z);
      return [p[0] / p[2], p[1] / p[2]];
    });
    const mx = (bar[1][0] + bar[2][0]) / 2 - (bar[0][0] + bar[3][0]) / 2, my = (bar[1][1] + bar[2][1]) / 2 - (bar[0][1] + bar[3][1]) / 2, length = Math.hypot(mx, my);
    if (length > 0) {
      const k = clamp(length, 3, COMMA.BAR_LENGTH) / length;
      bar[1] = [bar[0][0] + (bar[1][0] - bar[0][0]) * k, bar[0][1] + (bar[1][1] - bar[0][1]) * k];
      bar[2] = [bar[3][0] + (bar[2][0] - bar[3][0]) * k, bar[3][1] + (bar[2][1] - bar[3][1]) * k];
    }
    return bar;
  }

  // the two leads the display considers, closer than 3 m being the same vehicle
  function commaLeads(m) {
    const leads = m.leads.slice(0, 2).map((l) => ({ present: l.prob > COMMA.PROB, d: l.x, y: -l.y }));
    if (leads.length === 2 && leads[0].present && Math.abs(leads[1].d - leads[0].d) < 3) leads[1] = { present: false, d: 0, y: 0 };
    return leads;
  }

  class CommaFilter {
    constructor(rc, initialized) { const dt = 1 / 60; this.alpha = dt / (rc + dt); this.x = 0; this.initialized = initialized; }
    update(v) { this.x = this.initialized ? (1 - this.alpha) * this.x + this.alpha * v : v; this.initialized = true; return this.x; }
  }

  // The display's lead bars over time: each model frame is COMMA.RENDERS of its frames, the bars' distance, offset and
  // opacity lagging as its FirstOrderFilters do. A lead that goes fades out where its bar last was.
  class CommaLeadBars {
    constructor() {
      this.vehicles = [0, 1].map(() => ({ d: new CommaFilter(0.2, false), y: new CommaFilter(0.2, false), fade: new CommaFilter(0.1, true), bar: [] }));
    }
    advance(m, T, W, H, roadHeight, renders) {
      const leads = commaLeads(m);
      for (let r = 0; r < renders; r++) {
        this.vehicles.forEach((v, i) => {
          if (i >= leads.length) return;
          const lead = leads[i], visible = lead.present && lead.d < COMMA.MAX_DRAW && m.position.x.length > 0;
          // snap to a new vehicle instead of sliding over
          if (!visible || Math.abs(lead.y - v.y.x) > 3) v.d.initialized = v.y.initialized = false;
          v.fade.update(visible ? COMMA.OPACITY : 0);
          if (visible) v.bar = commaLeadBar(T, m.position, v.d.update(lead.d), v.y.update(lead.y), roadHeight);
        });
      }
    }
    alpha(v) { return Math.floor(clamp(255 * v.fade.x, 0, 255)); }
    // each bar while any of it shows, the far edge's middle on the road: where that lead's rear is
    get leads() {
      return this.vehicles.flatMap((v, index) => (v.bar.length >= 3 && this.alpha(v) > 0 ? [{ index, x: v.d.x, y: -v.y.x }] : []));
    }
    get polygons() {
      return this.vehicles.filter((v) => v.bar.length >= 3 && this.alpha(v) > 0).map((v) => ({ points: v.bar, color: [255, 255, 255, this.alpha(v)] }));
    }
  }

  function commaHsla(h, s, l, a) {
    const byte = (v) => Math.floor(clamp(v * 255, 0, 255));
    if (s === 0) return [byte(l), byte(l), byte(l), byte(a)];
    const m2 = l <= 0.5 ? l * (1 + s) : l + s - l * s, m1 = 2 * l - m2;
    const channel = (hue) => {
      const h1 = hue - Math.floor(hue);
      if (h1 < 1 / 6) return m1 + (m2 - m1) * h1 * 6;
      if (h1 < 0.5) return m2;
      if (h1 < 2 / 3) return m1 + (m2 - m1) * (2 / 3 - h1) * 6;
      return m1;
    };
    return [byte(channel(h + 1 / 3)), byte(channel(h)), byte(channel(h - 1 / 3)), byte(a)];
  }

  // the path's color, climbing the picture with its acceleration
  function commaPathFill(path, acceleration, H) {
    const maxLen = Math.min(Math.floor(path.length / 2), acceleration.length), colors = [], stops = [];
    let i = 0;
    while (i < maxLen) {
      const y = path[i][1];
      if (y < 0 || y > H) { i++; continue; }
      const point = 1 - y / H, a = acceleration[i];
      const hue = clamp(60 + a * 35, 0, 120), saturation = Math.min(Math.abs(a * 1.5), 1);
      const lightness = interp(saturation, [0, 1], [0.95, 0.62]), alpha = interp(point, [0.75 / 2, 0.75], [0.4, 0]);
      stops.push(point);
      colors.push(commaHsla(hue / 360, saturation, lightness, alpha));
      i += 1 + (i + 2 < maxLen ? 1 : 0);
    }
    return colors.length > 1 ? { colors, stops } : { color: [255, 255, 255, 30] };
  }

  // the display's shader at t up the picture: [r, g, b, a] from 0 to 1, straight alpha, mixed between stops
  function commaColor(t, colors, stops) {
    const value = (c) => c.map((v) => v / 255), s = stops.map((v) => clamp(v, 0, 1)), n = colors.length;
    if (t <= s[0]) return value(colors[0]);
    if (t >= s[n - 1]) return value(colors[n - 1]);
    for (let i = 0; i < n - 1; i++) {
      if (t >= s[i] && t <= s[i + 1]) {
        const k = (t - s[i]) / Math.max(s[i + 1] - s[i], 1e-6), a = value(colors[i]), b = value(colors[i + 1]);
        return a.map((v, c) => v + (b[c] - v) * k);
      }
    }
    return value(colors[n - 1]);
  }

  // What the display draws over a frame of the models', in drawing order: { points, color } or { points, colors, stops }.
  // leadBars: its bars, or null for each lead's settled at this frame.
  function commaPolygons(m, T, W, H, roadHeight, leadBars = null) {
    const out = [], lastX = m.position.x[m.position.x.length - 1];
    if (lastX === undefined) return out;
    const add = (points, fill) => { if (!(fill.color && fill.color[3] === 0) && points.length >= 3) out.push({ points, ...fill }); };
    const lengthIdx = (xs, d) => { for (let i = xs.length - 1; i >= 0; i--) if (xs[i] <= d) return i; return 0; };
    const laneColor = (prob, adjacent) => {
      const a = Math.floor(clamp(prob, 0, 0.7) * 255);
      return { color: adjacent ? [0, 255, 64, a] : [255, 255, 255, a] };
    };
    let maxDistance = clamp(lastX, COMMA.MIN_DRAW, COMMA.MAX_DRAW);
    const maxIdx = lengthIdx(m.laneLines[0].x, maxDistance);
    // the factor stays at 0.16 from the second line on: the fourth line and the road edges have it too
    let widthFactor = 0.12;
    m.laneLines.forEach((line, i) => {
      if (i === 1 || i === 2) widthFactor = 0.16;
      add(commaRibbon(T, W, H, line, widthFactor * m.laneLineProbs[i], 0, maxIdx, true), laneColor(m.laneLineProbs[i], i === 1 || i === 2));
    });
    m.roadEdges.forEach((edge, i) => add(commaRibbon(T, W, H, edge, widthFactor, 0, maxIdx, true), laneColor(1 - m.roadEdgeStds[i], m.laneLineProbs[i + 1] < 0.25)));
    // radarState.leadOne, as radard makes it from the model's lead
    if (m.leads.length && m.leads[0].prob > COMMA.PROB) {
      const d = (m.leads[0].x - COMMA.RADAR) * 2;
      maxDistance = clamp(d - Math.min(d * 0.35, 10), 0, maxDistance);
    }
    const path = commaRibbon(T, W, H, m.position, 0.9, roadHeight, lengthIdx(m.position.x, maxDistance), false);
    add(path, commaPathFill(path, m.accelerationX, H));
    if (leadBars) return out.concat(leadBars);
    for (const lead of commaLeads(m)) {
      if (lead.present && lead.d < COMMA.MAX_DRAW) add(commaLeadBar(T, m.position, lead.d, lead.y, roadHeight), { color: [255, 255, 255, Math.floor(255 * COMMA.OPACITY)] });
    }
    return out;
  }

  function keepLast(arr, v, n) { arr.push(v); if (arr.length > n) arr.shift(); }

  // A frame's scene as plain JSON, for the parity tests: what the app's DioramaScene.State must match. The path is
  // the drawn plan (px, py, pz along it, pv its speeds; planEnd where it is cut, at the braking's end), pose the car in the world (x, y, yaw), stopLine the held
  // stop line's world quad or null, stripes the crosswalks and stop line as world quads, view 0 road .. 1 bird's-eye.
  function snapshot(s) {
    if (!s) return null;
    const a = (v) => Array.from(v || []);
    return {
      frame: s.frame, seconds: s.seconds, px: a(s.px), py: a(s.py), pz: a(s.pz), pv: a(s.pv), pose: a(s.pose),
      lead: s.lead && { x: s.lead.x, y: s.lead.y, heading: s.lead.heading, braking: s.lead.braking },
      leads: s.leads.map((l) => ({ x: l.x, y: l.y, heading: l.heading, braking: l.braking })),
      stopLine: s.stopLine, stripes: s.stripes, view: s.view, viewWhy: s.viewWhy, planEnd: s.planEnd ?? null,
      hud: { v: s.hud.v, a: s.hud.a, lead: s.hud.lead },
    };
  }

  const api = {
    FPS, DT, MPH, IDX_N, X_IDXS, EGO_FRONT, EGO_REAR,
    clamp, smoothstep, interp, percentile, m3mul, m3inv, m3vec, VIEW_FROM_DEVICE,
    decode, readFile, parseDrive, plan, lane, leadAt, FrameCamera, RoadMap, Engine, ViewChooser, VIEW,
    readSeg, morph, runs, components, snapshot, egoLane, fitToLane, planEnd, PLAN_BRAKE,
    COMMA, commaModel, commaTransform, commaPolygons, commaColor, CommaLeadBars,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.DioramaEngine = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
