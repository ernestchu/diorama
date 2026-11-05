// diorama-engine.js's StopLines and the helpers it uses (the diorama-stopline branch), for the Swift port's parity test.
const trail = {
  egoToWorld(p, x, y) { const c = Math.cos(p[2]), s = Math.sin(p[2]); return [p[0] + c * x - s * y, p[1] + s * x + c * y]; },
  worldToEgo(p, wx, wy) { const c = Math.cos(p[2]), s = Math.sin(p[2]), dx = wx - p[0], dy = wy - p[1]; return [c * dx + s * dy, -s * dx + c * dy]; },
};
function makeCam(f, cx, cy) {
  // P = K VIEW_FROM_DEVICE, VIEW_FROM_DEVICE = [[0,1,0],[0,0,1],[1,0,0]]
  const P = [[cx, f, 0], [cy, 0, f], [1, 0, 0]];
  const det = (m) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const d = det(P);
  const inv = [
    [(P[1][1] * P[2][2] - P[1][2] * P[2][1]) / d, (P[0][2] * P[2][1] - P[0][1] * P[2][2]) / d, (P[0][1] * P[1][2] - P[0][2] * P[1][1]) / d],
    [(P[1][2] * P[2][0] - P[1][0] * P[2][2]) / d, (P[0][0] * P[2][2] - P[0][2] * P[2][0]) / d, (P[0][2] * P[1][0] - P[0][0] * P[1][2]) / d],
    [(P[1][0] * P[2][1] - P[1][1] * P[2][0]) / d, (P[0][1] * P[2][0] - P[0][0] * P[2][1]) / d, (P[0][0] * P[1][1] - P[0][1] * P[1][0]) / d],
  ];
  return {
    groundHit(u, v, gz) {
      const d = inv.map((r) => r[0] * u + r[1] * v + r[2]);
      if (!(d[2] > 1e-6)) return null;
      const t = gz / d[2];
      return t > 0 ? [d[0] * t, d[1] * t, d[2] * t] : null;
    },
  };
}
const STOP_SAME = 3, STOP_SIDE = 3, STOP_KEEP = 30, STOP_MAX = 50, STOP_WIDTH = 0.4, STOP_SEEN = 3, STOP_NEAR = 6;
const halfTurn = (d) => d - Math.PI * Math.round(d / Math.PI);
function stopMove(h, m, ang, len, w) {
  h.m = [h.m[0] + w * (m[0] - h.m[0]), h.m[1] + w * (m[1] - h.m[1])];
  h.ang = h.ang + w * halfTurn(ang - h.ang);
  h.len = h.len + w * (len - h.len);
}
class StopLines {
  constructor() { this.held = []; }
  observe(k, pose, msg, cam, road, trail) {
    const same = (h, x, y) => {
      const [hx, hy] = trail.worldToEgo(pose, h.m[0], h.m[1]);
      return Math.abs(hx - x) < STOP_SAME && Math.abs(hy - y) < STOP_SIDE;
    };
    for (const l of msg.lines) {
      const A = cam.groundHit(l.x1, l.y1, road), B = cam.groundHit(l.x2, l.y2, road);
      if (!A || !B || (A[0] + B[0]) / 2 > STOP_MAX) continue;
      const a = trail.egoToWorld(pose, A[0], A[1]), b = trail.egoToWorld(pose, B[0], B[1]);
      const m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]), x = (A[0] + B[0]) / 2, y = (A[1] + B[1]) / 2;
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
// steps: [{k, pose: [x, y, yaw], lines: [{score, x1, y1, x2, y2}]}] -> the quad (or null) after each
function run(steps, f, cx, cy, road) {
  const cam = makeCam(f, cx, cy), s = new StopLines();
  return steps.map((st) => {
    s.observe(st.k, st.pose, { lines: st.lines }, cam, road, trail);
    const q = s.quads(st.k, st.pose, trail);
    return q.length ? q[0] : null;
  });
}
