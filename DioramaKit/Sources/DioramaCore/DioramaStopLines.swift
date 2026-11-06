import Foundation

/// One stop line as the stop-line model saw it: its score and two end points in the narrow frame's pixels.
public struct DioramaStopLine: Sendable, Equatable {
  public let score: Double, x1: Double, y1: Double, x2: Double, y2: Double
  public init(score: Double, x1: Double, y1: Double, x2: Double, y2: Double) {
    (self.score, self.x1, self.y1, self.x2, self.y2) = (score, x1, y1, x2, y2)
  }
}

/// The stop lines held in the world, as diorama-engine.js's StopLines. Each sighting is first fitted to the ego lane
/// (egoLane, fitToLane): near, the frame's edges cut the model's line short, and farther out it runs into the next
/// lanes, so its ends are moved along it to the lane's boundaries. The model gives at most one line a frame, the
/// ego lane's, so a sighting whose middle is within `same` m of a held line's along the road (and `side` m across), as
/// the car sees it then, is that line, whatever its angle: the held line's middle, heading and length move 0.3 of the
/// way to it. Held lines that come that close merge, the more seen keeping its place. Only the nearest line ahead seen
/// `seen` times is drawn. A line is dropped once behind the car, or `keep` frames after it was last seen unless within
/// `near` m ahead (hidden by the hood).
struct DioramaStopLines {
  static let same = 3.0, side = 3.0, keep = 30, maxAhead = 50.0, width = 0.4, seen = 3, near = 6.0

  struct Held {
    var m: SIMD2<Double>, ang: Double, len: Double, last: Int, hits: Int
  }

  private(set) var held: [Held] = []

  typealias Pose = (x: Double, y: Double, yaw: Double)

  static func egoToWorld(_ p: Pose, _ x: Double, _ y: Double) -> SIMD2<Double> {
    let c = cos(p.yaw), s = sin(p.yaw)
    return [p.x + c * x - s * y, p.y + s * x + c * y]
  }

  static func worldToEgo(_ p: Pose, _ w: SIMD2<Double>) -> SIMD2<Double> {
    let c = cos(p.yaw), s = sin(p.yaw), dx = w.x - p.x, dy = w.y - p.y
    return [c * dx + s * dy, -s * dx + c * dy]
  }

  /// The ego lane's two boundaries this frame, left then right, each 33 road points (X_IDXS, y): lane line 1 or 2, or the
  /// road edge where it lies inside the line. No probability gate: at an intersection the lines' probabilities fall to
  /// about 0.05, but where the model puts them still bounds the lane (the training labels were cut the same way). Nil
  /// without lane lines or road edges.
  static func egoLane(_ out: DioramaOutputs) -> [[SIMD2<Double>]]? {
    let n = Diorama.idxN
    guard out.laneLines.count >= 4 * n * 2, out.roadEdges.count >= 2 * n * 2 else { return nil }
    return (0..<2).map { side in
      (0..<n).map { i in
        let ly = DioramaOutputs.line(out.laneLines, 1 + side, i, 0), ey = DioramaOutputs.line(out.roadEdges, side, i, 0)
        let inside = side == 1 ? ey < ly : ey > ly  // left: the edge right of the line; right: left of it
        return SIMD2(Diorama.xIdxs[i], inside ? ey : ly)
      }
    }
  }

  /// The line A-B (road points, in the frame the boundaries are in) fitted to the lane: moved along its own extension to
  /// where it crosses the two boundaries, or nil where it does not cross both. No clamping: it can grow or shrink.
  static func fitToLane(_ A: SIMD2<Double>, _ B: SIMD2<Double>, _ bounds: [[SIMD2<Double>]]) -> (SIMD2<Double>, SIMD2<Double>)? {
    let d = B - A, dd = d.x * d.x + d.y * d.y
    guard dd >= 1e-6 else { return nil }
    let ts: [Double?] = bounds.map { c in
      let s = c.map { p in (p.x - A.x) * -d.y + (p.y - A.y) * d.x }  // which side of the line each point is
      for i in 0..<(c.count - 1) where s[i] == 0 || s[i] * s[i + 1] < 0 {
        let f = s[i] / (s[i] - s[i + 1])
        let p = c[i] + f * (c[i + 1] - c[i])
        return ((p.x - A.x) * d.x + (p.y - A.y) * d.y) / dd
      }
      return nil
    }
    guard let t0 = ts[0], let t1 = ts[1], t0 != t1 else { return nil }
    let (lo, hi) = (min(t0, t1), max(t0, t1))
    return (A + lo * d, A + hi * d)
  }

  /// d less the nearest whole half turn, rounding as Math.round does.
  static func halfTurn(_ d: Double) -> Double { d - .pi * (d / .pi + 0.5).rounded(.down) }

  static func move(_ h: inout Held, _ m: SIMD2<Double>, _ ang: Double, _ len: Double, _ w: Double) {
    h.m += w * (m - h.m)
    h.ang += w * halfTurn(ang - h.ang)
    h.len += w * (len - h.len)
  }

  /// The lines seen from `pose` (the frame they were seen in) through the narrow frame's camera, counted as seen at frame
  /// `k`. With `bounds` (egoLane in frame k's road frame, the car at `now`), each is first fitted to the ego lane.
  mutating func observe(
    _ k: Int, _ pose: Pose, _ lines: [DioramaStopLine], _ cam: DioramaFrameCamera, now: Pose? = nil, bounds: [[SIMD2<Double>]]? = nil
  ) {
    func isSame(_ h: Held, _ x: Double, _ y: Double) -> Bool {
      let e = Self.worldToEgo(pose, h.m)
      return abs(e.x - x) < Self.same && abs(e.y - y) < Self.side
    }
    for l in lines {
      guard let A = cam.groundHit(l.x1, l.y1, cam.height), let B = cam.groundHit(l.x2, l.y2, cam.height), (A.0 + B.0) / 2 <= Self.maxAhead
      else { continue }
      var a = Self.egoToWorld(pose, A.0, A.1), b = Self.egoToWorld(pose, B.0, B.1)
      if let bounds, let now, let fit = Self.fitToLane(Self.worldToEgo(now, a), Self.worldToEgo(now, b), bounds) {
        // the boundaries are in this frame's road frame, the sighting in its own frame's
        (a, b) = (Self.egoToWorld(now, fit.0.x, fit.0.y), Self.egoToWorld(now, fit.1.x, fit.1.y))
      }
      let m = (a + b) / 2, ang = atan2(b.y - a.y, b.x - a.x), len = ((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y)).squareRoot()
      let mid = Self.worldToEgo(pose, m), x = mid.x, y = mid.y
      // the most seen of the held lines it matches, the first of them on a tie (a stable sort, as the page's)
      var best: Int? = nil
      for i in held.indices where isSame(held[i], x, y) {
        if best == nil || held[i].hits > held[best!].hits { best = i }
      }
      guard let i = best else {
        held.append(Held(m: m, ang: ang, len: len, last: k, hits: 1))
        continue
      }
      Self.move(&held[i], m, ang, len, 0.3)
      held[i].last = k
      held[i].hits += 1
    }
    held = stableSorted(held) { $0.hits > $1.hits }
    var i = 0
    while i < held.count {
      let e = Self.worldToEgo(pose, held[i].m)
      var j = held.count - 1
      while j > i {
        let g = held[j]
        if isSame(g, e.x, e.y) {
          Self.move(&held[i], g.m, g.ang, g.len, Double(g.hits) / Double(held[i].hits + g.hits))
          held[i].hits += g.hits
          held[i].last = max(held[i].last, g.last)
          held.remove(at: j)
        }
        j -= 1
      }
      i += 1
    }
  }

  /// The line to draw at frame `k`, if any, as four corners in the world: the nearest ahead seen often enough.
  mutating func quad(_ k: Int, _ pose: Pose) -> [SIMD2<Double>]? {
    let ahead = { (h: Held) in Self.worldToEgo(pose, h.m).x }
    held = held.filter { h in
      let x = ahead(h)
      return x > -3 && (k - h.last <= Self.keep || (h.hits >= Self.seen && x < Self.near))
    }
    guard let h = stableSorted(held.filter { $0.hits >= Self.seen }, { ahead($0) < ahead($1) }).first else { return nil }
    let u = SIMD2(cos(h.ang), sin(h.ang)) * (h.len / 2), n = SIMD2(-sin(h.ang), cos(h.ang)) * (Self.width / 2)
    let a = h.m - u, b = h.m + u
    return [a + n, b + n, b - n, a - n]
  }
}

/// Array.sort is not stable; the page's (JavaScript's) is.
private func stableSorted<T>(_ a: [T], _ less: (T, T) -> Bool) -> [T] {
  a.enumerated().sorted { less($0.element, $1.element) || (!less($1.element, $0.element) && $0.offset < $1.offset) }.map(\.element)
}
