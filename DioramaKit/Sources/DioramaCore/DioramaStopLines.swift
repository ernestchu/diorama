import Foundation

/// One stop line as the stop-line model saw it: its score and two end points in the narrow frame's pixels.
public struct DioramaStopLine: Sendable, Equatable {
  public let score: Double, x1: Double, y1: Double, x2: Double, y2: Double
  public init(score: Double, x1: Double, y1: Double, x2: Double, y2: Double) {
    (self.score, self.x1, self.y1, self.x2, self.y2) = (score, x1, y1, x2, y2)
  }
}

/// The stop lines held in the world, as diorama-engine.js's StopLines. The model gives at most one line a frame, the
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

  /// d less the nearest whole half turn, rounding as Math.round does.
  static func halfTurn(_ d: Double) -> Double { d - .pi * (d / .pi + 0.5).rounded(.down) }

  static func move(_ h: inout Held, _ m: SIMD2<Double>, _ ang: Double, _ len: Double, _ w: Double) {
    h.m += w * (m - h.m)
    h.ang += w * halfTurn(ang - h.ang)
    h.len += w * (len - h.len)
  }

  /// The lines seen in frame `k`, from that frame's pose, through the narrow frame's camera.
  mutating func observe(_ k: Int, _ pose: Pose, _ lines: [DioramaStopLine], _ cam: DioramaFrameCamera) {
    func isSame(_ h: Held, _ x: Double, _ y: Double) -> Bool {
      let e = Self.worldToEgo(pose, h.m)
      return abs(e.x - x) < Self.same && abs(e.y - y) < Self.side
    }
    for l in lines {
      guard let A = cam.groundHit(l.x1, l.y1, cam.height), let B = cam.groundHit(l.x2, l.y2, cam.height), (A.0 + B.0) / 2 <= Self.maxAhead
      else { continue }
      let a = Self.egoToWorld(pose, A.0, A.1), b = Self.egoToWorld(pose, B.0, B.1)
      let m = (a + b) / 2, ang = atan2(b.y - a.y, b.x - a.x), len = ((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y)).squareRoot()
      let x = (A.0 + B.0) / 2, y = (A.1 + B.1) / 2
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
