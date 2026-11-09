import Foundation

/// The driving model's prediction as the comma 4's own display draws it: ModelRenderer in zoompilot's
/// selfdrive/ui/mici/onroad/model_renderer.py at `pinnedCommit`, ported rule for rule, its quirks too, and held to it
/// by CommaUIParityTests against what the file itself draws (sim/comma_ui_parity.py). These colors are the comma's,
/// and have nothing to do with DioramaRenderer's.
///
/// The link carries the model's outputs and nothing else of the car's state, so it is drawn as the comma draws it
/// engaged, with openpilot longitudinal control, no steering torque and no camera offset. The leads are the model's
/// own, as radard makes them on a car without radar. The lead bar is smoothed as the display smooths it (LeadBars). The
/// lane-centering line (blendful), the confidence ball and the badges are not drawn.
public enum CommaUI {
  /// The commit of https://github.com/ernestchu/zoompilot/tree/ernestchu/diorama this matches.
  public static let pinnedCommit = "2ef7f8522f35a91e017b102673ec31fb9f1cebe7"

  static let clipMargin = 500.0
  static let minDrawDistance = 10.0, maxDrawDistance = 100.0
  static let leadBarLength = 12.0, leadBarWidth = 1.8  // px, m
  static let radarToCamera = 1.52
  /// calibrationd's HEIGHT_INIT: the camera's height before the comma has calibrated one.
  static let heightInit = 1.22
  /// radard's and the display's: a lead the model is at least this sure of is there.
  public static let leadProb = 0.5
  /// UIStatus.ENGAGED's lane line, and the lead bar's opacity while engaged.
  static let engaged = (r: UInt8(0), g: UInt8(255), b: UInt8(64)), leadOpacity = 0.8

  public struct RGBA: Equatable, Sendable {
    public let r: UInt8, g: UInt8, b: UInt8, a: UInt8
    public init(r: UInt8, g: UInt8, b: UInt8, a: UInt8) { (self.r, self.g, self.b, self.a) = (r, g, b, a) }
  }

  public enum Fill: Equatable, Sendable {
    case solid(RGBA)
    /// Up the picture: stop 0 is its bottom edge and 1 its top, as the display's shader takes them.
    case gradient(colors: [RGBA], stops: [Double])
  }

  /// A ribbon: its left edge from near to far, then its right edge back.
  public struct Polygon: Equatable, Sendable {
    public let points: [SIMD2<Double>]
    public let fill: Fill
  }

  public struct Line: Sendable {
    public var x: [Double], y: [Double], z: [Double]
    public init(x: [Double], y: [Double], z: [Double]) { (self.x, self.y, self.z) = (x, y, z) }
  }

  public struct Lead: Sendable {
    public var prob: Double, x: Double, y: Double
  }

  /// The fields of modelV2 the display reads.
  public struct Model: Sendable {
    public var position: Line
    public var accelerationX: [Double]
    public var laneLines: [Line], laneLineProbs: [Double]
    public var roadEdges: [Line], roadEdgeStds: [Double]
    public var leads: [Lead]

    /// As modeld fills modelV2 from the heads (fill_model_msg.py); nil without all of them.
    public init?(_ out: DioramaOutputs) {
      let n = Diorama.idxN
      guard out.plan.count >= n * Diorama.planWidth, out.laneLines.count >= 4 * n * 2, out.roadEdges.count >= 2 * n * 2,
        out.roadEdgesStds.count >= 2 * n * 2, out.laneLinesProb.count >= 8,
        out.leadProb.count >= 2, out.lead.count >= 2 * Diorama.leadLen * Diorama.leadWidth
      else { return nil }
      func column(_ c: Int) -> [Double] { (0..<n).map { out.plan($0, c) } }
      func line(_ l: [Double], _ i: Int) -> Line {
        Line(x: Diorama.xIdxs, y: (0..<n).map { DioramaOutputs.line(l, i, $0, 0) }, z: (0..<n).map { DioramaOutputs.line(l, i, $0, 1) })
      }
      position = Line(x: column(0), y: column(1), z: column(2))
      accelerationX = column(6)
      laneLines = (0..<4).map { line(out.laneLines, $0) }
      laneLineProbs = (0..<4).map { out.laneLinesProb[2 * $0 + 1] }
      roadEdges = (0..<2).map { line(out.roadEdges, $0) }
      roadEdgeStds = (0..<2).map { DioramaOutputs.line(out.roadEdgesStds, $0, 0, 0) }
      leads = (0..<2).map { Lead(prob: out.leadProb[$0], x: out.lead($0, 0, 0), y: out.lead($0, 0, 1)) }
    }
  }

  /// Road point to pixel in the frame the models see: already calibrated, so K view_from_device.
  public static func transform(_ frame: DioramaMeta.Frame) -> [Double] {
    Diorama.m3mul([frame.f, 0, frame.cx, 0, frame.f, frame.cy, 0, 0, 1], Diorama.viewFromDevice)
  }

  /// What the display draws over a frame of the models', in drawing order, `leadBars` already advanced to it.
  public static func polygons(_ out: DioramaOutputs, _ meta: DioramaMeta, leadBars: LeadBars) -> [Polygon] {
    guard let frame = meta.frame, let model = Model(out) else { return [] }
    return polygons(model, transform: transform(frame), width: frame.w, height: frame.h, roadHeight: meta.height ?? heightInit, leadBars: leadBars.polygons)
  }

  /// The display's lead bars over time: each model frame is `rendersPerFrame` of the display's frames, and the bars'
  /// distance, offset and opacity lag as its FirstOrderFilters do. A lead that goes fades out where its bar last was.
  public struct LeadBars: Sendable {
    /// 20 Hz frames from the model, on a 60 Hz display.
    public static let rendersPerFrame = 3
    private static let dt = 1.0 / 60
    private static let positionRC = 0.2, fadeRC = 0.1

    private struct Filter: Sendable {
      let alpha: Double
      var x = 0.0, initialized: Bool

      init(rc: Double, initialized: Bool) {
        alpha = LeadBars.dt / (rc + LeadBars.dt)
        self.initialized = initialized
      }

      mutating func update(_ v: Double) -> Double {
        x = initialized ? (1 - alpha) * x + alpha * v : v
        initialized = true
        return x
      }
    }

    private struct Vehicle: Sendable {
      var d = Filter(rc: positionRC, initialized: false), y = Filter(rc: positionRC, initialized: false)
      var fade = Filter(rc: fadeRC, initialized: true)
      var bar: [SIMD2<Double>] = []
    }

    private var vehicles = [Vehicle(), Vehicle()]

    public init() {}

    /// Each bar while any of it shows, the far edge's middle on the road, where that lead's rear is. `index`: which of
    /// the model's leads.
    public var leads: [(index: Int, x: Double, y: Double)] {
      vehicles.enumerated().compactMap { i, v in
        v.bar.count >= 3 && UInt8(Diorama.clamp(255 * v.fade.x, 0, 255)) > 0 ? (i, v.d.x, -v.y.x) : nil
      }
    }

    /// The bars as they are now, in drawing order.
    public var polygons: [Polygon] {
      vehicles.compactMap { v in
        let a = UInt8(Diorama.clamp(255 * v.fade.x, 0, 255))
        return v.bar.count >= 3 && a > 0 ? Polygon(points: v.bar, fill: .solid(RGBA(r: 255, g: 255, b: 255, a: a))) : nil
      }
    }

    /// `renders` frames of the display showing this model frame.
    public mutating func advance(_ out: DioramaOutputs, _ meta: DioramaMeta, renders: Int) {
      guard let frame = meta.frame, let model = Model(out) else { return }
      advance(model, transform: CommaUI.transform(frame), width: frame.w, height: frame.h, roadHeight: meta.height ?? heightInit, renders: renders)
    }

    public mutating func advance(_ m: Model, transform: [Double], width: Double, height: Double, roadHeight: Double, renders: Int) {
      let view = View(transform: transform, width: width, height: height), leads = currentLeads(m)
      for _ in 0..<renders {
        for i in vehicles.indices where i < leads.count {
          let lead = leads[i]
          let visible = lead.present && lead.d < maxDrawDistance && !m.position.x.isEmpty
          // snap to a new vehicle instead of sliding over
          if !visible || abs(lead.y - vehicles[i].y.x) > 3 {
            vehicles[i].d.initialized = false
            vehicles[i].y.initialized = false
          }
          _ = vehicles[i].fade.update(visible ? leadOpacity : 0)
          if visible {
            vehicles[i].bar = view.leadBar(m.position, vehicles[i].d.update(lead.d), vehicles[i].y.update(lead.y), roadHeight)
          }
        }
      }
    }
  }

  /// The two leads the display considers, closer than 3 m being the same vehicle.
  static func currentLeads(_ m: Model) -> [(present: Bool, d: Double, y: Double)] {
    var leads = m.leads.prefix(2).map { (present: $0.prob > leadProb, d: $0.x, y: -$0.y) }
    if leads.count == 2, leads[0].present, abs(leads[1].d - leads[0].d) < 3 { leads[1] = (false, 0, 0) }
    return leads
  }

  /// `transform`: road point to pixel, row by row. `width` × `height`: the picture. `roadHeight`: the camera's, above the road.
  /// `leadBars`: the bars to draw, or nil for each lead's settled at this frame.
  public static func polygons(
    _ m: Model, transform: [Double], width: Double, height: Double, roadHeight: Double, leadBars: [Polygon]? = nil
  ) -> [Polygon] {
    guard let lastX = m.position.x.last, m.laneLines.count == 4, m.roadEdges.count == 2 else { return [] }
    let view = View(transform: transform, width: width, height: height)
    var out: [Polygon] = []
    func add(_ points: [SIMD2<Double>], _ fill: Fill) {
      if case .solid(let c) = fill, c.a == 0 { return }  // the display draws it, and nothing shows
      if points.count >= 3 { out.append(Polygon(points: points, fill: fill)) }
    }

    var maxDistance = Diorama.clamp(lastX, minDrawDistance, maxDrawDistance)
    let maxIdx = lengthIdx(m.laneLines[0].x, maxDistance)

    // The factor stays at 0.16 from the second line on: the fourth line and the road edges have it too.
    var widthFactor = 0.12
    for (i, line) in m.laneLines.enumerated() {
      if i == 1 || i == 2 { widthFactor = 0.16 }
      add(
        view.ribbon(line, widthFactor * m.laneLineProbs[i], 0, maxIdx, allowInvert: true),
        .solid(laneColor(m.laneLineProbs[i], adjacent: i == 1 || i == 2)))
    }
    for (i, edge) in m.roadEdges.enumerated() {
      add(
        view.ribbon(edge, widthFactor, 0, maxIdx, allowInvert: true),
        .solid(laneColor(1 - m.roadEdgeStds[i], adjacent: m.laneLineProbs[i + 1] < 0.25)))
    }

    // radarState.leadOne, as radard makes it from the model's lead
    if let lead = m.leads.first, lead.prob > leadProb {
      let d = (lead.x - radarToCamera) * 2
      maxDistance = Diorama.clamp(d - min(d * 0.35, 10), 0, maxDistance)
    }
    let path = view.ribbon(m.position, 0.9, roadHeight, lengthIdx(m.position.x, maxDistance), allowInvert: false)
    add(path, pathFill(path, m.accelerationX, height: height))

    if let leadBars { return out + leadBars }
    for lead in currentLeads(m) where lead.present && lead.d < maxDrawDistance {
      add(view.leadBar(m.position, lead.d, lead.y, roadHeight), .solid(RGBA(r: 255, g: 255, b: 255, a: UInt8(255 * leadOpacity))))
    }
    return out
  }

  /// The last index at or below `distance`, or 0.
  static func lengthIdx(_ xs: [Double], _ distance: Double) -> Int { xs.lastIndex { $0 <= distance } ?? 0 }

  static func laneColor(_ prob: Double, adjacent: Bool) -> RGBA {
    let a = UInt8(Diorama.clamp(prob, 0, 0.7) * 255)
    return adjacent ? RGBA(r: engaged.r, g: engaged.g, b: engaged.b, a: a) : RGBA(r: 255, g: 255, b: 255, a: a)
  }

  /// The path's color climbs the picture with its acceleration: green speeding up, red slowing down, near white coasting.
  static func pathFill(_ path: [SIMD2<Double>], _ acceleration: [Double], height: Double) -> Fill {
    let maxLen = min(path.count / 2, acceleration.count)
    var colors: [RGBA] = [], stops: [Double] = []
    var i = 0
    while i < maxLen {
      let y = path[i].y
      if y < 0 || y > height {
        i += 1
        continue
      }
      let point = 1 - y / height, a = acceleration[i]
      let hue = Diorama.clamp(60 + a * 35, 0, 120), saturation = min(abs(a * 1.5), 1)
      let lightness = Diorama.interp(saturation, [0, 1], [0.95, 0.62]), alpha = Diorama.interp(point, [0.75 / 2, 0.75], [0.4, 0])
      stops.append(point)
      colors.append(hsla(hue / 360, saturation, lightness, alpha))
      i += 1 + (i + 2 < maxLen ? 1 : 0)
    }
    return colors.count > 1 ? .gradient(colors: colors, stops: stops) : .solid(RGBA(r: 255, g: 255, b: 255, a: 30))
  }

  /// Python's colorsys.hls_to_rgb, truncated to bytes as the display does.
  static func hsla(_ h: Double, _ s: Double, _ l: Double, _ a: Double) -> RGBA {
    func byte(_ v: Double) -> UInt8 { UInt8(Diorama.clamp(v * 255, 0, 255)) }
    guard s != 0 else { return RGBA(r: byte(l), g: byte(l), b: byte(l), a: byte(a)) }
    let m2 = l <= 0.5 ? l * (1 + s) : l + s - l * s, m1 = 2 * l - m2
    func channel(_ hue: Double) -> Double {
      let h = hue - floor(hue)
      if h < 1.0 / 6 { return m1 + (m2 - m1) * h * 6 }
      if h < 0.5 { return m2 }
      if h < 2.0 / 3 { return m1 + (m2 - m1) * (2.0 / 3 - h) * 6 }
      return m1
    }
    return RGBA(r: byte(channel(h + 1.0 / 3)), g: byte(channel(h)), b: byte(channel(h - 1.0 / 3)), a: byte(a))
  }

  /// The display's shader at `t` up the picture, each channel from 0 to 1: straight alpha, mixed between stops.
  public static func color(at t: Double, colors: [RGBA], stops: [Double]) -> SIMD4<Double> {
    func value(_ c: RGBA) -> SIMD4<Double> { SIMD4(Double(c.r), Double(c.g), Double(c.b), Double(c.a)) / 255 }
    let stops = stops.map { Diorama.clamp($0, 0, 1) }
    guard let first = colors.first, let last = colors.last, let s0 = stops.first, let sn = stops.last else { return .zero }
    if t <= s0 { return value(first) }
    if t >= sn { return value(last) }
    for i in 0..<colors.count - 1 where t >= stops[i] && t <= stops[i + 1] {
      let k = (t - stops[i]) / max(stops[i + 1] - stops[i], 1e-6)
      return value(colors[i]) + (value(colors[i + 1]) - value(colors[i])) * k
    }
    return value(last)
  }

  private struct View {
    let transform: [Double], width: Double, height: Double

    func project(_ x: Double, _ y: Double, _ z: Double) -> SIMD3<Double> {
      SIMD3(
        transform[0] * x + transform[1] * y + transform[2] * z, transform[3] * x + transform[4] * y + transform[5] * z,
        transform[6] * x + transform[7] * y + transform[8] * z)
    }

    func ribbon(_ line: Line, _ yOff: Double, _ zOff: Double, _ maxIdx: Int, allowInvert: Bool) -> [SIMD2<Double>] {
      var left: [SIMD2<Double>] = [], right: [SIMD2<Double>] = []
      for i in 0...min(maxIdx, line.x.count - 1) where line.x[i] >= 0 {
        let l = project(line.x[i], line.y[i] - yOff, line.z[i] + zOff), r = project(line.x[i], line.y[i] + yOff, line.z[i] + zOff)
        guard abs(l.z) >= 1e-6, abs(r.z) >= 1e-6 else { continue }
        let a = SIMD2(l.x / l.z, l.y / l.z), b = SIMD2(r.x / r.z, r.y / r.z)
        guard inClip(a), inClip(b) else { continue }
        left.append(a)
        right.append(b)
      }
      // A hill's far side comes back down the picture: only points higher up than all before them stay.
      if !allowInvert, left.count > 1 {
        var lowest = Double.infinity, keep: [Int] = []
        for (i, p) in left.enumerated() {
          lowest = min(lowest, p.y)
          if p.y == lowest { keep.append(i) }
        }
        left = keep.map { left[$0] }
        right = keep.map { right[$0] }
      }
      return left + right.reversed()
    }

    private func inClip(_ p: SIMD2<Double>) -> Bool {
      p.x >= -clipMargin && p.x <= width + clipMargin && p.y >= -clipMargin && p.y <= height + clipMargin
    }

    /// On the road behind the lead, following the lane; at most `leadBarLength` long on screen.
    func leadBar(_ lane: Line, _ d: Double, _ yRel: Double, _ roadHeight: Double) -> [SIMD2<Double>] {
      let xs = [d, d - min(6, 0.25 * d)]
      let at = Diorama.interp(d, lane.x, lane.y)
      let ys = xs.map { Diorama.interp($0, lane.x, lane.y) - at - yRel }
      let zs = xs.map { Diorama.interp($0, lane.x, lane.z) + roadHeight }
      let half = leadBarWidth / 2
      var bar = [(xs[0], ys[0] + half, zs[0]), (xs[1], ys[1] + half, zs[1]), (xs[1], ys[1] - half, zs[1]), (xs[0], ys[0] - half, zs[0])].map {
        let p = project($0.0, $0.1, $0.2)
        return SIMD2(p.x / p.z, p.y / p.z)
      }
      let middle = (bar[1] + bar[2]) / 2 - (bar[0] + bar[3]) / 2
      let length = (middle * middle).sum().squareRoot()
      guard length > 0 else { return bar }
      let k = Diorama.clamp(length, 3, leadBarLength) / length
      bar[1] = bar[0] + (bar[1] - bar[0]) * k
      bar[2] = bar[3] + (bar[2] - bar[3]) * k
      return bar
    }
  }
}
