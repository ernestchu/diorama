import CoreGraphics
import DioramaCore
import Foundation
import JavaScriptCore
import Testing

@testable import DioramaLink
@testable import DioramaUI

/// The app's drawing of the comma's prediction against what the comma 4 display's own ModelRenderer, at the commit
/// CommaUI is pinned to, draws for the same outputs: Fixtures/comma_ui, from sim/comma_ui_parity.py.
private struct Golden: @unchecked Sendable {
  let json: [String: Any]
  var cases: [String: [String: Any]] { json["cases"] as! [String: [String: Any]] }
  var width: Double { json["width"] as! Double }
  var height: Double { json["height"] as! Double }

  static let folder = URL(filePath: #filePath).deletingLastPathComponent().appending(path: "Fixtures/comma_ui")
  static let shared = Golden(json: try! JSONSerialization.jsonObject(with: Data(contentsOf: folder.appending(path: "golden.json"))) as! [String: Any])

  static func outputs(_ heads: [String: [Double]]) -> DioramaOutputs {
    // the numbers were float32 on the comma's side
    func h(_ k: String) -> [Double] { heads[k]!.map { Double(Float($0)) } }
    var out = DioramaOutputs()
    out.plan = h("plan")
    out.laneLines = h("lane_lines")
    out.roadEdges = h("road_edges")
    out.roadEdgesStds = h("road_edges_stds")
    out.laneLinesProb = h("lane_lines_prob")
    out.lead = h("lead")
    out.leadProb = h("lead_prob")
    return out
  }

  func polygons(_ name: String) throws -> [CommaUI.Polygon] {
    let heads = cases[name]!["heads"] as! [String: [Double]]
    let model = try #require(CommaUI.Model(Self.outputs(heads)))
    return CommaUI.polygons(model, transform: json["transform"] as! [Double], width: width, height: height, roadHeight: json["road_height"] as! Double)
  }
}

private func rgba(_ c: [Int]) -> CommaUI.RGBA { CommaUI.RGBA(r: UInt8(c[0]), g: UInt8(c[1]), b: UInt8(c[2]), a: UInt8(c[3])) }

private func expectClose(_ a: CommaUI.RGBA, _ b: CommaUI.RGBA, _ what: String) {
  // the comma computes in float32 and truncates, the app in double
  for (x, y) in [(a.r, b.r), (a.g, b.g), (a.b, b.b), (a.a, b.a)] { #expect(abs(Int(x) - Int(y)) <= 1, "\(what): \(a) vs \(b)") }
}

@Test(arguments: Golden.shared.cases.keys.sorted()) func commaDrawingMatchesTheDisplay(_ name: String) throws {
  let want = Golden.shared.cases[name]!["draws"] as! [[String: Any]]
  let got = try Golden.shared.polygons(name)
  try #require(got.count == want.count, "\(name): \(got.count) polygons, the display draws \(want.count)")
  for (i, (g, w)) in zip(got, want).enumerated() {
    let flat = w["points"] as! [Double]
    try #require(g.points.count * 2 == flat.count, "\(name) #\(i): \(g.points.count) points, the display has \(flat.count / 2)")
    for (j, p) in g.points.enumerated() {
      #expect(abs(p.x - flat[2 * j]) < 0.05 && abs(p.y - flat[2 * j + 1]) < 0.05, "\(name) #\(i) point \(j): \(p) vs \(flat[2 * j]), \(flat[2 * j + 1])")
    }
    switch g.fill {
    case .solid(let c):
      expectClose(c, rgba(try #require(w["color"] as? [Int], "\(name) #\(i): the display has a gradient")), "\(name) #\(i)")
    case .gradient(let colors, let stops):
      let wantColors = try #require(w["colors"] as? [[Int]], "\(name) #\(i): the display has one color")
      let wantStops = w["stops"] as! [Double]
      try #require(colors.count == wantColors.count && stops.count == wantStops.count, "\(name) #\(i): \(colors.count) stops, the display has \(wantColors.count)")
      for (c, wc) in zip(colors, wantColors) { expectClose(c, rgba(wc), "\(name) #\(i)") }
      for (s, ws) in zip(stops, wantStops) { #expect(abs(s - ws) < 1e-3, "\(name) #\(i): stop \(s) vs \(ws)") }
    }
  }
}

@Test func commaFrameTransformIsTheDisplays() {
  let g = Golden.shared.json
  let frame = DioramaMeta(["frame": ["w": 512.0, "h": 256.0, "f": g["f"]!, "cx": g["cx"]!, "cy": g["cy"]!]]).frame!
  for (a, b) in zip(CommaUI.transform(frame), g["transform"] as! [Double]) { #expect(abs(a - b) < 1e-3) }
}

/// Painted, not just described: every pixel away from an edge, where the display does not antialias and the app does.
@Test(arguments: ["cruise", "braking", "lead_close", "curve_left"]) func commaPaintingMatchesTheDisplay(_ name: String) throws {
  let w = Int(Golden.shared.width), h = Int(Golden.shared.height)
  let packed = try Data(contentsOf: Golden.folder.appending(path: "\(name).rgba.deflate"))
  let want = [UInt8](try (packed as NSData).decompressed(using: .zlib) as Data)
  #expect(want.count == w * h * 4)

  let cg = try #require(
    CGContext(
      data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
  cg.setFillColor(CGColor(srgbRed: 128.0 / 255, green: 128.0 / 255, blue: 128.0 / 255, alpha: 1))
  cg.fill(CGRect(x: 0, y: 0, width: w, height: h))
  cg.translateBy(x: 0, y: Double(h))
  cg.scaleBy(x: 1, y: -1)
  CommaPainter.draw(try Golden.shared.polygons(name), height: Double(h), in: cg)
  let got = Array(UnsafeBufferPointer(start: cg.data!.assumingMemoryBound(to: UInt8.self), count: w * h * 4))

  // Within 1.5 px of a polygon's edge the two may differ: the display does not antialias, and a ribbon thinner than a
  // pixel can miss every pixel's center there.
  let polygons = try Golden.shared.polygons(name)
  let edges = try #require(CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue))
  edges.translateBy(x: 0, y: Double(h))
  edges.scaleBy(x: 1, y: -1)
  edges.setStrokeColor(gray: 1, alpha: 1)
  edges.setLineWidth(3)
  edges.setLineJoin(.round)
  edges.setLineCap(.round)
  for polygon in polygons {
    edges.addLines(between: polygon.points.map { CGPoint(x: $0.x, y: $0.y) })
    edges.closePath()
    edges.strokePath()
  }
  let edge = Array(UnsafeBufferPointer(start: edges.data!.assumingMemoryBound(to: UInt8.self), count: w * h))
  var compared = 0, painted = 0, total = 0, worst = 0, worstAt = (0, 0), got3 = [UInt8](), want3 = [UInt8]()
  for y in 2..<h - 2 {
    for x in 2..<w - 2 where edge[y * w + x] == 0 {
      compared += 1
      var moved = false
      for c in 0..<3 {
        let d = abs(Int(got[(y * w + x) * 4 + c]) - Int(want[(y * w + x) * 4 + c]))
        total += d
        if d > worst {
          (worst, worstAt) = (d, (x, y))
          (got3, want3) = (Array(got[(y * w + x) * 4..<(y * w + x) * 4 + 3]), Array(want[(y * w + x) * 4..<(y * w + x) * 4 + 3]))
        }
        if want[(y * w + x) * 4 + c] != 128 { moved = true }
      }
      if moved { painted += 1 }
    }
  }
  #expect(compared > w * h / 2, "\(name): only \(compared) pixels compared")
  #expect(painted > 500, "\(name): only \(painted) painted pixels compared")
  #expect(worst <= 4, "\(name): pixel \(worstAt) is \(got3), the display's \(want3)")
  #expect(Double(total) / Double(compared * 3) < 0.6, "\(name): off by \(Double(total) / Double(compared * 3)) on average")
}

/// The bar over time, against the display's own filters: it comes in, follows, snaps to another car, fades out and two come.
@Test func commaLeadBarSmoothsAsTheDisplayDoes() throws {
  let g = Golden.shared.json, sequence = g["lead_sequence"] as! [String: Any], base = sequence["base"] as! [String: [Double]]
  var bars = CommaUI.LeadBars(), sawBar = false, sawFading = false
  for (k, frame) in (sequence["frames"] as! [[String: Any]]).enumerated() {
    var heads = base
    heads["lead"] = (frame["lead"] as! [Double])
    heads["lead_prob"] = (frame["lead_prob"] as! [Double])
    let model = try #require(CommaUI.Model(Golden.outputs(heads)))
    bars.advance(
      model, transform: g["transform"] as! [Double], width: g["width"] as! Double, height: g["height"] as! Double, roadHeight: g["road_height"] as! Double,
      renders: CommaUI.LeadBars.rendersPerFrame)
    let want = frame["bars"] as! [[String: Any]], got = bars.polygons
    try #require(got.count == want.count, "frame \(k): \(got.count) bars, the display has \(want.count)")
    for (g, w) in zip(got, want) {
      let flat = w["points"] as! [Double]
      try #require(g.points.count * 2 == flat.count)
      for (j, p) in g.points.enumerated() { #expect(abs(p.x - flat[2 * j]) < 0.1 && abs(p.y - flat[2 * j + 1]) < 0.1, "frame \(k) point \(j): \(p) vs \(flat[2 * j]), \(flat[2 * j + 1])") }
      guard case .solid(let c) = g.fill else { Issue.record("frame \(k): a bar with a gradient"); continue }
      let alpha = w["alpha"] as! Int
      #expect(abs(Int(c.a) - alpha) <= 1, "frame \(k): alpha \(c.a) vs \(alpha)")
      sawBar = true
      if alpha < 200 { sawFading = true }
    }
  }
  #expect(sawBar && sawFading)
}

/// The page's copy of the drawing (web/page/diorama-engine.js, run as it is) against the same recordings of the display.
@Test func thePagesCommaDrawingMatchesTheDisplay() throws {
  let root = URL(filePath: #filePath).deletingLastPathComponent().appending(path: "../..").standardized
  let js = try #require(JSContext())
  var failure: String?
  js.exceptionHandler = { _, e in failure = e?.toString() }
  js.evaluateScript(try String(contentsOf: root.appending(path: "web/page/diorama-engine.js"), encoding: .utf8))
  js.setObject(try String(contentsOf: Golden.folder.appending(path: "golden.json"), encoding: .utf8), forKeyedSubscript: "golden" as NSString)
  let result = js.evaluateScript(
    """
    (() => {
      const E = DioramaEngine, g = JSON.parse(golden), bad = [];
      const out = (h) => Object.fromEntries(Object.entries(h).map(([k, v]) => [k, Float64Array.from(v, Math.fround)]));
      const near = (a, b, tol) => Math.abs(a - b) <= tol;
      function same(name, got, want) {
        if (got.length !== want.length) { bad.push(`${name}: ${got.length} polygons, the display ${want.length}`); return; }
        got.forEach((p, i) => {
          const w = want[i], flat = p.points.flat();
          if (flat.length !== w.points.length) { bad.push(`${name} #${i}: ${flat.length} numbers vs ${w.points.length}`); return; }
          flat.forEach((v, j) => { if (!near(v, w.points[j], 0.1)) bad.push(`${name} #${i} point ${j}: ${v} vs ${w.points[j]}`); });
          if (w.color || w.alpha !== undefined) {
            const c = w.color || [255, 255, 255, w.alpha];
            if (!p.color || p.color.some((v, j) => !near(v, c[j], 1))) bad.push(`${name} #${i}: color ${p.color} vs ${c}`);
          } else if (!p.colors || p.colors.length !== w.colors.length) bad.push(`${name} #${i}: gradient`);
          else {
            p.colors.forEach((c, j) => c.forEach((v, k) => { if (!near(v, w.colors[j][k], 1)) bad.push(`${name} #${i}: stop color ${j}`); }));
            p.stops.forEach((v, j) => { if (!near(v, w.stops[j], 1e-3)) bad.push(`${name} #${i}: stop ${j}`); });
          }
        });
      }
      for (const [name, c] of Object.entries(g.cases)) same(name, E.commaPolygons(E.commaModel(out(c.heads)), g.transform, g.width, g.height, g.road_height), c.draws);
      const bars = new E.CommaLeadBars();
      g.lead_sequence.frames.forEach((f, k) => {
        bars.advance(E.commaModel(out({ ...g.lead_sequence.base, lead: f.lead, lead_prob: f.lead_prob })), g.transform, g.width, g.height, g.road_height, E.COMMA.RENDERS);
        same(`bars ${k}`, bars.polygons, f.bars);
      });
      return bad.length + ' ' + bad.slice(0, 8).join('; ');
    })()
    """)?.toString()
  #expect(failure == nil, "\(failure ?? "")")
  #expect(result?.hasPrefix("0 ") == true, "\(result ?? "nothing")")
}

/// What the footage window holds with the switch each way.
@Test func footageHoldsCommaAloneOrBoth() throws {
  let heads = Golden.shared.cases["lead_close"]!["heads"] as! [String: [Double]]
  let names: [(String, Int)] = [("plan", 990), ("lane_lines", 528), ("lane_lines_prob", 8), ("road_edges", 264), ("lead", 144), ("lead_prob", 3), ("pose", 12)]
  // raw heads that parse to those: means, then log stds of zero
  var values: [Float] = []
  for (name, count) in names {
    switch name {
    case "lane_lines_prob", "lead_prob":
      values += heads[name]!.map { p in Float(log(max(p, 1e-6) / max(1 - p, 1e-6))) }
    case "pose":
      values += [Float](repeating: 0, count: count)
    default:
      values += heads[name == "plan" ? "plan" : name]!.map(Float.init)
      values += [Float](repeating: 0, count: count / 2)
    }
  }
  let meta = DioramaMeta([
    "frame": ["w": 512.0, "h": 256.0, "f": 910.0, "cx": 256.0, "cy": 47.6], "height": 1.22,
    "drive": names.map { [$0.0, $0.1] as [Any] },
    "seg": StopLineMessages.segMeta,
  ])
  let line = StopLineDecoder.Line(score: 0.9, x1: 100, y1: 180, x2: 300, y2: 182)
  let lead = StopLineDecoder.Box(score: 0.8, x1: 200, y1: 100, x2: 300, y2: 150)

  func run(_ diorama: Bool) -> InputPicture? {
    var input = ModelInput()
    input.note(.meta(meta))
    input.note(.drive(frame: 7, seconds: 0, ms: 0, values: values))
    input.note(.input(frame: 7, width: 256, height: 128, packed: [UInt8](repeating: 128, count: 6 * 256 * 128)))
    for message in StopLineMessages.messages(frame: 7, ms: 5, line: line, lead: lead) { input.note(DioramaMessage(message)) }
    return input.latest(meta, diorama: diorama)
  }
  let comma = try #require(run(false)), both = try #require(run(true))
  // four lane lines, the path and the lead bar; these log-stds of zero fade the road edges out entirely
  #expect(comma.comma.count == 6 && comma.comma.count == both.comma.count)
  #expect(comma.labels.isEmpty)
  #expect(both.labels.contains { $0.text == "car 1.00" } && both.labels.contains { $0.text == "stop 0.90" })
}
