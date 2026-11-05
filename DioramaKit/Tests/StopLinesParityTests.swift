import Foundation
import JavaScriptCore
import Testing

@testable import DioramaCore

/// The Swift StopLines against diorama-engine.js's (Fixtures/stoplines.js), fed the same sightings: the stop-line
/// model's lines from a drive's stop (models/stops/expected.json), seen from a made-up path that slows, drifts and
/// turns, so that the world frame, the merging and the dropping all have something to do. Skipped without the frames.
@Test func stopLinesMatchThePage() throws {
  let models = URL(filePath: #filePath).deletingLastPathComponent().appending(path: "../../models").standardized
  guard let data = try? Data(contentsOf: models.appending(path: "stops/expected.json")),
    let records = try JSONSerialization.jsonObject(with: data) as? [[String: Any]]
  else { return }
  let (f, cx, cy, road) = (910.0, 256.0, 47.6, 1.37)

  var steps: [[String: Any]] = []
  var (x, y, yaw) = (0.0, 0.0, 0.0)
  for (i, r) in records.enumerated() {
    let v = max(12 - 0.05 * Double(i % 300), 0)  // slowing to a stop, then again
    x += cos(yaw) * v / 20
    y += sin(yaw) * v / 20
    yaw += 0.002 * sin(Double(i) / 40)
    let lines = (r["line"] as? [Double]).map { l in [["score": l[0], "x1": l[1], "y1": l[2], "x2": l[3], "y2": l[4]]] } ?? []
    steps.append(["k": r["frame"] as! Int, "pose": [x, y, yaw], "lines": lines])
  }

  let js = JSContext()!
  js.exceptionHandler = { _, e in Issue.record("JS: \(e!)") }
  let url = Bundle.module.url(forResource: "stoplines", withExtension: "js", subdirectory: "Fixtures")!
  js.evaluateScript(try String(contentsOf: url, encoding: .utf8))
  let json = String(decoding: try JSONSerialization.data(withJSONObject: steps), as: UTF8.self)
  let want = js.evaluateScript("JSON.stringify(run(\(json), \(f), \(cx), \(cy), \(road)))")!.toString()!
  let page = try JSONSerialization.jsonObject(with: Data(want.utf8)) as! [Any]

  let meta = DioramaMeta(["frame": ["w": 512, "h": 256, "f": f, "cx": cx, "cy": cy], "height": road])
  let cam = DioramaFrameCamera(meta, frame: meta.frame!)
  var tracker = DioramaStopLines()
  var drawn = 0
  for (i, st) in steps.enumerated() {
    let p = st["pose"] as! [Double]
    let pose = (x: p[0], y: p[1], yaw: p[2])
    let lines = (st["lines"] as! [[String: Double]]).map { DioramaStopLine(score: $0["score"]!, x1: $0["x1"]!, y1: $0["y1"]!, x2: $0["x2"]!, y2: $0["y2"]!) }
    tracker.observe(st["k"] as! Int, pose, lines, cam)
    let got = tracker.quad(st["k"] as! Int, pose)
    if let q = page[i] as? [[Double]] {
      drawn += 1
      guard let got else {
        Issue.record("frame \(st["k"]!): the page draws a line, the Swift none")
        continue
      }
      for (a, b) in zip(got, q) { #expect(abs(a.x - b[0]) < 1e-6 && abs(a.y - b[1]) < 1e-6, "frame \(st["k"]!): \(a) vs \(b)") }
    } else {
      #expect(got == nil, "frame \(st["k"]!): the Swift draws a line, the page none")
    }
  }
  #expect(drawn > 50)  // the stop's line is held and drawn
}
