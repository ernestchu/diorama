import CoreML
import Foundation
import Testing

@testable import DioramaLink
@testable import DioramaUI

/// The Core ML model and the Swift decoder against the ONNX twin decoded as diorama/stopline.py
/// decodes, on frames of recorded drives: models/parity (sim/stopline_parity.py) and models/stops (a stop at a
/// painted line, the frames as frames.u8 and expected.json, as from stops.npz).
/// Skipped where the model or the frames have not been fetched.
@Test(arguments: ["parity", "stops"]) func stopLineMatchesTheReference(_ set: String) throws {
  let models = URL(filePath: #filePath).deletingLastPathComponent().appending(path: "../../models").standardized
  let package = models.appending(path: "StoplinePilot.mlpackage")
  let parity = models.appending(path: set)
  guard FileManager.default.fileExists(atPath: package.path), let frames = try? Data(contentsOf: parity.appending(path: "frames.u8")),
    let expected = try? JSONSerialization.jsonObject(with: Data(contentsOf: parity.appending(path: "expected.json"))) as? [[String: Any]]
  else { return }
  let model = try StopLineModel(url: try MLModel.compileModel(at: package))
  let size = 6 * 128 * 256
  #expect(frames.count == expected.count * size)
  for (i, want) in expected.enumerated() {
    let got = try model.run(frames.subdata(in: i * size..<(i + 1) * size))
    let line = (want["line"] as? [Double])?.map(Float.init)
    if let line, let g = got.line {
      #expect(abs(g.score - line[0]) < 0.03, "frame \(want["frame"]!): score \(g.score) vs \(line[0])")
      for (a, b) in zip([g.x1, g.y1, g.x2, g.y2], line[1...]) { #expect(abs(a - b) < 3, "frame \(want["frame"]!): \(a) vs \(b)") }
    } else {
      // only a score at the threshold may land either side of it in float16
      let score = line?[0] ?? got.line?.score ?? 0
      #expect(line == nil && got.line == nil || abs(score - StopLineDecoder.lineScore) < 0.03, "frame \(want["frame"]!): \(String(describing: got.line)) vs \(String(describing: line))")
    }
    let lead = (want["lead"] as? [Double])?.map(Float.init)
    #expect((lead == nil) == (got.lead == nil) || abs((lead?[0] ?? got.lead?.score ?? 0) - StopLineDecoder.leadScore) < 0.03)
  }
}
