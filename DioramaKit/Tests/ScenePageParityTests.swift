import Foundation
import JavaScriptCore
import Testing

@testable import DioramaCore

/// DioramaScene against the page's own engine (web/page/diorama-engine.js, read from the repo and run as it is), frame
/// for frame over the same drive.diorama: the drawn plan, the pose, the lead, the stop line, the stripes, the view
/// and the HUD, to 1e-6. The drive is synthetic (web/tools/synthetic_drive.py: cruising, a lead, a stop at a stop
/// line, pulling away), with nothing from a real drive, and committed beside the test.
@Test func sceneMatchesThePage() throws {
  let root = URL(filePath: #filePath).deletingLastPathComponent().appending(path: "../..").standardized
  let drive = try Data(contentsOf: root.appending(path: "DioramaKit/Tests/Fixtures/synthetic/drive.diorama"))
  let engine = try String(contentsOf: root.appending(path: "web/page/diorama-engine.js"), encoding: .utf8)

  let page = try pageSnapshots(engine: engine, drive: drive)
  let app = appSnapshots(drive: drive)
  #expect(page.count == app.count, "the page has \(page.count) frames, the app \(app.count)")
  #expect(page.count == 480)

  var compared = 0, withStopLine = 0, withLead = 0, bird = 0, road = 0
  for (want, got) in zip(page, app) {
    let frame = want["frame"] as? Int ?? -1
    if let field = firstDifference(want, got) {
      Issue.record("frame \(frame) first differs at \(field)")
      return
    }
    compared += 1
    if want["stopLine"] is [Any] { withStopLine += 1 }
    if want["lead"] is [String: Any] { withLead += 1 }
    if (want["view"] as? Double ?? 0) > 0.5 { bird += 1 } else { road += 1 }
  }
  // the drive exercises what the comparison is for
  #expect(withStopLine > 100, "only \(withStopLine) frames with a stop line")
  #expect(withLead > 50, "only \(withLead) frames with a lead")
  #expect(bird > 100 && road > 100, "\(road) frames in the road view, \(bird) in bird's-eye")
  print("scene parity: \(compared) frames, \(withStopLine) with a stop line, \(withLead) with a lead, \(road) road view, \(bird) bird's-eye")
}

/// The page's DioramaEngine.snapshot of every frame the drive steps it to.
private func pageSnapshots(engine: String, drive: Data) throws -> [[String: Any]] {
  let js = try #require(JSContext())
  var failure: String?
  js.exceptionHandler = { _, e in failure = e?.toString() }
  js.evaluateScript(engine)
  // the drive's bytes as an ArrayBuffer, without a copy through JSValue
  let bytes = UnsafeMutableRawPointer.allocate(byteCount: drive.count, alignment: 16)
  drive.copyBytes(to: bytes.assumingMemoryBound(to: UInt8.self), count: drive.count)
  let buffer = JSObjectMakeArrayBufferWithBytesNoCopy(
    js.jsGlobalContextRef, bytes, drive.count, { p, _ in p?.deallocate() }, nil, nil)
  js.setObject(JSValue(jsValueRef: buffer, in: js), forKeyedSubscript: "driveBuffer" as NSString)
  let json = js.evaluateScript(
    """
    (() => {
      const E = DioramaEngine, engine = new E.Engine(), out = [];
      for (const m of E.readFile(driveBuffer)) {
        const before = engine.state;
        engine.push(m);
        if (m.kind === 'D' && engine.state && engine.state !== before) out.push(E.snapshot(engine.state));
      }
      return JSON.stringify(out);
    })()
    """)?.toString()
  if let failure { throw CocoaError(.featureUnsupported, userInfo: [NSLocalizedDescriptionKey: "the page's engine: \(failure)"]) }
  let text = try #require(json)
  return try #require(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [[String: Any]])
}

/// DioramaScene's state after every drive message, in the snapshot's shape. Its stripes are the crosswalks with the
/// stop line after them, as the page's are.
private func appSnapshots(drive: Data) -> [[String: Any]] {
  let scene = DioramaScene()
  var out: [[String: Any]] = []
  for data in DioramaMessage.file(drive) {
    let msg = DioramaMessage(data)
    let before = scene.frame
    scene.push(msg)
    guard case .drive = msg, let s = scene.state, s.frame != before || out.isEmpty else { continue }
    let quad = { (q: [SIMD2<Double>]) in q.map { [$0.x, $0.y] } }
    var snap: [String: Any] = [
      "frame": s.frame, "seconds": s.seconds, "px": s.px, "py": s.py, "pz": s.pz,
      "pv": (0..<Diorama.idxN).map { s.out.plan($0, 3) },
      "pose": [s.pose.x, s.pose.y, s.pose.z],
      "stripes": (s.stripes + (s.stopLine.map { [$0] } ?? [])).map(quad),
      "view": s.view, "viewWhy": s.viewWhy,
      "hud": ["v": s.speed, "a": s.accel, "lead": s.leadDistance as Any],
    ]
    snap["lead"] = s.lead.map { l in ["x": l.x, "y": l.y, "heading": l.heading, "braking": l.braking, "seg": l.seg] as [String: Any] }
    snap["stopLine"] = s.stopLine.map(quad)
    out.append(snap)
  }
  return out
}

/// The first field (a dotted path) where the two differ by more than 1e-6, or nil. A missing value and JSON null are
/// the same; seconds are float32 on the wire and compared at that precision.
private func firstDifference(_ want: Any?, _ got: Any?, _ path: String = "") -> String? {
  let none = { (v: Any?) in v == nil || v is NSNull || (v as Any?).map { "\($0)" == "nil" } == true }
  if none(want) || none(got) { return none(want) && none(got) ? nil : "\(path) (\(want ?? "nil") vs \(got ?? "nil"))" }
  if let a = want as? [String: Any], let b = got as? [String: Any] {
    for key in a.keys.sorted() {
      if let d = firstDifference(a[key], b[key], path.isEmpty ? key : "\(path).\(key)") { return d }
    }
    return nil
  }
  if let a = want as? [Any], let b = got as? [Any] {
    guard a.count == b.count else { return "\(path) (\(a.count) items vs \(b.count))" }
    for i in a.indices {
      if let d = firstDifference(a[i], b[i], "\(path)[\(i)]") { return d }
    }
    return nil
  }
  if let a = want as? String, let b = got as? String { return a == b ? nil : "\(path) (\(a) vs \(b))" }
  if let a = want as? NSNumber, let b = got as? NSNumber {
    if CFGetTypeID(a) == CFBooleanGetTypeID() || CFGetTypeID(b) == CFBooleanGetTypeID() {
      return a.boolValue == b.boolValue ? nil : "\(path) (\(a) vs \(b))"
    }
    let tol = path == "seconds" ? 1e-4 : 1e-6
    return abs(a.doubleValue - b.doubleValue) <= tol * max(1, abs(a.doubleValue)) ? nil : "\(path) (\(a) vs \(b))"
  }
  if let b = got as? Bool, let a = want as? NSNumber { return a.boolValue == b ? nil : "\(path) (\(a) vs \(b))" }
  return "\(path) (\(type(of: want!)) vs \(type(of: got!)))"
}
