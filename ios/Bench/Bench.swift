import CoreML
import CoreVideo
import DioramaLink
import DioramaUI
import Foundation
import UIKit
import os

/// The trunk benchmark: every stop-line model bundled in the app (Models/, with models.json beside them), timed the
/// way the r18 vs r34 run on 2026-11-07 timed them, untethered. Each model gets, in turn: a cool-down, its compile and
/// load times, prediction-only timings for each compute-unit setting with Core ML's placement of its operations, the
/// app's full step (StopLineModel.run: input packing, prediction, decoding) back to back, and the same step paced at
/// 20 Hz. A failure is recorded and the run carries on with the next model. The report is written to Documents
/// after every model, so a run cut short still leaves what it measured.
struct BenchSettings: Sendable {
  var pacedSeconds = 180.0
  /// Rest before each model, after which it also waits (up to `maxCoolSeconds`) for the thermal state to be nominal.
  var restSeconds = 60.0
  var maxCoolSeconds = 600.0
  var includeCPUOnly = true
  var warmup = 30
  var runs = 200
  var cpuOnlyRuns = 50
}

struct BenchEntry: Sendable {
  let name: String, arch: String, sha256: String, inputs: [String]
  let params: Double, gmacs: Double
  let package: URL
}

@MainActor
@Observable
final class BenchRun {
  enum State: Equatable { case idle, running, finished, failed(String) }
  private(set) var state = State.idle
  private(set) var phase = ""
  private(set) var progress = 0.0
  /// One line per model done: its name, the Neural Engine's share of operations, and its medians.
  private(set) var summaries: [(name: String, text: String)] = []
  private(set) var reportURL: URL?
  private var task: Task<Void, Never>?

  let entries: [BenchEntry]
  let manifestNote: String
  let loadError: String?

  init() {
    let dir = Bundle.main.bundleURL.appending(path: "Models")
    var entries: [BenchEntry] = []
    var note = "", failure: String?
    do {
      let json = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appending(path: "models.json"))) as? [String: Any] ?? [:]
      note = json["note"] as? String ?? ""
      for m in json["models"] as? [[String: Any]] ?? [] {
        let name = m["name"] as? String ?? "?"
        let folder = dir.appending(path: name)
        let pkg = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil))?.first { $0.pathExtension == "mlpackage" }
        entries.append(
          BenchEntry(
            name: name, arch: m["arch"] as? String ?? "", sha256: m["sha256"] as? String ?? "", inputs: m["inputs"] as? [String] ?? [],
            params: (m["params_M"] as? NSNumber)?.doubleValue ?? 0, gmacs: (m["gmacs"] as? NSNumber)?.doubleValue ?? 0,
            package: pkg ?? folder.appending(path: "missing.mlpackage")))
      }
    } catch {
      failure = "models.json: \(error.localizedDescription)"
    }
    (self.entries, self.manifestNote, self.loadError) = (entries, note, failure)
  }

  var running: Bool { state == .running }

  func start(_ settings: BenchSettings) {
    guard !running else { return }
    state = .running
    summaries = []
    reportURL = nil
    progress = 0
    UIApplication.shared.isIdleTimerDisabled = true
    let entries = entries, note = manifestNote
    task = Task.detached(priority: .userInitiated) { [weak self] in
      let bench = Bench(settings: settings, entries: entries, note: note) { update in
        await MainActor.run { self?.apply(update) }
      }
      await bench.run()
      await MainActor.run {
        UIApplication.shared.isIdleTimerDisabled = false
        if self?.state == .running { self?.state = .finished }
      }
    }
  }

  func cancel() { task?.cancel() }

  private func apply(_ u: Bench.Update) {
    switch u {
    case .phase(let p, let f):
      phase = p
      progress = f
    case .model(let name, let text): summaries.append((name, text))
    case .saved(let url): reportURL = url
    case .failed(let why): state = .failed(why)
    }
  }
}

/// The run itself, off the main actor.
final class Bench: @unchecked Sendable {
  enum Update: Sendable {
    case phase(String, Double)
    case model(String, String)
    case saved(URL)
    case failed(String)
  }

  let settings: BenchSettings
  let entries: [BenchEntry]
  let note: String
  let send: @Sendable (Update) async -> Void
  let clock = ContinuousClock()
  let started = Date()
  let t0 = ContinuousClock.now
  private var report: [String: Any] = [:]
  private var thermalLog: [[String: Any]] = []
  private var lastThermal = ""
  private let url: URL
  static let log = Logger(subsystem: "diorama", category: "bench")

  init(settings: BenchSettings, entries: [BenchEntry], note: String, send: @escaping @Sendable (Update) async -> Void) {
    (self.settings, self.entries, self.note, self.send) = (settings, entries, note, send)
    let stamp = ISO8601DateFormatter.string(from: Date(), timeZone: .current, formatOptions: [.withFullDate, .withTime])
    url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appending(path: "bench-\(stamp).json")
  }

  // MARK: the run

  func run() async {
    let info = Bundle.main.infoDictionary ?? [:]
    report = [
      "schema": 1,
      "timestamp": ISO8601DateFormatter().string(from: started),
      "device": [
        "machine": Self.machine(), "name": await MainActor.run { UIDevice.current.model },
        "os": ProcessInfo.processInfo.operatingSystemVersionString,
        "lowPowerMode": ProcessInfo.processInfo.isLowPowerModeEnabled,
        "processors": ProcessInfo.processInfo.activeProcessorCount,
        "memoryGB": Double(ProcessInfo.processInfo.physicalMemory) / 1e9,
      ],
      "app": [
        "version": info["CFBundleShortVersionString"] ?? "", "build": info["CFBundleVersion"] ?? "",
        "commit": info["DioramaGitCommit"] ?? "unknown",
      ],
      "settings": [
        "pacedSeconds": settings.pacedSeconds, "restSeconds": settings.restSeconds, "maxCoolSeconds": settings.maxCoolSeconds,
        "warmup": settings.warmup, "runs": settings.runs, "cpuOnlyRuns": settings.includeCPUOnly ? settings.cpuOnlyRuns : 0,
        "pacedHz": 20, "frames": "synthetic (a road-like luma ramp, a bright bar, flat chroma)",
      ],
      "manifestNote": note,
      "models": [[String: Any]](),
    ]
    // the thermal state is logged whenever it changes, checked at each phase and every 10 s of the paced runs
    sampleThermal("start")

    for (i, e) in entries.enumerated() {
      if Task.isCancelled { break }
      let base = Double(i) / Double(entries.count)
      var m: [String: Any] = [
        "name": e.name, "arch": e.arch, "sha256": e.sha256, "inputs": e.inputs, "params_M": e.params, "gmacs": e.gmacs,
      ]
      await cool(e.name, base, &m)
      do {
        try await measure(e, base, &m)
      } catch is CancellationError {
        m["error"] = "cancelled"
      } catch {
        m["error"] = "\(error)"
        Self.log.error("bench \(e.name, privacy: .public): \(error, privacy: .public)")
      }
      m["thermalAtEnd"] = Self.thermal()
      var models = report["models"] as! [[String: Any]]
      models.append(m)
      report["models"] = models
      save()
      await send(.model(e.name, Self.summary(m)))
    }
    sampleThermal("end")
    report["finished"] = !Task.isCancelled
    report["seconds"] = Self.ms(clock.now - t0) / 1000
    save()
    await send(.phase(Task.isCancelled ? "cancelled" : "done", 1))
  }

  /// A fixed rest, then up to `maxCoolSeconds` more until the phone is back to nominal.
  private func cool(_ name: String, _ base: Double, _ m: inout [String: Any]) async {
    let start = clock.now
    m["thermalBeforeRest"] = Self.thermal()
    await send(.phase("\(name): resting \(Int(settings.restSeconds)) s", base))
    try? await Task.sleep(for: .seconds(settings.restSeconds))
    while ProcessInfo.processInfo.thermalState != .nominal && clock.now - start < .seconds(settings.restSeconds + settings.maxCoolSeconds) {
      if Task.isCancelled { return }
      sampleThermal(nil)
      await send(.phase("\(name): cooling (\(Self.thermal()))", base))
      try? await Task.sleep(for: .seconds(5))
    }
    m["cooledSeconds"] = Self.ms(clock.now - start) / 1000
    m["thermalAtStart"] = Self.thermal()
  }

  private func measure(_ e: BenchEntry, _ base: Double, _ m: inout [String: Any]) async throws {
    let step = 1 / Double(entries.count)
    guard FileManager.default.fileExists(atPath: e.package.path) else { throw BenchError.missing(e.package.lastPathComponent) }
    await send(.phase("\(e.name): compiling", base + 0.02 * step))
    let c0 = clock.now
    let compiled = try await MLModel.compileModel(at: e.package)
    m["compileMs"] = Self.ms(clock.now - c0)
    defer { try? FileManager.default.removeItem(at: compiled) }

    let packed = Self.frame()
    var units: [(String, MLComputeUnits, Int)] = [("cpuAndNeuralEngine", .cpuAndNeuralEngine, settings.runs), ("all", .all, settings.runs)]
    if settings.includeCPUOnly { units.append(("cpuOnly", .cpuOnly, settings.cpuOnlyRuns)) }
    var perUnits: [String: Any] = [:]
    for (k, (label, unit, runs)) in units.enumerated() {
      try Task.checkCancellation()
      sampleThermal(nil)
      await send(.phase("\(e.name): prediction, \(label)", base + (0.05 + 0.1 * Double(k)) * step))
      var u: [String: Any] = [:]
      do {
        let config = MLModelConfiguration()
        config.computeUnits = unit
        let l0 = clock.now
        let model = try Self.load(compiled, config)
        u["firstLoadMs"] = Self.ms(clock.now - l0)
        let l1 = clock.now
        _ = try Self.load(compiled, config)
        u["reloadMs"] = Self.ms(clock.now - l1)
        let input = try Self.inputs(packed, wide: e.inputs.contains("frame_wide"))
        let p0 = clock.now
        _ = try Self.predict(model, input)
        u["firstPredictionMs"] = Self.ms(clock.now - p0)
        for _ in 0..<settings.warmup { _ = try Self.predict(model, input) }
        var times: [Double] = []
        times.reserveCapacity(runs)
        for _ in 0..<runs {
          let s = clock.now
          _ = try Self.predict(model, input)
          times.append(Self.ms(clock.now - s))
        }
        u["prediction"] = Self.stats(times)
        u["predictionRawMs"] = times
        u["placement"] = await Self.placement(compiled, unit)
      } catch {
        u["error"] = "\(error)"
      }
      perUnits[label] = u
    }
    m["computeUnits"] = perUnits

    // the app's step: StopLineModel.run where the model takes the app's inputs; with the wide camera's too, the same
    // step with them added (the app has no such model yet)
    try Task.checkCancellation()
    await send(.phase("\(e.name): app step", base + 0.4 * step))
    let runStep: () throws -> Void
    if e.inputs.contains("frame_wide") {
      let wide = try WideStep(url: compiled)
      runStep = { try wide.run(packed) }
      m["appStepPath"] = "StopLineModel.run's packing and decoding, plus frame_wide and uv_wide (the narrow frame again), .all"
    } else {
      let app = try StopLineModel(url: compiled)
      runStep = { _ = try app.run(packed) }
      m["appStepPath"] = "StopLineModel.run (.all), as the app runs it"
    }
    for _ in 0..<settings.warmup { try runStep() }
    var times: [Double] = []
    for _ in 0..<settings.runs {
      let s = clock.now
      try runStep()
      times.append(Self.ms(clock.now - s))
    }
    m["appStep"] = Self.stats(times)

    // paced at 20 Hz, as the live view runs it
    var paced: [String: Any] = ["thermalStart": Self.thermal()]
    let start = clock.now
    var all: [Double] = [], window: [Double] = [], minutes: [[String: Any]] = []
    var tick = 0, late = 0
    while clock.now - start < .seconds(settings.pacedSeconds) {
      try Task.checkCancellation()
      let due = start + .milliseconds(50 * tick)
      if due > clock.now { try? await clock.sleep(until: due) } else if tick > 0 { late += 1 }
      let s = clock.now
      try runStep()
      let t = Self.ms(clock.now - s)
      all.append(t)
      window.append(t)
      tick += 1
      if window.count == 1200 {
        var w = Self.stats(window) as [String: Any]
        w["thermal"] = Self.thermal()
        minutes.append(w)
        window = []
      }
      if tick % 200 == 0 { sampleThermal(nil) }
      if tick % 100 == 0 {
        let f = (clock.now - start) / .seconds(settings.pacedSeconds)
        await send(.phase("\(e.name): paced at 20 Hz, \(Int(Self.ms(clock.now - start) / 1000)) of \(Int(settings.pacedSeconds)) s", base + (0.45 + 0.55 * f) * step))
      }
    }
    let secs = Self.ms(clock.now - start) / 1000
    paced["seconds"] = secs
    paced["frames"] = all.count
    paced["rateHz"] = Double(all.count) / secs
    paced["framesLate"] = late
    paced["latency"] = all.isEmpty ? [:] : Self.stats(all)
    paced["perMinute"] = minutes
    paced["thermalEnd"] = Self.thermal()
    m["paced"] = paced
  }

  // MARK: helpers

  enum BenchError: Error, CustomStringConvertible {
    case missing(String), noOutput
    var description: String {
      switch self {
      case .missing(let f): "no model package (\(f)) in the app"
      case .noOutput: "the model gave no line or lead output"
      }
    }
  }

  private func sampleThermal(_ event: String?) {
    let t = Self.thermal()
    guard event != nil || t != lastThermal else { return }
    lastThermal = t
    var s: [String: Any] = ["seconds": Self.ms(clock.now - t0) / 1000, "thermal": t, "lowPowerMode": ProcessInfo.processInfo.isLowPowerModeEnabled]
    if let event { s["event"] = event }
    thermalLog.append(s)
    report["thermalLog"] = thermalLog
  }

  private func save() {
    sampleThermal(nil)
    guard let data = try? JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]) else { return }
    try? data.write(to: url, options: .atomic)
    let url = url
    Task { await send(.saved(url)) }
  }

  static func summary(_ m: [String: Any]) -> String {
    if let e = m["error"] as? String { return "error: \(e)" }
    var parts: [String] = []
    let units = m["computeUnits"] as? [String: Any] ?? [:]
    if let ane = units["cpuAndNeuralEngine"] as? [String: Any] {
      if let p = (ane["prediction"] as? [String: Double])?["median"] { parts.append(String(format: "predict %.2f ms", p)) }
      if let ops = (ane["placement"] as? [String: Any])?["ops"] as? [String: Int] {
        let total = ops.values.reduce(0, +)
        parts.append("ANE \(ops["ANE"] ?? 0)/\(total) ops")
      }
    }
    if let a = (m["appStep"] as? [String: Double])?["median"] { parts.append(String(format: "step %.1f ms", a)) }
    if let p = m["paced"] as? [String: Any], let l = (p["latency"] as? [String: Double])?["median"] {
      parts.append(String(format: "20 Hz %.1f ms, %d late", l, p["framesLate"] as? Int ?? 0))
    }
    return parts.joined(separator: " · ")
  }

  static func machine() -> String {
    var u = utsname()
    uname(&u)
    return withUnsafeBytes(of: &u.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
  }

  static func thermal() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: "nominal"
    case .fair: "fair"
    case .serious: "serious"
    case .critical: "critical"
    @unknown default: "unknown"
    }
  }

  static func ms(_ d: Duration) -> Double { Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15 }

  static func stats(_ t: [Double]) -> [String: Double] {
    let s = t.sorted()
    func q(_ p: Double) -> Double { s[min(s.count - 1, Int((Double(s.count - 1) * p).rounded()))] }
    return ["n": Double(s.count), "median": q(0.5), "p90": q(0.9), "p99": q(0.99), "min": s[0], "max": s[s.count - 1], "mean": s.reduce(0, +) / Double(s.count)]
  }

  // synchronous on purpose: in an async function Swift picks Core ML's async overloads
  static func load(_ url: URL, _ config: MLModelConfiguration) throws -> MLModel { try MLModel(contentsOf: url, configuration: config) }
  static func predict(_ m: MLModel, _ i: MLFeatureProvider) throws -> MLFeatureProvider { try m.prediction(from: i) }

  /// The packed narrow frame the comma sends (four Y sub-planes, then U and V at 256x128): a synthetic road.
  static func frame() -> Data {
    let (w, h) = (256, 128)
    var d = Data(count: 6 * w * h)
    d.withUnsafeMutableBytes { (p: UnsafeMutableRawBufferPointer) in
      for plane in 0..<4 {
        for r in 0..<h {
          let row = 2 * r + (plane == 1 || plane == 3 ? 1 : 0)
          for c in 0..<w { p[plane * w * h + r * w + c] = row < 60 ? 190 : (row >= 170 && row < 176) ? 230 : UInt8(90 + (row + c) % 20) }
        }
      }
      for i in (4 * w * h)..<(6 * w * h) { p[i] = 128 }
    }
    return d
  }

  /// The model's inputs from a packed frame, as StopLineModel builds them; with the wide camera's inputs too (the
  /// same frame again) for a model that takes them.
  static func inputs(_ packed: Data, wide: Bool) throws -> MLFeatureProvider {
    let (luma, uv) = try planes(packed)
    var d: [String: MLFeatureValue] = ["frame": MLFeatureValue(pixelBuffer: luma), "uv": MLFeatureValue(multiArray: uv)]
    if wide {
      let (l2, uv2) = try planes(packed)
      d["frame_wide"] = MLFeatureValue(pixelBuffer: l2)
      d["uv_wide"] = MLFeatureValue(multiArray: uv2)
    }
    return try MLDictionaryFeatureProvider(dictionary: d)
  }

  static func planes(_ packed: Data) throws -> (CVPixelBuffer, MLMultiArray) {
    var pb: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, 512, 256, kCVPixelFormatType_OneComponent8, nil, &pb)
    guard let luma = pb else { throw BenchError.noOutput }
    fill(luma, StopLineDecoder.luma(packed))
    let uv = try MLMultiArray(shape: [1, 2, 128, 256], dataType: .float16)
    fill(uv, packed)
    return (luma, uv)
  }

  static func fill(_ luma: CVPixelBuffer, _ y: [UInt8]) {
    CVPixelBufferLockBaseAddress(luma, [])
    let base = CVPixelBufferGetBaseAddress(luma)!.assumingMemoryBound(to: UInt8.self)
    let stride = CVPixelBufferGetBytesPerRow(luma)
    y.withUnsafeBufferPointer { src in
      for r in 0..<256 { (base + r * stride).update(from: src.baseAddress! + r * 512, count: 512) }
    }
    CVPixelBufferUnlockBaseAddress(luma, [])
  }

  static func fill(_ uv: MLMultiArray, _ packed: Data) {
    let plane = 128 * 256
    let dst = uv.dataPointer.assumingMemoryBound(to: Float16.self)
    packed.withUnsafeBytes { (p: UnsafeRawBufferPointer) in
      for i in 0..<(2 * plane) { dst[i] = Float16(p[4 * plane + i]) }
    }
  }

  static func placement(_ url: URL, _ units: MLComputeUnits) async -> [String: Any] {
    let config = MLModelConfiguration()
    config.computeUnits = units
    do {
      let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)
      guard case .program(let program) = plan.modelStructure, let main = program.functions["main"] else { return ["error": "not an ML program"] }
      var ops: [String: Int] = [:], cost: [String: Double] = [:], fallback: [String: Int] = [:]
      for op in main.block.operations {
        let d: String
        switch plan.deviceUsage(for: op)?.preferred {
        case .neuralEngine: d = "ANE"
        case .gpu: d = "GPU"
        case .cpu: d = "CPU"
        default: continue  // a constant: no device runs it
        }
        ops[d, default: 0] += 1
        cost[d, default: 0] += plan.estimatedCost(of: op)?.weight ?? 0
        if d != "ANE" { fallback["\(d): \(op.operatorName)", default: 0] += 1 }
      }
      return ["ops": ops, "costShare": cost, "notOnANE": fallback]
    } catch {
      return ["error": "\(error)"]
    }
  }
}

/// StopLineModel.run's step for a model that also takes the wide camera: the same packing, prediction and decoding,
/// with frame_wide and uv_wide filled from the same frame.
final class WideStep: @unchecked Sendable {
  private let model: MLModel
  private let luma: CVPixelBuffer, lumaWide: CVPixelBuffer
  private let uv: MLMultiArray, uvWide: MLMultiArray

  init(url: URL) throws {
    let config = MLModelConfiguration()
    config.computeUnits = .all
    model = try Bench.load(url, config)
    (luma, uv) = try Bench.planes(Bench.frame())
    (lumaWide, uvWide) = try Bench.planes(Bench.frame())
  }

  func run(_ packed: Data) throws {
    let y = StopLineDecoder.luma(packed)
    Bench.fill(luma, y)
    Bench.fill(uv, packed)
    Bench.fill(lumaWide, y)
    Bench.fill(uvWide, packed)
    let input = try MLDictionaryFeatureProvider(dictionary: [
      "frame": MLFeatureValue(pixelBuffer: luma), "uv": MLFeatureValue(multiArray: uv),
      "frame_wide": MLFeatureValue(pixelBuffer: lumaWide), "uv_wide": MLFeatureValue(multiArray: uvWide),
    ])
    let out = try Bench.predict(model, input)
    guard let line = out.featureValue(for: "line")?.multiArrayValue, let lead = out.featureValue(for: "lead")?.multiArrayValue else {
      throw Bench.BenchError.noOutput
    }
    _ = StopLineDecoder.decode(line: Self.floats(line), lead: Self.floats(lead), gh: line.shape[2].intValue, gw: line.shape[3].intValue)
  }

  /// StopLineModel.floats, which the package keeps internal.
  static func floats(_ a: MLMultiArray) -> [Float] {
    let shape = a.shape.map(\.intValue), strides = a.strides.map(\.intValue)
    let n = shape.reduce(1, *)
    var out = [Float](repeating: 0, count: n)
    a.withUnsafeBytes { raw in
      for i in 0..<n {
        var rem = i, off = 0
        for d in stride(from: shape.count - 1, through: 0, by: -1) {
          off += (rem % shape[d]) * strides[d]
          rem /= shape[d]
        }
        switch a.dataType {
        case .float16: out[i] = Float(raw.load(fromByteOffset: 2 * off, as: Float16.self))
        case .float32: out[i] = raw.load(fromByteOffset: 4 * off, as: Float.self)
        default: out[i] = Float(truncating: a[off])
        }
      }
    }
    return out
  }
}
