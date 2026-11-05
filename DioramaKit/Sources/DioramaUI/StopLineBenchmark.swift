#if canImport(CoreML)
  import DioramaLink
  import Foundation
  import Observation

  /// The stop-line model at the comma's pace on this device alone: a frame every 50 ms for a minute or ten, each
  /// timed through everything the app does with one (the frame unpacked, the model, the decode). The run's layout
  /// follows jetlink's Benchmark (Zeph Leggett, MIT): the verdict, the totals, and the run ten seconds at a time.
  @MainActor
  @Observable
  public final class StopLineBenchmark {
    public nonisolated static let budgetMs = 50.0  // 20 Hz
    public nonisolated static let tightMs = 35.0
    public nonisolated static let windowSeconds = 10.0

    public struct Window: Identifiable, Sendable {
      public let id: Int
      public let frames: Int, p50: Double, p99: Double, max: Double, over: Int
      public let thermal: ProcessInfo.ThermalState
    }

    public struct Report: Sendable {
      public let seconds: Double, total: Double
      public let frames: Int, mean: Double, p50: Double, p99: Double, max: Double
      public let over: Int, tight: Int
      public let thermalAtStart: ProcessInfo.ThermalState, thermalAtEnd: ProcessInfo.ThermalState
      public let windows: [Window]
      public let cancelled: Bool
      /// Where Core ML ran the model, when it said.
      public let placement: StopLineModel.Placement?

      public enum Verdict: Sendable { case fast, tight, slow }
      /// Slow: frames past the 50 ms budget, more than one in a thousand. Tight: the 99th percentile past 35 ms.
      public var verdict: Verdict {
        if frames == 0 || Double(over) > Double(frames) / 1000 { return .slow }
        return p99 > StopLineBenchmark.tightMs ? .tight : .fast
      }

      public var text: String {
        let f = { (v: Double) in String(format: "%.1f ms", v) }
        var lines = [
          "Diorama stop-line benchmark\(cancelled ? " (stopped early)" : "")",
          "\(frames) frames in \(Int(seconds.rounded())) s at 20 Hz",
          "mean \(f(mean)), p50 \(f(p50)), p99 \(f(p99)), max \(f(max))",
          "over 50 ms: \(over), over 35 ms: \(tight)",
          "thermal: \(thermalAtStart.label) to \(thermalAtEnd.label)",
        ]
        if let placement { lines.append("runs on: \(placement.text) (Core ML)") }
        for w in windows {
          lines.append("\(w.id * 10)–\(w.id * 10 + 10) s: p50 \(f(w.p50)), p99 \(f(w.p99)), max \(f(w.max)), over \(w.over), \(w.thermal.label)")
        }
        return lines.joined(separator: "\n")
      }
    }

    public private(set) var running = false
    public private(set) var elapsed = 0.0
    public private(set) var total = 0.0
    public private(set) var frames = 0
    public private(set) var p50: Double?
    public private(set) var p99: Double?
    public private(set) var report: Report?
    public private(set) var failure: String?
    /// Where Core ML runs the model on this device; nil until loaded, or when Core ML does not say.
    public private(set) var placement: StopLineModel.Placement?
    private var placementLoaded = false

    private var task: Task<Void, Never>?

    public init() {}

    /// A run of `seconds`, on a model of its own: the live view's is left alone.
    public func start(seconds: Double) {
      guard !running else { return }
      failure = nil
      report = nil
      running = true
      elapsed = 0
      total = seconds
      frames = 0
      p50 = nil
      p99 = nil
      task = Task { await run(seconds: seconds) }
    }

    public func cancel() { task?.cancel() }

    /// Asks Core ML once where it runs the model here.
    public func loadPlacement() async {
      guard !placementLoaded else { return }
      placementLoaded = true
      placement = await StopLineModel.placement()
    }

    private func run(seconds: Double) async {
      defer { running = false }
      let loaded = await Task.detached(priority: .userInitiated) { StopLineModel.bundled() }.value
      guard let model = loaded else {
        failure = "There is no stop-line model in this build."
        return
      }
      let frame = Self.sampleFrame()
      // one untimed frame: the first prediction compiles for the device
      _ = try? await Task.detached(priority: .userInitiated) { try model.run(frame) }.value

      let thermalAtStart = ProcessInfo.processInfo.thermalState
      let clock = ContinuousClock()
      let start = clock.now
      var times: [Double] = []
      var windows: [Window] = []
      var windowTimes: [Double] = []
      var tick = 0
      times.reserveCapacity(Int(seconds * 20))
      while !Task.isCancelled {
        let due = start + .milliseconds(50 * tick)
        if due > clock.now { try? await clock.sleep(until: due) }
        if Task.isCancelled { break }
        let t0 = clock.now
        let ok = (try? await Task.detached(priority: .userInitiated) { try model.run(frame) }.value) != nil
        let ms = (clock.now - t0).milliseconds
        guard ok else {
          failure = "The model failed on a frame."
          return
        }
        times.append(ms)
        windowTimes.append(ms)
        tick += 1
        elapsed = (clock.now - start).seconds
        frames = times.count
        if frames % 10 == 0 {
          let sorted = times.sorted()
          p50 = Self.percentile(sorted, 0.5)
          p99 = Self.percentile(sorted, 0.99)
        }
        if Double(tick) * 0.05 >= Double(windows.count + 1) * Self.windowSeconds {
          windows.append(Self.window(windows.count, windowTimes))
          windowTimes = []
        }
        if elapsed >= seconds { break }
      }
      if windowTimes.count >= 20 { windows.append(Self.window(windows.count, windowTimes)) }
      let sorted = times.sorted()
      report = Report(
        seconds: elapsed, total: seconds, frames: times.count, mean: times.isEmpty ? 0 : times.reduce(0, +) / Double(times.count),
        p50: Self.percentile(sorted, 0.5), p99: Self.percentile(sorted, 0.99), max: sorted.last ?? 0,
        over: times.filter { $0 > Self.budgetMs }.count, tight: times.filter { $0 > Self.tightMs }.count,
        thermalAtStart: thermalAtStart, thermalAtEnd: ProcessInfo.processInfo.thermalState, windows: windows,
        cancelled: Task.isCancelled, placement: placement)
    }

    private nonisolated static func window(_ i: Int, _ t: [Double]) -> Window {
      let s = t.sorted()
      return Window(
        id: i, frames: t.count, p50: percentile(s, 0.5), p99: percentile(s, 0.99), max: s.last ?? 0, over: t.filter { $0 > budgetMs }.count,
        thermal: ProcessInfo.processInfo.thermalState)
    }

    nonisolated static func percentile(_ sorted: [Double], _ q: Double) -> Double {
      guard !sorted.isEmpty else { return 0 }
      return sorted[min(sorted.count - 1, Int((Double(sorted.count - 1) * q).rounded()))]
    }

    /// A packed narrow frame to time the model on: a road-like picture (sky, then a grey road with a white bar
    /// across it), so the decode has a peak to find as it would on a drive.
    nonisolated static func sampleFrame() -> Data {
      let (w, h) = (256, 128)
      var d = Data(count: 6 * w * h)
      d.withUnsafeMutableBytes { (p: UnsafeMutableRawBufferPointer) in
        for plane in 0..<4 {
          for r in 0..<h {
            let row = 2 * r + (plane == 1 || plane == 3 ? 1 : 0)  // the full frame's row
            for c in 0..<w {
              let sky = row < 60, bar = row >= 170 && row < 176
              p[plane * w * h + r * w + c] = sky ? 190 : bar ? 230 : UInt8(90 + (row + c) % 20)
            }
          }
        }
        for i in (4 * w * h)..<(6 * w * h) { p[i] = 128 }
      }
      return d
    }
  }

  extension Duration {
    var seconds: Double { Double(components.seconds) + Double(components.attoseconds) / 1e18 }
    var milliseconds: Double { seconds * 1000 }
  }

  extension ProcessInfo.ThermalState {
    public var label: String {
      switch self {
      case .nominal: "nominal"
      case .fair: "fair"
      case .serious: "serious"
      case .critical: "critical"
      @unknown default: "unknown"
      }
    }
  }
#endif
