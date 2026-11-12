#if canImport(SwiftUI)
  import DioramaCore
  import DioramaLink
  import Foundation
  import Observation

  /// The comma over the link: each frame it shows becomes the scene's drive message, and what the
  /// stop-line model sees in it the scene's segmentation and stop-line messages. Dials again a second
  /// after the link drops, for as long as it is started.
  @MainActor
  @Observable
  public final class CommaFeed: DioramaFeed {
    /// A comma has said hello and is streaming the drive: what keeps the screen on.
    public private(set) var streaming = false
    /// Whether the drive is dark, as the Auto Detect theme takes it (FootageTone): true at night, false by day, nil
    /// until the footage has said for long enough.
    public private(set) var footageDark: Bool?
    public let host: String
    private let name: String
    public let model: StopLineModel?
    private var task: Task<Void, Never>?

    /// `host`: the comma's address, or the Mac standing in for it.
    public init(host: String = Wire.commaHost, name: String, model: StopLineModel?) {
      (self.host, self.name, self.model) = (host, name, model)
    }

    public func start(input: Bool, deliver: @escaping @MainActor @Sendable ([Data]) async -> Void) {
      stop()
      let (host, name, model) = (host, name, model)
      let streaming: @MainActor @Sendable (Bool) -> Void = { [weak self] on in self?.streaming = on }
      let footageDark: @MainActor @Sendable (Bool) -> Void = { [weak self] d in self?.footageDark = d }
      task = Task.detached(priority: .userInitiated) {
        let runner = StopLineRunner(model: model)
        var tone = FootageTone()
        while !Task.isCancelled {
          await deliver([DioramaPackets.status("dialing the comma at \(host)")])
          let link = CommaLink(host: host, name: name)
          var start: Date? = nil
          var waiting: [Data] = []
          for await event in link.events() {
            if Task.isCancelled { break }
            switch event {
            case .hello(let layout, let camera, _, let comma):
              await streaming(true)
              start = Date()
              var meta: [String: Any] = [
                "version": 1, "fps": 20, "live": true,
                "frame": ["w": 2 * layout.width, "h": 2 * layout.height, "f": 910.0, "cx": Double(layout.width), "cy": 47.6],
                "drive": layout.outputs.map { [$0.name, $0.count] },
                "seg": StopLineMessages.segMeta,
                "source": ["route": comma],
              ]
              if let h = camera["height"] { meta["height"] = h }
              waiting = [DioramaPackets.meta(meta)]
              waiting.append(
                DioramaPackets.status(
                  camera["height"] == nil ? "waiting for the comma's calibration" : model == nil ? "off: the stop-line model is not downloaded yet" : "running"))
            case .notShowing:
              waiting.append(DioramaPackets.status("off: this comma's zoompilot does not stream for Diorama"))
            case .frame(let f):
              let seconds = Float(Date().timeIntervalSince(start ?? Date()))
              waiting.append(DioramaPackets.drive(frame: f.id, seconds: seconds, modelMs: 0, values: f.outputs))
              if input { waiting.append(DioramaPackets.input(frame: f.id, packed: [UInt8](f.packed))) }
              waiting += await runner.see(f)
              if let d = tone.see(f.packed) { await footageDark(d) }
            case .closed(let why):
              await streaming(false)
              waiting.append(DioramaPackets.status("the link closed: \(why)"))
            }
            if !waiting.isEmpty {
              await deliver(waiting)
              waiting = []
            }
          }
          link.close()
          await streaming(false)
          try? await Task.sleep(for: .seconds(1))
        }
      }
    }

    public func stop() {
      task?.cancel()
      task = nil
      streaming = false
    }
  }

  /// Day or night from the footage, for the Auto Detect theme. It reads the sky: the median luma of the frame's
  /// top third, where the night sky is black even when headlights light the road and the comma's camera turns
  /// its exposure up. Each second's frames give one reading, and the answer goes by the average of the last two
  /// minutes of readings (of what there is, from the tenth second on): daylight changes over many minutes, the
  /// view over seconds.
  ///
  /// The thresholds, from recorded drives: a night drive reads 0.11 to 0.15, day drives 0.25 to 0.43 with the sky
  /// in view, and 0.17 to 0.18 facing a fence in a car park, which is between the two and changes nothing. Around
  /// sunset the readings swing across both thresholds for minutes as trees, buildings and open sky go by: on two
  /// recorded sunsets the readings themselves flipped the answer 3 and 7 times; their two-minute average, once
  /// and not at all.
  public struct FootageTone: Sendable {
    public static let darkBelow = 0.16, lightAbove = 0.21
    static let framesPerReading = 20  // a second at the comma's 20 Hz
    /// The readings averaged, a second each; and how many there have to be before the first answer.
    /// Ten, so the first answer does not go by a few seconds that happen to be unlike the rest.
    public static let windowSeconds = 120, firstAfter = 10

    public private(set) var dark: Bool?
    private var sum = 0.0, frames = 0
    private var window: [Double] = []

    public init() {}

    /// One packed frame; the answer when it changes.
    public mutating func see(_ packed: Data) -> Bool? {
      guard let m = Self.skyMedian(packed) else { return nil }
      sum += m
      frames += 1
      guard frames == Self.framesPerReading else { return nil }
      defer { (sum, frames) = (0, 0) }
      return read(sum / Double(frames))
    }

    /// One second's reading; the answer when it changes.
    public mutating func read(_ reading: Double) -> Bool? {
      window.append(reading)
      if window.count > Self.windowSeconds { window.removeFirst() }
      guard window.count >= Self.firstAfter else { return nil }
      let level = window.reduce(0, +) / Double(window.count)
      let side: Bool? = level < Self.darkBelow ? true : level > Self.lightAbove ? false : nil
      guard let side, side != dark else { return nil }
      dark = side
      return side
    }

    /// The median of a sparse sample of the top third of the frame's luma (the packed frame's first Y sub-plane,
    /// 256 x 128: its top 43 rows), 0 to 1.
    static func skyMedian(_ packed: Data) -> Double? {
      let (w, rows) = (256, 43)
      guard packed.count >= 256 * 128 else { return nil }
      var counts = [Int](repeating: 0, count: 256)
      var n = 0
      packed.withUnsafeBytes { (p: UnsafeRawBufferPointer) in
        for i in Swift.stride(from: 0, to: w * rows, by: 7) {
          counts[Int(p[i])] += 1
          n += 1
        }
      }
      var below = 0
      for (v, c) in counts.enumerated() {
        below += c
        if 2 * below >= n { return Double(v) / 255 }
      }
      return nil
    }
  }

  /// The stop-line model off the link's thread: a frame that arrives while it runs is skipped, and the
  /// messages of the last run go out with the next frame's drive message, so the scene has the pose
  /// they were seen from before it reads them.
  actor StopLineRunner {
    private let model: StopLineModel?
    private var busy = false
    private var done: [Data] = []

    init(model: StopLineModel?) { self.model = model }

    func see(_ f: ShownFrame) -> [Data] {
      let out = done
      done = []
      if let model, !busy {
        busy = true
        Task.detached(priority: .userInitiated) {
          let msgs = (try? model.run(f.packed)).map { StopLineMessages.messages(frame: f.id, ms: $0.ms, line: $0.line) } ?? []
          await self.finished(msgs)
        }
      }
      return out
    }

    private func finished(_ msgs: [Data]) {
      done += msgs
      busy = false
    }
  }
#endif
