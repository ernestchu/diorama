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
    public let host: String
    private let name: String
    private let model: StopLineModel?
    private var task: Task<Void, Never>?

    /// `host`: the comma's address, or the Mac standing in for it.
    public init(host: String = Wire.commaHost, name: String, model: StopLineModel?) {
      (self.host, self.name, self.model) = (host, name, model)
    }

    public func start(input: Bool, deliver: @escaping @MainActor @Sendable ([Data]) async -> Void) {
      stop()
      let (host, name, model) = (host, name, model)
      let streaming: @MainActor @Sendable (Bool) -> Void = { [weak self] on in self?.streaming = on }
      task = Task.detached(priority: .userInitiated) {
        let runner = StopLineRunner(model: model)
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
                  camera["height"] == nil ? "waiting for the comma's calibration" : model == nil ? "off: no stop-line model in this build" : "running"))
            case .notShowing:
              waiting.append(DioramaPackets.status("off: this comma's zoompilot does not stream for Diorama"))
            case .frame(let f):
              let seconds = Float(Date().timeIntervalSince(start ?? Date()))
              waiting.append(DioramaPackets.drive(frame: f.id, seconds: seconds, modelMs: 0, values: f.outputs))
              if input { waiting.append(DioramaPackets.input(frame: f.id, packed: [UInt8](f.packed))) }
              waiting += await runner.see(f)
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
          let msgs = (try? model.run(f.packed)).map { StopLineMessages.messages(frame: f.id, ms: $0.ms, line: $0.line, lead: $0.lead) } ?? []
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
