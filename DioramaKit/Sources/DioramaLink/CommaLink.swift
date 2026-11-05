import Foundation
import Network

/// One session with the comma: the phone dials it, answers its hello as a
/// viewer, and hands on each frame it shows. The comma never waits for the
/// phone: shown frames get no answer, and only a ping does.
public final class CommaLink: @unchecked Sendable {
  public enum Event: Sendable {
    /// The comma's hello: what it streams, its calibration (`camera`), and its name.
    case hello(ShowLayout, camera: [String: Double], wideFromDevice: [Double]?, name: String)
    case frame(ShownFrame)
    /// The comma said hello without `show`: a zoompilot that does not stream for Diorama.
    case notShowing
    case closed(String)
  }

  public let host: String, port: UInt16
  private let name: String
  private let queue = DispatchQueue(label: "diorama.link")
  private var connection: NWConnection?
  private var layout: ShowLayout?
  private var continuation: AsyncStream<Event>.Continuation?

  public init(host: String = Wire.commaHost, port: UInt16 = Wire.port, name: String) {
    (self.host, self.port, self.name) = (host, port, name)
  }

  /// Dials the comma; the stream ends when the session does.
  public func events() -> AsyncStream<Event> {
    AsyncStream(bufferingPolicy: .bufferingNewest(8)) { cont in
      queue.async { self.open(cont) }
      cont.onTermination = { [weak self] _ in self?.queue.async { self?.close() } }
    }
  }

  public func close() {
    connection?.cancel()
    connection = nil
  }

  private func open(_ cont: AsyncStream<Event>.Continuation) {
    continuation = cont
    let tcp = NWProtocolTCP.Options()
    tcp.noDelay = true
    tcp.connectionTimeout = 3
    let params = NWParameters(tls: nil, tcp: tcp)
    // over the cable's network interface, never Wi-Fi or cellular, unless dialing the Mac standing in for the comma
    if host == Wire.commaHost { params.requiredInterfaceType = .wiredEthernet }
    let c = NWConnection(host: NWEndpoint.Host(host), port: NWEndpoint.Port(rawValue: port)!, using: params)
    connection = c
    c.stateUpdateHandler = { [weak self] state in
      switch state {
      case .ready: self?.readHeader()
      case .failed(let e): self?.finish("\(e)")
      case .waiting(let e): self?.finish("\(e)")
      case .cancelled: self?.finish("closed")
      default: break
      }
    }
    c.start(queue: queue)
  }

  private func finish(_ why: String) {
    guard let cont = continuation else { return }
    continuation = nil
    cont.yield(.closed(why))
    cont.finish()
    connection?.cancel()
  }

  private func read(_ n: Int, _ then: @escaping (Data) -> Void) {
    guard let c = connection else { return }
    c.receive(minimumIncompleteLength: n, maximumLength: n) { [weak self] data, _, done, error in
      guard let self else { return }
      if let error { return self.finish("\(error)") }
      guard let data, data.count == n else { return self.finish(done ? "the comma closed the link" : "short read") }
      then(data)
    }
  }

  private func readHeader() {
    read(Wire.headerSize) { [weak self] bytes in
      guard let self else { return }
      do {
        let h = try Wire.parseHeader(bytes)
        let total = h.length + (h.flags & Wire.padded != 0 ? 1 : 0)
        if total == 0 { return self.take(h, Data()) }
        self.read(total) { body in self.take(h, body.prefix(h.length)) }
      } catch {
        self.finish("broken stream: \(error)")
      }
    }
  }

  private func take(_ h: Wire.Header, _ payload: Data) {
    switch Wire.Msg(rawValue: h.type) {
    case .helloReq: hello(h.seq, payload)
    case .inferReq:
      if let layout, let frame = ShownFrame(payload: payload, layout: layout) { continuation?.yield(.frame(frame)) }
    case .ping: send(Wire.frame(.pong, seq: h.seq))
    case .stateReq: send(Wire.json(.stateResp, seq: h.seq, ["viewer": true]))
    case .engineReq:
      // a viewer runs no driving model
      send(Wire.json(.engineResp, seq: h.seq, ["state": "failed", "error": "viewer"]))
    case .leave, .none, .some: break
    }
    readHeader()
  }

  private func hello(_ seq: UInt32, _ payload: Data) {
    let json = (try? JSONSerialization.jsonObject(with: payload)) as? [String: Any]
    let client = json?["client"] as? [String: Any] ?? [:]
    send(Wire.json(.helloResp, seq: seq, ["viewer": true, "name": name, "protocol": Int(Wire.version)]))
    guard let layout = ShowLayout(json: client["show"]) else {
      self.layout = nil
      continuation?.yield(.notShowing)
      return
    }
    self.layout = layout
    var camera: [String: Double] = [:]
    var wide: [Double]?
    if let cam = client["camera"] as? [String: Any] {
      if let h = (cam["height"] as? NSNumber)?.doubleValue { camera["height"] = h }
      wide = (cam["wide_from_device_euler"] as? [NSNumber])?.map(\.doubleValue)
    }
    continuation?.yield(.hello(layout, camera: camera, wideFromDevice: wide, name: client["name"] as? String ?? "comma"))
  }

  private func send(_ data: Data) {
    connection?.send(content: data, completion: .contentProcessed { _ in })
  }
}
