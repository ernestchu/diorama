// The framing follows jetlink's link protocol 3 (jetlink/protocol.py,
// Copyright (c) 2026-, Zeph Leggett, MIT), which zoompilot speaks to a phone
// over the cable. Diorama adds Flag.show and the hello's `show` and `viewer`.
import Foundation

/// jetlink's link protocol 3, as much of it as a viewer needs.
public enum Wire {
  public static let magic: UInt32 = 0x4B4E_4C4A  // "JLNK"
  public static let version: UInt16 = 3
  public static let headerSize = 32
  /// A message whose header and payload are an exact multiple of this ends
  /// with one pad byte, flagged `padded`, so a USB bulk read always ends short.
  public static let packetMultiple = 512
  /// The comma at the far end of the iPhone's cable.
  public static let commaHost = "192.168.60.1"
  public static let port: UInt16 = 5599

  public enum Msg: UInt16, Sendable {
    case helloReq = 1, helloResp = 2, engineReq = 3, engineResp = 4
    case inferReq = 8, inferResp = 9
    case stateReq = 12, stateResp = 13, error = 14, ping = 15, pong = 16, leave = 19
  }

  /// The header's flags.
  public static let padded: UInt32 = 1 << 7
  /// INFER_REQ's own flags (its first payload word after frame_id).
  public static let resetQueues: UInt32 = 1 << 0
  /// The frame is shown, not inferred: the packed narrow frame and the driving
  /// model's outputs follow, and nothing is answered. Diorama's.
  public static let show: UInt32 = 1 << 3

  public struct Header: Equatable, Sendable {
    public var type: UInt16, seq: UInt32, flags: UInt32, length: Int
    public init(type: UInt16, seq: UInt32, flags: UInt32 = 0, length: Int) {
      (self.type, self.seq, self.flags, self.length) = (type, seq, flags, length)
    }
  }

  public enum Failure: Error, Equatable {
    case badMagic(UInt32), badVersion(UInt16), tooLong(Int)
  }

  /// Nothing the comma sends is near this; a length past it is a broken stream.
  static let maxLength = 8 << 20

  public static func parseHeader(_ b: Data) throws -> Header {
    precondition(b.count >= headerSize)
    let m: UInt32 = b.le(0), v: UInt16 = b.le(4)
    guard m == magic else { throw Failure.badMagic(m) }
    guard v == version else { throw Failure.badVersion(v) }
    let length = Int(b.le(16) as UInt32)
    guard length <= maxLength else { throw Failure.tooLong(length) }
    return Header(type: b.le(6), seq: b.le(8), flags: b.le(12), length: length)
  }

  /// One message: the header, the payload, and the pad byte where it needs one.
  public static func frame(_ type: Msg, seq: UInt32, payload: Data = Data()) -> Data {
    var flags: UInt32 = 0
    let pad = (headerSize + payload.count) % packetMultiple == 0
    if pad { flags |= padded }
    var d = Data(capacity: headerSize + payload.count + 1)
    d.appendLE(magic)
    d.appendLE(version)
    d.appendLE(type.rawValue)
    d.appendLE(seq)
    d.appendLE(flags)
    d.appendLE(UInt32(payload.count))
    d.appendLE(UInt64(0))
    d.append(contentsOf: [0, 0, 0, 0])
    d.append(payload)
    if pad { d.append(0) }
    return d
  }

  public static func json(_ type: Msg, seq: UInt32, _ object: [String: Any]) -> Data {
    frame(type, seq: seq, payload: (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])) ?? Data("{}".utf8))
  }
}

/// What the comma streams, as its hello lays it out.
public struct ShowLayout: Equatable, Sendable {
  /// planes, height, width of the packed narrow frame
  public var planes = 6, height = 128, width = 256
  /// The driving model's heads, in order, with their float counts.
  public var outputs: [Field] = []
  public struct Field: Equatable, Sendable {
    public var name: String, count: Int
    public init(name: String, count: Int) { (self.name, self.count) = (name, count) }
  }

  public var frameBytes: Int { planes * height * width }
  public var outputFloats: Int { outputs.reduce(0) { $0 + $1.count } }

  public init(planes: Int = 6, height: Int = 128, width: Int = 256, outputs: [Field]) {
    (self.planes, self.height, self.width, self.outputs) = (planes, height, width, outputs)
  }

  /// From the hello's `show`; nil when it has none or a wrong one.
  public init?(json: Any?) {
    guard let show = json as? [String: Any], let frame = show["frame"] as? [Int], frame.count == 3,
      let outs = show["outputs"] as? [[Any]], !outs.isEmpty
    else { return nil }
    (planes, height, width) = (frame[0], frame[1], frame[2])
    var fields: [Field] = []
    for o in outs {
      guard o.count == 2, let name = o[0] as? String, let n = o[1] as? Int, n > 0 else { return nil }
      fields.append(Field(name: name, count: n))
    }
    outputs = fields
    guard planes > 0, height > 0, width > 0 else { return nil }
  }

  public var json: [String: Any] { ["frame": [planes, height, width], "outputs": outputs.map { [$0.name, $0.count] }] }
}

/// One frame the comma showed.
public struct ShownFrame: Sendable {
  public let id: UInt32
  /// The packed narrow frame: `planes` planes of `height` x `width`, uint8.
  public let packed: Data
  public let outputs: [Float]

  /// An INFER_REQ's payload with `show` set; nil when it is not one, or does not match the layout.
  public init?(payload p: Data, layout: ShowLayout) {
    guard p.count >= 8 else { return nil }
    let flags: UInt32 = p.le(4)
    guard flags & Wire.show != 0, p.count == 8 + layout.frameBytes + 4 * layout.outputFloats else { return nil }
    id = p.le(0)
    let start = p.startIndex + 8
    packed = p.subdata(in: start..<start + layout.frameBytes)
    let o = start + layout.frameBytes
    outputs = p.subdata(in: o..<p.endIndex).withUnsafeBytes { raw in
      (0..<layout.outputFloats).map { Float(bitPattern: raw.loadUnaligned(fromByteOffset: 4 * $0, as: UInt32.self).littleEndian) }
    }
  }
}

extension Data {
  func le<T: FixedWidthInteger>(_ at: Int) -> T {
    withUnsafeBytes { T(littleEndian: $0.loadUnaligned(fromByteOffset: at, as: T.self)) }
  }

  mutating func appendLE<T: FixedWidthInteger>(_ value: T) {
    Swift.withUnsafeBytes(of: value.littleEndian) { append(contentsOf: $0) }
  }
}
