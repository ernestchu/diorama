import Foundation

/// The stop-line model's maps, decoded (diorama/stopline.py's `decode`, for the
/// Core ML export, whose heat maps are already through a sigmoid). The maps are
/// at stride 4 of the 512x256 medmodel frame; what comes out is in its pixels.
public enum StopLineDecoder {
  public static let lineScore: Float = 0.3, leadScore: Float = 0.3
  public static let stride: Float = 4, endScale: Float = 32, whScale: Float = 16

  public struct Line: Equatable, Sendable {
    public var score: Float, x1: Float, y1: Float, x2: Float, y2: Float
  }

  /// The lead's box: score, then its corners.
  public struct Box: Equatable, Sendable {
    public var score: Float, x1: Float, y1: Float, x2: Float, y2: Float
  }

  /// `line` (8, gh, gw) and `lead` (5, gh, gw), channel-major, as floats.
  /// At most one line (the top peak, the maximum of its 5x5 neighbourhood), and the lead's box.
  public static func decode(line: [Float], lead: [Float], gh: Int, gw: Int) -> (Line?, Box?) {
    let n = gh * gw
    precondition(line.count >= 7 * n && lead.count >= 5 * n)
    func at(_ m: [Float], _ c: Int, _ y: Int, _ x: Int) -> Float { m[c * n + y * gw + x] }

    // the top peak: the highest cell that is also the maximum of its 5x5 neighbourhood
    var best = -1
    var bestScore: Float = -1
    for y in 0..<gh {
      for x in 0..<gw {
        let v = line[y * gw + x]
        guard v > bestScore else { continue }
        var isPeak = true
        search: for dy in -2...2 {
          let yy = y + dy
          guard yy >= 0 && yy < gh else { continue }
          for dx in -2...2 {
            let xx = x + dx
            if xx >= 0 && xx < gw && line[yy * gw + xx] > v {
              isPeak = false
              break search
            }
          }
        }
        if isPeak {
          best = y * gw + x
          bestScore = v
        }
      }
    }
    var found: Line? = nil
    if best >= 0 && bestScore >= lineScore {
      let y = best / gw, x = best % gw
      let cx = Float(x) + at(line, 1, y, x), cy = Float(y) + at(line, 2, y, x)
      let e = (3...6).map { at(line, $0, y, x) * endScale }
      found = Line(
        score: bestScore, x1: (cx + e[0]) * stride, y1: (cy + e[1]) * stride, x2: (cx + e[2]) * stride, y2: (cy + e[3]) * stride)
    }

    var arg = 0
    for i in 1..<n where lead[i] > lead[arg] { arg = i }
    var box: Box? = nil
    if lead[arg] >= leadScore {
      let y = arg / gw, x = arg % gw
      let cx = Float(x) + at(lead, 1, y, x), cy = Float(y) + at(lead, 2, y, x)
      let w = at(lead, 3, y, x) * whScale, h = at(lead, 4, y, x) * whScale
      box = Box(score: lead[arg], x1: (cx - w / 2) * stride, y1: (cy - h / 2) * stride, x2: (cx + w / 2) * stride, y2: (cy + h / 2) * stride)
    }
    return (found, box)
  }

  /// The packed narrow frame's four Y sub-planes as the 512x256 luma image the model takes.
  public static func luma(_ packed: Data, width: Int = 256, height: Int = 128) -> [UInt8] {
    let W = 2 * width
    var y = [UInt8](repeating: 0, count: W * 2 * height)
    packed.withUnsafeBytes { (p: UnsafeRawBufferPointer) in
      let plane = width * height
      for r in 0..<height {
        for c in 0..<width {
          let i = r * width + c
          y[(2 * r) * W + 2 * c] = p[i]  // p[0]: even rows, even columns
          y[(2 * r + 1) * W + 2 * c] = p[plane + i]  // p[1]: odd rows, even columns
          y[(2 * r) * W + 2 * c + 1] = p[2 * plane + i]  // p[2]: even rows, odd columns
          y[(2 * r + 1) * W + 2 * c + 1] = p[3 * plane + i]  // p[3]: odd rows, odd columns
        }
      }
    }
    return y
  }
}

/// The scene's messages for what the stop-line model saw in a frame: the lead box as one Car in a segmentation
/// (`S`, at half the frame's resolution), which the scene's lead check reads, and the stop line (`L`).
public enum StopLineMessages {
  public static let ds = 2
  /// The segmentation meta the scene reads them with.
  public static var segMeta: [String: Any] { ["labels": ["Car"], "road_class": [String: Int](), "things": ["Car": "car"], "ds": ds, "narrow_only": true] }

  public static func messages(frame: UInt32, ms: Float, line: StopLineDecoder.Line?, lead: StopLineDecoder.Box?) -> [Data] {
    let w = 512 / ds, h = 256 / ds
    var qmap = [UInt8](repeating: 255, count: w * h)
    if let b = lead {
      let x1 = max(Int((b.x1 / Float(ds)).rounded()), 0), x2 = min(max(Int((b.x2 / Float(ds)).rounded()), 0), w)
      let y1 = max(Int((b.y1 / Float(ds)).rounded()), 0), y2 = min(max(Int((b.y2 / Float(ds)).rounded()), 0), h)
      if x1 < x2 && y1 < y2 { for r in y1..<y2 { for c in x1..<x2 { qmap[r * w + c] = 0 } } }
    }
    var seg = Data([UInt8(ascii: "S")])
    seg.appendLE(frame)
    seg.appendLE(ms.bitPattern)
    seg.appendLE(UInt16(w))
    seg.appendLE(UInt16(h))
    seg.appendLE(UInt16(1))
    seg.append(lead == nil ? 255 : 0)  // the one query's class, Car; dropped without a lead
    seg.append(lead == nil ? 0 : 255)  // its score
    var runs = Data()
    var count: UInt32 = 0
    var i = 0
    while i < qmap.count {
      var j = i + 1
      while j < qmap.count && j - i < 256 && qmap[j] == qmap[i] { j += 1 }
      runs.append(qmap[i])
      runs.append(UInt8(j - i - 1))
      count += 1
      i = j
    }
    seg.appendLE(count)
    seg.append(runs)

    var l = Data([UInt8(ascii: "L")])
    l.appendLE(frame)
    l.append(0)  // narrow
    l.append(line == nil ? 0 : 1)
    if let line { for v in [line.score, line.x1, line.y1, line.x2, line.y2] { l.appendLE(v.bitPattern) } }
    return [seg, l]
  }
}
