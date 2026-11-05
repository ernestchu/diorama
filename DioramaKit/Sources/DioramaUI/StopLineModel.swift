#if canImport(CoreML)
  import CoreML
  import CoreVideo
  import DioramaLink
  import Foundation

  /// The stop-line model (StoplinePilot, Core ML): the packed narrow frame in, its stop line and lead box out.
  /// Its inputs are raw pixel values; the graph normalises them itself.
  public final class StopLineModel: @unchecked Sendable {
    public static let name = "StoplinePilot"
    private let model: MLModel
    private let uv: MLMultiArray
    private var luma: CVPixelBuffer

    /// The compiled model in the app's bundle, or nil when the app was built without it.
    public static func bundled() -> StopLineModel? {
      guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else { return nil }
      return try? StopLineModel(url: url)
    }

    public init(url: URL) throws {
      let config = MLModelConfiguration()
      config.computeUnits = .all
      model = try MLModel(contentsOf: url, configuration: config)
      uv = try MLMultiArray(shape: [1, 2, 128, 256], dataType: .float16)
      var pb: CVPixelBuffer?
      CVPixelBufferCreate(kCFAllocatorDefault, 512, 256, kCVPixelFormatType_OneComponent8, nil, &pb)
      guard let pb else { throw CocoaError(.featureUnsupported) }
      luma = pb
    }

    /// One frame: the stop line and the lead box, and the model's time in ms. Not reentrant.
    public func run(_ packed: Data) throws -> (line: StopLineDecoder.Line?, lead: StopLineDecoder.Box?, ms: Float) {
      let t = Date()
      let y = StopLineDecoder.luma(packed)
      CVPixelBufferLockBaseAddress(luma, [])
      let base = CVPixelBufferGetBaseAddress(luma)!.assumingMemoryBound(to: UInt8.self)
      let stride = CVPixelBufferGetBytesPerRow(luma)
      y.withUnsafeBufferPointer { src in
        for r in 0..<256 { (base + r * stride).update(from: src.baseAddress! + r * 512, count: 512) }
      }
      CVPixelBufferUnlockBaseAddress(luma, [])
      let plane = 128 * 256
      let dst = uv.dataPointer.assumingMemoryBound(to: Float16.self)
      packed.withUnsafeBytes { (p: UnsafeRawBufferPointer) in
        for i in 0..<(2 * plane) { dst[i] = Float16(p[4 * plane + i]) }
      }
      let input = try MLDictionaryFeatureProvider(dictionary: ["frame": MLFeatureValue(pixelBuffer: luma), "uv": MLFeatureValue(multiArray: uv)])
      let out = try model.prediction(from: input)
      guard let line = out.featureValue(for: "line")?.multiArrayValue, let lead = out.featureValue(for: "lead")?.multiArrayValue else {
        throw CocoaError(.coderValueNotFound)
      }
      let gh = line.shape[2].intValue, gw = line.shape[3].intValue
      let (l, b) = StopLineDecoder.decode(line: Self.floats(line), lead: Self.floats(lead), gh: gh, gw: gw)
      return (l, b, Float(Date().timeIntervalSince(t) * 1000))
    }

    /// A multi-array's values, in its logical order whatever its strides.
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
#endif
