#if canImport(CoreML)
  import CoreML
  import CoreVideo
  import DioramaLink
  import Foundation

  /// The stop-line model (StoplinePilot, Core ML): the packed narrow frame in, its stop line out.
  /// Its inputs are raw pixel values; the graph normalises them itself.
  public final class StopLineModel: @unchecked Sendable {
    public static let name = "StoplinePilot"
    private let model: MLModel
    private let uv: MLMultiArray
    private var luma: CVPixelBuffer

    /// The model downloaded from Hugging Face (StopLineStore), or nil before the first download.
    public static func installed() -> StopLineModel? {
      guard let url = StopLineStore.shared.installed()?.url else { return nil }
      return try? StopLineModel(url: url)
    }

    /// Where Core ML runs the model on this device: each processor's share of the work, from Core ML's own plan
    /// (MLComputePlan) for the configuration the app runs it with. Constants and reshapes that run nowhere are left out.
    public struct Placement: Sendable, Equatable {
      public struct Share: Sendable, Equatable, Identifiable {
        public let device: String
        /// This device's part of the model, 0 to 1: of Core ML's estimated cost, or of the operations where Core ML
        /// gives no estimates (the iOS Simulator).
        public let cost: Double
        public let operations: Int
        public var id: String { device }
      }
      /// Most work first.
      public let shares: [Share]
      /// The shares are of the operations, not of the estimated cost: Core ML gave no estimates.
      public var byOperations = false

      public var text: String {
        shares.map { "\($0.device) \(Int(($0.cost * 100).rounded()))% (\($0.operations) operations)" }.joined(separator: ", ")
          + (byOperations ? ", by operations" : "")
      }
    }

    public static func placement(url: URL? = StopLineStore.shared.installed()?.url) async -> Placement? {
      guard let url else { return nil }
      let config = MLModelConfiguration()
      config.computeUnits = .all
      guard let plan = try? await MLComputePlan.load(contentsOf: url, configuration: config),
        case .program(let program) = plan.modelStructure, let main = program.functions["main"]
      else { return nil }
      var cost: [String: Double] = [:]
      var ops: [String: Int] = [:]
      for op in main.block.operations {
        let device: String
        switch plan.deviceUsage(for: op)?.preferred {
        case .neuralEngine: device = "Neural Engine"
        case .gpu: device = "GPU"
        case .cpu: device = "CPU"
        default: continue  // a constant or a reshape: no device runs it
        }
        ops[device, default: 0] += 1
        cost[device, default: 0] += plan.estimatedCost(of: op)?.weight ?? 0
      }
      let total = cost.values.reduce(0, +), count = Double(ops.values.reduce(0, +))
      let byOperations = total <= 0
      let shares = ops.keys.map { d in
        Placement.Share(device: d, cost: byOperations ? Double(ops[d]!) / count : (cost[d] ?? 0) / total, operations: ops[d]!)
      }
      .sorted { ($0.cost, $0.operations) > ($1.cost, $1.operations) }
      return shares.isEmpty ? nil : Placement(shares: shares, byOperations: byOperations)
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

    /// One frame: the stop line, and the model's time in ms. Not reentrant.
    public func run(_ packed: Data) throws -> (line: StopLineDecoder.Line?, ms: Float) {
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
      // only `line`: a `lead` output, where a model still has one, is left alone
      guard let line = out.featureValue(for: "line")?.multiArrayValue else { throw CocoaError(.coderValueNotFound) }
      let gh = line.shape[2].intValue, gw = line.shape[3].intValue
      return (StopLineDecoder.decode(line: Self.floats(line), gh: gh, gw: gw), Float(Date().timeIntervalSince(t) * 1000))
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
