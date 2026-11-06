#if canImport(Metal) && canImport(ImageIO)
  import CoreGraphics
  import Foundation
  import ImageIO
  import Metal
  import Testing
  import UniformTypeIdentifiers

  @testable import DioramaCore
  @testable import DioramaUI

  /// The app's Metal drawing against the page's WebGL drawing of the same frames: the page's stills (web/tools/still.js
  /// in headless Chrome, at 390x844 and device pixel ratio 1, the dark theme, each view held, the page's own cameras)
  /// beside DioramaRenderer's of DioramaScene at the same frame, the same size, theme, view and cameras. Antialiasing
  /// and the two GPUs' rasterisers differ, so the pictures are compared by the masks of their key colours (the planned
  /// path's blue, the paint's white, the road edges' red): IoU at least 0.9 with a pixel's slack for antialiasing and
  /// centroids within 2 px, or for a mask of only a few pixels its size; and as a whole by the mean absolute difference. A failing frame leaves the app's picture and a difference image beside the page's.
  ///
  /// The drive is synthetic (web/tools/synthetic_drive.py) and its stills are the page's of it (still.js, headless
  /// Chrome), both committed beside the test; nothing comes from a real drive. Remake the stills after changing the
  /// page's drawing: web/README.md says how.
  @Test func drawingMatchesThePage() throws {
    let root = URL(filePath: #filePath).deletingLastPathComponent().appending(path: "../..").standardized
    let dir = root.appending(path: "DioramaKit/Tests/Fixtures/synthetic")
    let stills = dir.appending(path: "stills")
    let drive = try Data(contentsOf: dir.appending(path: "drive.diorama"))
    let pictures = try FileManager.default.contentsOfDirectory(atPath: stills.path).filter { $0.hasSuffix(".png") }.sorted()
    #expect(pictures.count == 10)
    let device = try #require(MTLCreateSystemDefaultDevice())
    let messages = DioramaMessage.file(drive)
    // a failing frame's pictures go where git does not look
    let failed = root.appending(path: "DioramaKit/.build/pixel-parity-failed")
    try? FileManager.default.removeItem(at: failed)

    for name in pictures {
      let parts = name.dropLast(4).split(separator: "-")
      let frame = try #require(Int(parts[0])), bird = parts[1] == "bird"
      let page = try readPNG(stills.appending(path: name))

      // the scene stepped to the frame, as the page's still mode does: every message up to the frame's drive message
      let scene = DioramaScene()
      for data in messages {
        let msg = DioramaMessage(data)
        if case .drive(let k, _, _, _) = msg, k > frame { break }
        scene.push(msg)
      }
      #expect(scene.state?.frame == frame, "\(name): the scene has no frame \(frame)")

      let renderer = try #require(DioramaRenderer(device: device))
      renderer.theme = .dark
      renderer.fixedView = bird ? 1 : 0
      let app = try draw(renderer, renderer.take(scene), page.width, page.height, device)

      var report: [String] = []
      var maskFailed = false
      for (mask, test) in masks {
        let a = page.mask(test), b = app.mask(test)
        if a.count == 0 && b.count == 0 { continue }
        if max(a.count, b.count) < smallMask {
          // a few pixels of a thin or soft line: what antialiasing decides; only their number is compared
          let close = abs(a.count - b.count) <= max(4, max(a.count, b.count) / 3)
          report.append("\(mask) \(a.count) vs \(b.count) px")
          if !close { Issue.record("\(name): \(mask) \(a.count) px on the page, \(b.count) in the app"); maskFailed = true }
          continue
        }
        // IoU with a pixel's slack: each mask's pixels within 1 px of the other's, the smaller share of the two
        let iou = min(within1(a, b, page.width, page.height), within1(b, a, page.width, page.height))
        let (ca, cb) = (centroid(a, page.width), centroid(b, page.width))
        let shift = ((ca.x - cb.x) * (ca.x - cb.x) + (ca.y - cb.y) * (ca.y - cb.y)).squareRoot()
        report.append("\(mask) IoU±1 \(String(format: "%.3f", iou)) shift \(String(format: "%.1f", shift)) px (\(a.count) vs \(b.count))")
        if iou < 0.9 || (shift > 2 && !centroidless.contains(mask)) {
          Issue.record("\(name): \(mask) IoU±1 \(iou), centroids \(shift) px apart (page \(a.count) px, app \(b.count) px)")
          maskFailed = true
        }
      }
      let mad = page.meanAbsoluteDifference(app)
      report.append("mean |diff| \(String(format: "%.2f", mad))")
      if mad > madLimit { Issue.record("\(name): mean absolute difference \(mad) over \(madLimit)") }
      print("pixel parity \(name): \(report.joined(separator: ", "))")

      if mad > madLimit || maskFailed {
        try? FileManager.default.createDirectory(at: failed, withIntermediateDirectories: true)
        try app.writePNG(failed.appending(path: "\(frame)-\(parts[1])-app.png"))
        try page.difference(app).writePNG(failed.appending(path: "\(frame)-\(parts[1])-diff.png"))
      }
    }
  }

  /// The whole picture's mean absolute difference (0-255, over RGB) above which the drawing has drifted. Calibrated on
  /// the first run: 0.01 to 0.09 over eight frames, both views.
  private let madLimit = 0.5
  /// Masks smaller than this many pixels are thin or soft lines' antialiased edges; IoU means nothing for them.
  private let smallMask = 40

  /// The key colours, in the dark theme, as RGB tests. Paint white takes in the cars' specular highlights as well as the
  /// lines, scattered pixels that move its centroid without meaning anything, so it is held to its IoU alone.
  private let centroidless: Set<String> = ["paint white"]
  private let masks: [(String, @Sendable (UInt8, UInt8, UInt8) -> Bool)] = [
    ("path blue", { r, g, b in Int(b) > 150 && Int(b) - Int(r) > 80 && Int(b) - Int(g) > 30 }),
    ("paint white", { r, g, b in min(r, g, b) > 200 && Int(max(r, g, b)) - Int(min(r, g, b)) < 25 }),
    ("edge red", { r, g, b in Int(r) > 110 && Int(r) - Int(g) > 50 && Int(r) - Int(b) > 50 }),
  ]

  struct Picture {
    let width: Int, height: Int
    var rgba: [UInt8]

    func mask(_ test: (UInt8, UInt8, UInt8) -> Bool) -> Set<Int> {
      var out = Set<Int>()
      for i in 0..<(width * height) where test(rgba[4 * i], rgba[4 * i + 1], rgba[4 * i + 2]) { out.insert(i) }
      return out
    }

    func meanAbsoluteDifference(_ o: Picture) -> Double {
      var sum = 0
      for i in 0..<(width * height) {
        for c in 0..<3 { sum += abs(Int(rgba[4 * i + c]) - Int(o.rgba[4 * i + c])) }
      }
      return Double(sum) / Double(3 * width * height)
    }

    /// Four times the absolute difference, so a small drift shows.
    func difference(_ o: Picture) -> Picture {
      var d = self
      for i in 0..<(width * height) {
        for c in 0..<3 { d.rgba[4 * i + c] = UInt8(min(255, 4 * abs(Int(rgba[4 * i + c]) - Int(o.rgba[4 * i + c])))) }
        d.rgba[4 * i + 3] = 255
      }
      return d
    }

    func writePNG(_ url: URL) throws {
      let data = Data(rgba) as CFData
      let image = try #require(
        CGImage(
          width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 4 * width, space: CGColorSpace(name: CGColorSpace.sRGB)!,
          bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue), provider: CGDataProvider(data: data)!, decode: nil,
          shouldInterpolate: false, intent: .defaultIntent))
      let dest = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
      CGImageDestinationAddImage(dest, image, nil)
      #expect(CGImageDestinationFinalize(dest))
    }
  }

  /// The share of `a`'s pixels that lie within one pixel (8-neighbourhood) of one of `b`'s.
  private func within1(_ a: Set<Int>, _ b: Set<Int>, _ w: Int, _ h: Int) -> Double {
    guard !a.isEmpty else { return 1 }
    var hit = 0
    for i in a {
      let (x, y) = (i % w, i / w)
      search: for dy in -1...1 {
        for dx in -1...1 {
          let (xx, yy) = (x + dx, y + dy)
          if xx >= 0 && xx < w && yy >= 0 && yy < h && b.contains(yy * w + xx) {
            hit += 1
            break search
          }
        }
      }
    }
    return Double(hit) / Double(a.count)
  }

  private func centroid(_ m: Set<Int>, _ width: Int) -> (x: Double, y: Double) {
    guard !m.isEmpty else { return (0, 0) }
    var x = 0.0, y = 0.0
    for i in m {
      x += Double(i % width)
      y += Double(i / width)
    }
    return (x / Double(m.count), y / Double(m.count))
  }

  /// A PNG as sRGB RGBA bytes, as the browser wrote it.
  private func readPNG(_ url: URL) throws -> Picture {
    let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
    let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
    let (w, h) = (image.width, image.height)
    var rgba = [UInt8](repeating: 0, count: 4 * w * h)
    let ctx = try #require(
      CGContext(
        data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 4 * w, space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    return Picture(width: w, height: h, rgba: rgba)
  }

  /// One frame drawn offscreen as the app's view draws it (4x MSAA resolved into an 8-bit BGRA texture), read back as RGBA.
  private func draw(_ renderer: DioramaRenderer, _ frame: DioramaRenderer.Frame, _ w: Int, _ h: Int, _ device: MTLDevice) throws -> Picture {
    func texture(_ format: MTLPixelFormat, samples: Int, storage: MTLStorageMode) throws -> MTLTexture {
      let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: w, height: h, mipmapped: false)
      d.textureType = samples > 1 ? .type2DMultisample : .type2D
      d.sampleCount = samples
      d.usage = .renderTarget
      d.storageMode = storage
      return try #require(device.makeTexture(descriptor: d))
    }
    let msaa = try texture(DioramaRenderer.colorFormat, samples: DioramaRenderer.sampleCount, storage: .private)
    let resolved = try texture(DioramaRenderer.colorFormat, samples: 1, storage: .shared)
    let depth = try texture(DioramaRenderer.depthFormat, samples: DioramaRenderer.sampleCount, storage: .private)
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = msaa
    pass.colorAttachments[0].resolveTexture = resolved
    pass.colorAttachments[0].storeAction = .multisampleResolve
    pass.depthAttachment.texture = depth
    pass.depthAttachment.storeAction = .dontCare
    var buffer: MTLCommandBuffer?
    renderer.draw(frame, to: pass, size: CGSize(width: w, height: h)) { buffer = $0 }
    buffer?.waitUntilCompleted()
    var bgra = [UInt8](repeating: 0, count: 4 * w * h)
    resolved.getBytes(&bgra, bytesPerRow: 4 * w, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
    for i in 0..<(w * h) { bgra.swapAt(4 * i, 4 * i + 2) }  // BGRA -> RGBA
    return Picture(width: w, height: h, rgba: bgra)
  }
#endif
