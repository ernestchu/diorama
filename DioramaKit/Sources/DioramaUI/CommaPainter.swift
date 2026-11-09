#if canImport(CoreGraphics)
  import CoreGraphics
  import DioramaCore

  /// Paints CommaUI's polygons into a context whose y runs down the picture, one unit a pixel of it. A gradient is
  /// laid out in many stops so that its blending is the comma's shader's, whatever Core Graphics would do between two.
  enum CommaPainter {
    private static let space = CGColorSpace(name: CGColorSpace.sRGB)!
    private static let steps = 128

    static func draw(_ polygons: [CommaUI.Polygon], height: Double, in cg: CGContext) {
      for polygon in polygons {
        guard let first = polygon.points.first else { continue }
        cg.saveGState()
        defer { cg.restoreGState() }
        cg.beginPath()
        cg.move(to: CGPoint(x: first.x, y: first.y))
        for p in polygon.points.dropFirst() { cg.addLine(to: CGPoint(x: p.x, y: p.y)) }
        cg.closePath()
        switch polygon.fill {
        case .solid(let c):
          cg.setFillColor(CGColor(srgbRed: Double(c.r) / 255, green: Double(c.g) / 255, blue: Double(c.b) / 255, alpha: Double(c.a) / 255))
          cg.fillPath()
        case .gradient(let colors, let stops):
          let at = (0...steps).map { Double($0) / Double(steps) }
          let components = at.flatMap { t -> [CGFloat] in
            let c = CommaUI.color(at: t, colors: colors, stops: stops)
            return [c.x, c.y, c.z, c.w].map { CGFloat($0) }
          }
          guard let gradient = CGGradient(colorSpace: space, colorComponents: components, locations: at.map { CGFloat($0) }, count: at.count) else { continue }
          cg.clip()
          cg.drawLinearGradient(gradient, start: CGPoint(x: 0, y: height), end: .zero, options: [.drawsBeforeStartLocation, .drawsAfterEndLocation])
        }
      }
    }
  }
#endif
