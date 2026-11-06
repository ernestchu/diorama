import Testing

@testable import DioramaLink

/// The crossing-car filter on the lead's box, on diorama's tests/test_side_on.py cases (x1, y1, x2, y2 in the 512x256 frame).
@Test(arguments: [
  ((0, 0, 512, 254), false),  // cut by the frame's bottom: near, whatever its shape
  ((20, 0, 500, 240), false),  // near and filling the frame
  ((150, 100, 350, 200), false),  // wide, but under 8 m
  ((220, 70, 290, 115), false),  // a car's rear far ahead
  ((100, 60, 300, 130), true),  // a car crossing, side-on
  ((300, 90, 512, 139), true),  // one crossing at the frame's edge
  ((200, 60, 330, 100), true),  // one crossing far ahead
] as [((Float, Float, Float, Float), Bool)])
func sideOnLeadIsDropped(_ box: (Float, Float, Float, Float), _ sideOn: Bool) {
  let b = StopLineDecoder.Box(score: 0.9, x1: box.0, y1: box.1, x2: box.2, y2: box.3)
  #expect(StopLineDecoder.sideOn(b) == sideOn)
}

/// decode drops a side-on box: one peak in the lead map, decoded to the crossing car's box above.
@Test func decodeDropsASideOnLead() {
  let gh = 64, gw = 128, n = gh * gw
  let line = [Float](repeating: 0, count: 8 * n)
  func lead(_ x1: Float, _ y1: Float, _ x2: Float, _ y2: Float) -> [Float] {
    var m = [Float](repeating: 0, count: 5 * n)
    let (cx, cy) = ((x1 + x2) / 8, (y1 + y2) / 8)
    let (x, y) = (Int(cx), Int(cy))
    let i = y * gw + x
    m[i] = 0.9
    m[n + i] = cx - Float(x)
    m[2 * n + i] = cy - Float(y)
    m[3 * n + i] = (x2 - x1) / 4 / StopLineDecoder.whScale
    m[4 * n + i] = (y2 - y1) / 4 / StopLineDecoder.whScale
    return m
  }
  #expect(StopLineDecoder.decode(line: line, lead: lead(100, 60, 300, 130), gh: gh, gw: gw).1 == nil)
  let kept = StopLineDecoder.decode(line: line, lead: lead(220, 70, 290, 115), gh: gh, gw: gw).1
  #expect(kept != nil && abs(kept!.x1 - 220) < 1e-3 && abs(kept!.y2 - 115) < 1e-3)
}
