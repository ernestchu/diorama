#if canImport(SwiftUI) && canImport(MetalKit)
  import Foundation
  import Testing

  @testable import DioramaUI

  /// The camera's heading follows the car's without overshoot, settles, and never lags by more than maxLag.
  @Test func theCameraHeadingSettlesWithoutOvershoot() {
    var spring = HeadingSpring()
    #expect(spring.follow(0, at: 0) == 0)
    // the car turns 10 degrees at once; the camera, behind it, closes the gap from one side only
    let turn = 10 * Double.pi / 180
    var lags: [Double] = []
    for i in 1...60 { lags.append(spring.follow(turn, at: Double(i) / 20)) }
    #expect(lags[0] < 0)
    #expect(lags.allSatisfy { $0 <= 1e-12 })
    #expect(zip(lags, lags.dropFirst()).allSatisfy { $1 >= $0 - 1e-12 })
    #expect(abs(lags.last!) < 0.01 * turn)
  }

  @Test func theCameraLagIsHeldAndWrapsAroundHalfATurn() {
    var spring = HeadingSpring()
    _ = spring.follow(.pi - 0.05, at: 0)
    // across the wrap from +pi to -pi: a small turn, not most of a circle
    let lag = spring.follow(-.pi + 0.05, at: 0.05)
    #expect(abs(lag) < 0.11)
    // a sharp turn: the lag is held to maxLag
    var s2 = HeadingSpring()
    _ = s2.follow(0, at: 0)
    #expect(abs(s2.follow(1.5, at: 0.05)) <= s2.maxLag + 1e-12)
    // a jump in the drive's clock starts it over, level with the car
    #expect(s2.follow(1.5, at: 5) == 0)
  }
#endif
