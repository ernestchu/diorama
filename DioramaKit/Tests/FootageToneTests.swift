import Foundation
import Testing

@testable import DioramaUI

/// Readings, a second each, through FootageTone: the answer after each, "-" for none yet, "D" dark, "L" light.
private func run(_ readings: [Double]) -> String {
  var tone = FootageTone()
  return readings.map { r in
    _ = tone.read(r)
    return tone.dark.map { $0 ? "D" : "L" } ?? "-"
  }.joined()
}

/// How many times the answer changes, the first answer included.
private func changes(_ s: String) -> Int { zip(s, s.dropFirst()).filter { $0 != $1 }.count }

private func secs(_ n: Int, _ v: Double) -> [Double] { Array(repeating: v, count: n) }

/// A frame whose top third is `sky` and whose rest is `road`, packed as the comma sends it.
private func frame(sky: UInt8, road: UInt8) -> Data {
  var d = Data(repeating: 128, count: 6 * 256 * 128)
  for r in 0..<128 { for c in 0..<256 { d[r * 256 + c] = r < 43 ? sky : road } }
  return d
}

@Test func footageToneAnswersAfterTenSeconds() {
  #expect(run(secs(10, 0.12)) == "---------D")
  #expect(run(secs(10, 0.30)) == "---------L")
  // a dark start, as leaving a garage, is outweighed by the daylight after it
  #expect(run(secs(3, 0.1) + secs(7, 0.3)) == "---------L")
}

@Test func footageToneRidesOutHalfAMinuteOfDark() {
  // a day drive past a wall, or through a short underpass, for thirty seconds
  let s = run(secs(60, 0.3) + secs(30, 0.1) + secs(60, 0.3))
  #expect(!s.contains("D") && s.hasSuffix("L"))
}

@Test func footageToneTurnsDarkWhenTheDarkLasts() {
  let s = run(secs(60, 0.3) + secs(120, 0.12))
  #expect(changes(s) == 2 && s.hasSuffix("D"))
}

@Test func footageToneDoesNotFlickAtSunset() {
  // the sky in and out of view as the light fades: half a minute either side of both thresholds, for twenty minutes
  let swings = (0..<20).flatMap { _ in secs(30, 0.14) + secs(30, 0.23) }
  #expect(changes(run(swings)) == 1)
}

@Test func footageToneLeavesTheAnswerBetweenTheThresholds() {
  // facing a fence in a car park by day: neither
  #expect(!run(secs(60, 0.18)).contains(where: { $0 != "-" }))
}

@Test func footageToneReadsTheSkyNotTheRoad() {
  // night: a black sky over a road the headlights light up
  var night = FootageTone()
  for _ in 0..<(10 * 20) { _ = night.see(frame(sky: 25, road: 120)) }
  #expect(night.dark == true)
  // day: a bright sky over a dark road
  var day = FootageTone()
  for _ in 0..<(10 * 20) { _ = day.see(frame(sky: 140, road: 40)) }
  #expect(day.dark == false)
}
