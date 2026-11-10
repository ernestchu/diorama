import DioramaCore
import Foundation
import Testing

@testable import DioramaLink

/// The lead car sits at the comma's lead bar, from the driving model alone.
private let fields: [(String, Int)] = [("plan", 990), ("lane_lines", 528), ("lane_lines_prob", 8), ("road_edges", 264), ("lead", 144), ("lead_prob", 3), ("pose", 12)]

private func offset(_ name: String) -> Int { fields.prefix { $0.0 != name }.map(\.1).reduce(0, +) }

/// Raw outputs: the plan bent by `bend`, and the driving model's lead if any.
private func values(bend: Double, lead: (x: Float, y: Float, a: Float)?, second: (x: Float, y: Float, a: Float)? = nil, speed: Double? = nil) -> [Float] {
  var v = [Float](repeating: 0, count: fields.map(\.1).reduce(0, +))
  for i in 0..<33 {
    // along the lane lines' own distances, or as far as `speed` takes the car in the plan's 10 s
    let x = speed.map { $0 * 10 * pow(Double(i) / 32, 2) } ?? 192 * pow(Double(i) / 32, 2)
    v[offset("plan") + i * 15] = Float(x)
    v[offset("plan") + i * 15 + 1] = Float(bend * x * x)
    v[offset("plan") + i * 15 + 3] = 15
  }
  v[offset("pose")] = 15
  v[offset("lead_prob")] = -5
  if let lead {
    v[offset("lead_prob")] = 5
    v[offset("lead")] = lead.x
    v[offset("lead") + 1] = lead.y
    v[offset("lead") + 3] = lead.a
  }
  v[offset("lead_prob") + 1] = -5
  if let second {
    v[offset("lead_prob") + 1] = 5
    v[offset("lead") + 24] = second.x
    v[offset("lead") + 25] = second.y
    v[offset("lead") + 27] = second.a
  }
  return v
}

private func meta() -> DioramaMeta {
  DioramaMeta([
    "frame": ["w": 512.0, "h": 256.0, "f": 910.0, "cx": 256.0, "cy": 47.6], "height": 1.22, "live": true, "fps": 20,
    "drive": fields.map { [$0.0, $0.1] as [Any] }, "seg": StopLineMessages.segMeta,
  ])
}

/// The lead after each frame, the stop-line model seeing a line in every frame.
private func run(_ frames: [(x: Float, y: Float, a: Float)?], bend: Double = 0, speed: Double? = nil) -> [DioramaScene.Lead?] {
  let scene = DioramaScene()
  scene.push(.meta(meta()))
  return frames.enumerated().map { i, lead in
    if i > 0 { for message in StopLineMessages.messages(frame: UInt32(i), ms: 5, line: StopLineDecoder.Line(score: 0.9, x1: 100, y1: 180, x2: 300, y2: 182)) { scene.push(DioramaMessage(message)) } }
    scene.push(.drive(frame: i + 1, seconds: Double(i) * 0.05, ms: 0, values: values(bend: bend, lead: lead, speed: speed)))
    return scene.state?.lead
  }
}

@Test func theCarsRearIsAtTheLeadBar() throws {
  let car = try #require(run([(30, 1.5, 0)]).last!)
  #expect(abs(car.x - 30) < 1e-3 && abs(car.y - 1.5) < 1e-3)
}

@Test func theCarIsSmoothedAsTheBarIs() throws {
  // jumping 8 m: 0.05 s of a 0.2 s lag at 60 Hz moves it about a fifth of the way
  let leads = run([(30, 0, 0), (30, 0, 0), (38, 0, 0)])
  let x = try #require(leads[2]).x
  #expect(x > 30.5 && x < 33.5, "\(x)")

  var bars = CommaUI.LeadBars()
  let m = meta()
  for lead in [(30, 0, 0), (30, 0, 0), (38, 0, 0)] as [(Float, Float, Float)] {
    bars.advance(DioramaOutputs(values: values(bend: 0, lead: lead), fields: m.drive), m, renders: CommaUI.LeadBars.rendersPerFrame)
  }
  #expect(abs(try #require(bars.leads.first).x - x) < 1e-3)
}

@Test func theCarSnapsToAnotherOverThreeMetresAside() throws {
  let car = try #require(run([(30, 0, 0), (30, 0, 0), (25, 4, 0)]).last!)
  #expect(abs(car.x - 25) < 1e-3 && abs(car.y - 4) < 1e-3)
}

@Test func theCarLastsAsLongAsTheBarsFade() throws {
  let on = Array(repeating: (x: Float(30), y: Float(0), a: Float(0)) as (x: Float, y: Float, a: Float)?, count: 40)
  let off = Array(repeating: nil as (x: Float, y: Float, a: Float)?, count: 14)
  let leads = run(on + off)
  #expect(try #require(leads[40]).x == leads[39]!.x, "it stays where it was")
  #expect(leads[46] != nil, "still fading")
  #expect(leads.last! == nil, "gone")
}

@Test func thePlanHasNoSayInWhereTheCarIs() throws {
  let straight = try #require(run([(30, 1, 0), (31, 1, 0)]).last!), bent = try #require(run([(30, 1, 0), (31, 1, 0)], bend: 0.01).last!)
  #expect(straight.x == bent.x && straight.y == bent.y)
}

@Test func theCarFacesAlongThePlan() throws {
  // the plan bends right as y = 0.01 x^2: at 31 m its tangent is atan(0.62); the lane lines are straight
  let car = try #require(run([(30, 0, 0), (31, 0, 0)], bend: 0.01).last!)
  #expect(abs(car.heading - atan(0.62)) < 0.03, "\(car.heading)")
  let straight = try #require(run([(30, 0, 0), (31, 0, 0)]).last!)
  #expect(abs(straight.heading) < 1e-9)
}

@Test func beyondThePlanTheCarFacesAlongTheLane() throws {
  // the plan ends at 15 m (laid out at 1.5 m/s here); the lead is past it
  let car = try #require(run([(60, 0, 0), (60, 0, 0)], bend: 0.01, speed: 1.5).last!)
  #expect(abs(car.heading) < 1e-9, "\(car.heading)")
}

@Test func brakesWhenTheModelSays() throws {
  #expect(try #require(run([(30, 0, -2)]).last!).braking)
  #expect(!(try #require(run([(30, 0, 0)]).last!)).braking)
}

private typealias Car = (x: Float, y: Float, a: Float)

/// Every car after each frame.
private func runCars(_ frames: [(first: Car?, second: Car?)]) -> [[DioramaScene.Lead]] {
  let scene = DioramaScene()
  scene.push(.meta(meta()))
  return frames.enumerated().map { i, f in
    scene.push(.drive(frame: i + 1, seconds: Double(i) * 0.05, ms: 0, values: values(bend: 0, lead: f.first, second: f.second)))
    return scene.state?.leads ?? []
  }
}

@Test func aSecondBarDrawsASecondCar() throws {
  let cars = try #require(runCars([((30, 0.5, 0), (60, -1, -2))]).last)
  try #require(cars.count == 2)
  #expect(abs(cars[0].x - 30) < 1e-3 && abs(cars[0].y - 0.5) < 1e-3 && !cars[0].braking)
  #expect(abs(cars[1].x - 60) < 1e-3 && abs(cars[1].y + 1) < 1e-3 && cars[1].braking)
}

@Test func twoLeadsCloserThanThreeMetresAreOneCar() throws {
  #expect(runCars([((30, 0, 0), (32, 0, 0))]).last!.count == 1)
}

@Test func theSecondCarStandsAloneWhenTheFirstIsGone() throws {
  let cars = try #require(runCars([(nil, (45, 1, 0))]).last)
  try #require(cars.count == 1)
  #expect(abs(cars[0].x - 45) < 1e-3)
}

@Test func theCarsMatchTheBarsOneForOne() throws {
  var bars = CommaUI.LeadBars()
  let m = meta()
  bars.advance(DioramaOutputs(values: values(bend: 0, lead: (30, 0, 0), second: (60, 1, 0)), fields: m.drive), m, renders: CommaUI.LeadBars.rendersPerFrame)
  #expect(bars.leads.count == bars.polygons.count && runCars([((30, 0, 0), (60, 1, 0))]).last!.count == bars.polygons.count)
}
