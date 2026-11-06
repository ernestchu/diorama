// swift-tools-version: 6.2
//
// Diorama for iPhone: the comma's drive drawn as a live miniature. The comma
// runs the driving model and streams its narrow camera frame and the model's
// outputs over the link; the phone runs only the stop-line model.
import PackageDescription

let package = Package(
  name: "DioramaKit",
  platforms: [.macOS(.v15), .iOS(.v26)],
  products: [
    .library(name: "DioramaCore", targets: ["DioramaCore"]),
    .library(name: "DioramaLink", targets: ["DioramaLink"]),
    .library(name: "DioramaUI", targets: ["DioramaUI"]),
  ],
  targets: [
    // The scene: the drive's messages in, the road, the plan, the lead and the stop line out.
    // Optimised in a debug build too: the road map steps cell by cell at 20 Hz.
    .target(name: "DioramaCore", resources: [.process("Resources")], swiftSettings: [.unsafeFlags(["-O"], .when(configuration: .debug))]),
    // The comma's link: protocol 3's framing, and the frames and outputs the comma streams.
    .target(name: "DioramaLink", dependencies: ["DioramaCore"]),
    // The scene in Metal, the stop-line model, and the feed from the comma.
    .target(
      name: "DioramaUI", dependencies: ["DioramaCore", "DioramaLink"],
      swiftSettings: [.unsafeFlags(["-O"], .when(configuration: .debug))]),
    .testTarget(name: "DioramaKitTests", dependencies: ["DioramaCore", "DioramaLink", "DioramaUI"], path: "Tests"),
  ]
)
