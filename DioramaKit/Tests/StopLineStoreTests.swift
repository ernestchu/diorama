#if os(macOS)
  import CoreML
  import Foundation
  import Testing

  @testable import DioramaUI

  /// A fresh folder for a test, removed after it.
  private func scratch() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appending(path: "diorama-store-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }

  private func zip(_ args: [String], in dir: URL) throws {
    let p = Process()
    p.executableURL = URL(filePath: "/usr/bin/zip")
    p.arguments = ["-q"] + args
    p.currentDirectoryURL = dir
    try p.run()
    p.waitUntilExit()
    #expect(p.terminationStatus == 0)
  }

  /// A source folder as the Hugging Face repo lays it out: the archive and a manifest naming it.
  private func publish(_ archive: URL, version: Int, in source: URL, sha256: String? = nil) throws {
    let fm = FileManager.default
    let target = source.appending(path: StopLineStore.archive)
    try? fm.removeItem(at: target)
    try fm.copyItem(at: archive, to: target)
    let bytes = try fm.attributesOfItem(atPath: target.path)[.size] as! Int
    let manifest: [String: Any] = [
      "version": version, "created": "2026-11-06T00:00:00Z", "checkpoint": "test", "note": "", "metrics": NSNull(),
      "files": [StopLineStore.archive: ["sha256": try sha256 ?? StopLineStore.sha256(of: target), "bytes": bytes]],
    ]
    try JSONSerialization.data(withJSONObject: manifest).write(to: source.appending(path: "manifest.json"))
  }

  /// The ZIP reader against /usr/bin/zip: stored and deflated entries, an empty file, nested folders.
  @Test func zipExtracts() throws {
    let dir = try scratch()
    defer { try? FileManager.default.removeItem(at: dir) }
    let tree = dir.appending(path: "in/Thing.mlpackage/Data/deep")
    try FileManager.default.createDirectory(at: tree, withIntermediateDirectories: true)
    let big = Data((0..<300_000).map { UInt8(($0 * 7 + $0 / 1000) % 251) })
    try big.write(to: tree.appending(path: "weights.bin"))
    try Data("{\"a\": 1}".utf8).write(to: dir.appending(path: "in/Thing.mlpackage/Manifest.json"))
    try Data().write(to: tree.appending(path: "empty"))
    try zip(["-r", "../deflated.zip", "Thing.mlpackage"], in: dir.appending(path: "in"))
    try zip(["-r", "-0", "../stored.zip", "Thing.mlpackage"], in: dir.appending(path: "in"))
    for name in ["deflated", "stored"] {
      let out = dir.appending(path: "out-\(name)")
      try Zip.extract(dir.appending(path: "\(name).zip"), to: out)
      #expect(try Data(contentsOf: out.appending(path: "Thing.mlpackage/Data/deep/weights.bin")) == big, "\(name)")
      #expect(try Data(contentsOf: out.appending(path: "Thing.mlpackage/Manifest.json")) == Data("{\"a\": 1}".utf8))
      #expect(try Data(contentsOf: out.appending(path: "Thing.mlpackage/Data/deep/empty")).isEmpty)
    }
  }

  /// A download whose SHA-256 is not the manifest's is not installed, and nothing is left of it.
  @Test func wrongHashIsRefused() async throws {
    let dir = try scratch()
    defer { try? FileManager.default.removeItem(at: dir) }
    let source = dir.appending(path: "hf"), store = StopLineStore(source: source, directory: dir.appending(path: "store"))
    try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
    try Data("not a model".utf8).write(to: dir.appending(path: "fake.zip"))
    try publish(dir.appending(path: "fake.zip"), version: 1, in: source, sha256: String(repeating: "0", count: 64))
    await #expect(throws: StopLineStore.Failure.self) { try await store.update() }
    #expect(store.installed() == nil)
    #expect(try FileManager.default.contentsOfDirectory(atPath: store.directory.path).isEmpty)
  }

  /// The whole update on the real archive (models/StoplinePilot.mlpackage.zip, as `hf download ernestchu/diorama`
  /// fetches it; skipped without it): installed and runnable; nothing to do while current; a newer version installed
  /// beside the old, which goes at the next check.
  @Test func updateInstallsNewerVersions() async throws {
    let archive = URL(filePath: #filePath).deletingLastPathComponent().appending(path: "../../models/\(StopLineStore.archive)").standardized
    guard FileManager.default.fileExists(atPath: archive.path) else { return }
    let dir = try scratch()
    defer { try? FileManager.default.removeItem(at: dir) }
    let source = dir.appending(path: "hf"), store = StopLineStore(source: source, directory: dir.appending(path: "store"))
    try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)

    // the first version published is 0, and a fresh install takes it
    try publish(archive, version: 0, in: source)
    #expect(try await store.update() == 0)
    #expect(store.installed()?.version == 0)
    try publish(archive, version: 1, in: source)
    #expect(try await store.update() == 1)
    let v1 = try #require(store.installed())
    #expect(v1.version == 1)
    let model = try StopLineModel(url: v1.url)
    _ = try model.run(Data(count: 6 * 128 * 256))
    #expect(try await store.update() == nil)

    try publish(archive, version: 2, in: source)
    #expect(try await store.update() == 2)
    #expect(store.installed()?.version == 2)
    let kept = try FileManager.default.contentsOfDirectory(atPath: store.directory.path).sorted()
    #expect(kept == ["manifest.json", "v1", "v2"])
    #expect(try await store.update() == nil)
    #expect(try FileManager.default.contentsOfDirectory(atPath: store.directory.path).sorted() == ["manifest.json", "v2"])
  }

  /// The real thing, from Hugging Face: only with DIORAMA_HF=1 in the environment, as it downloads ~21 MB.
  @Test(.enabled(if: ProcessInfo.processInfo.environment["DIORAMA_HF"] == "1"))
  func updateFromHuggingFace() async throws {
    let dir = try scratch()
    defer { try? FileManager.default.removeItem(at: dir) }
    let store = StopLineStore(directory: dir)
    let version = try #require(try await store.update())
    let installed = try #require(store.installed())
    #expect(installed.version == version)
    _ = try StopLineModel(url: installed.url).run(Data(count: 6 * 128 * 256))
    #expect(try await store.update() == nil)
    print("Hugging Face: installed version \(version)")
  }
#endif
