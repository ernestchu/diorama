#if canImport(CoreML)
  import Compression
  import CoreML
  import CryptoKit
  import Foundation

  /// The stop-line model, from its Hugging Face repo (ernestchu/diorama), not the app's bundle: the repo's
  /// manifest.json names a version and each file's SHA-256; a newer version's StoplinePilot.mlpackage.zip is
  /// downloaded, checked, unzipped and compiled for this device, and kept in Application Support as
  /// `v<N>/StoplinePilot.mlmodelc` beside the manifest it came with. The model it replaces stays until the next check,
  /// so whatever has it loaded keeps working, and any failure leaves the last good model in place.
  public struct StopLineStore: Sendable {
    public static let source = URL(string: "https://huggingface.co/ernestchu/diorama/resolve/main/")!
    public static let archive = "StoplinePilot.mlpackage.zip"

    public struct Manifest: Codable, Sendable {
      public struct File: Codable, Sendable {
        public let sha256: String
        public let bytes: Int
      }
      public let version: Int
      public let files: [String: File]
    }

    public enum Failure: LocalizedError {
      case noArchive, wrongSize(Int, Int), wrongHash, noPackage, badZip(String), noSpace(Int)
      public var errorDescription: String? {
        switch self {
        case .noArchive: "the manifest names no \(StopLineStore.archive)"
        case .wrongSize(let got, let want): "the download is \(got) bytes, not \(want)"
        case .wrongHash: "the download's SHA-256 is not the manifest's"
        case .noPackage: "the archive holds no .mlpackage"
        case .badZip(let why): "the archive is broken: \(why)"
        case .noSpace(let mb): "the update needs about \(mb) MB free"
        }
      }
    }

    /// Where the manifest and the archive are: the repo's main branch, or a folder in tests.
    public let source: URL
    /// Where the compiled models and the installed manifest are kept.
    public let directory: URL

    public init(source: URL = StopLineStore.source, directory: URL? = nil) {
      self.source = source
      self.directory =
        directory ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appending(path: "StopLineModel")
    }

    /// The app's own store.
    public static let shared = StopLineStore()

    /// The installed model: its version and compiled model, or nil before the first download.
    public func installed() -> (version: Int, url: URL)? {
      guard let data = try? Data(contentsOf: directory.appending(path: "manifest.json")),
        let manifest = try? JSONDecoder().decode(Manifest.self, from: data)
      else { return nil }
      let url = directory.appending(path: "v\(manifest.version)/\(StopLineModel.name).mlmodelc")
      return FileManager.default.fileExists(atPath: url.path) ? (manifest.version, url) : nil
    }

    /// Fetches the manifest and installs its model if it is newer than the installed one. Returns the version
    /// installed, or nil when the installed one is current. Not to be run twice at once.
    public func update(session: URLSession = .shared) async throws -> Int? {
      let fm = FileManager.default
      try fm.createDirectory(at: directory, withIntermediateDirectories: true)
      var excluded = URLResourceValues()
      excluded.isExcludedFromBackup = true  // downloaded again whenever it is missing
      var dir = directory
      try? dir.setResourceValues(excluded)
      let current = installed()
      prune(keeping: current?.version)

      let manifestData = try await fetch(source.appending(path: "manifest.json"), session)
      let manifest = try JSONDecoder().decode(Manifest.self, from: manifestData)
      // any version is newer than none, version 0 too
      if let current, manifest.version <= current.version { return nil }
      guard let file = manifest.files[Self.archive] else { throw Failure.noArchive }

      // the archive, then the package unzipped from it, then the compiled model: about three times its size at most
      let needed = Int64(file.bytes) * 3
      if let free = try? directory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        .volumeAvailableCapacityForImportantUsage, free < needed
      {
        throw Failure.noSpace(Int(needed >> 20))
      }

      let work = directory.appending(path: ".incoming-\(UUID().uuidString)")
      try fm.createDirectory(at: work, withIntermediateDirectories: true)
      defer { try? fm.removeItem(at: work) }
      let zip = work.appending(path: Self.archive)
      try await download(source.appending(path: Self.archive), to: zip, session)
      let size = (try fm.attributesOfItem(atPath: zip.path)[.size] as? Int) ?? -1
      guard size == file.bytes else { throw Failure.wrongSize(size, file.bytes) }
      guard try Self.sha256(of: zip) == file.sha256.lowercased() else { throw Failure.wrongHash }

      let unzipped = work.appending(path: "unzipped")
      try Zip.extract(zip, to: unzipped)
      try? fm.removeItem(at: zip)
      guard let package = fm.enumerator(at: unzipped, includingPropertiesForKeys: nil)?
        .compactMap({ $0 as? URL }).first(where: { $0.pathExtension == "mlpackage" })
      else { throw Failure.noPackage }
      let compiled = try await MLModel.compileModel(at: package)
      defer { try? fm.removeItem(at: compiled) }
      try? fm.removeItem(at: unzipped)
      _ = try MLModel(contentsOf: compiled)  // it loads here, or it is not installed

      let target = directory.appending(path: "v\(manifest.version)")
      try? fm.removeItem(at: target)
      try fm.createDirectory(at: target, withIntermediateDirectories: true)
      try fm.moveItem(at: compiled, to: target.appending(path: "\(StopLineModel.name).mlmodelc"))
      try manifestData.write(to: directory.appending(path: "manifest.json"), options: .atomic)
      return manifest.version
    }

    /// Removes every model but the installed one, and anything a broken update left.
    private func prune(keeping version: Int?) {
      let fm = FileManager.default
      for name in (try? fm.contentsOfDirectory(atPath: directory.path)) ?? [] where name != "manifest.json" && name != "v\(version ?? -1)" {
        try? fm.removeItem(at: directory.appending(path: name))
      }
    }

    private func request(_ url: URL) -> URLRequest {
      URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 30)
    }

    private func fetch(_ url: URL, _ session: URLSession) async throws -> Data {
      let (data, response) = try await session.data(for: request(url))
      try Self.check(response, url)
      return data
    }

    private func download(_ url: URL, to file: URL, _ session: URLSession) async throws {
      let (tmp, response) = try await session.download(for: request(url))
      try Self.check(response, url)
      try FileManager.default.moveItem(at: tmp, to: file)
    }

    private static func check(_ response: URLResponse, _ url: URL) throws {
      if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
        throw URLError(.badServerResponse, userInfo: [NSLocalizedDescriptionKey: "\(url.lastPathComponent): HTTP \(http.statusCode)"])
      }
    }

    static func sha256(of file: URL) throws -> String {
      let handle = try FileHandle(forReadingFrom: file)
      defer { try? handle.close() }
      var hash = SHA256()
      while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty { hash.update(data: chunk) }
      return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
  }

  /// Just enough of ZIP to unpack the model's archive: stored and deflated entries, read through the central
  /// directory (so entries with data descriptors work), no ZIP64, no encryption. iOS has no unzip of its own.
  enum Zip {
    static func extract(_ zip: URL, to dir: URL) throws {
      let data = try Data(contentsOf: zip, options: .alwaysMapped)
      try data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
        func u16(_ o: Int) throws -> Int {
          guard o >= 0 && o + 2 <= raw.count else { throw StopLineStore.Failure.badZip("past its end") }
          return Int(raw[o]) | Int(raw[o + 1]) << 8
        }
        func u32(_ o: Int) throws -> Int { try u16(o) | (u16(o + 2) << 16) }

        // the end of central directory record: within the last 64 KiB and 22 bytes, its comment after it
        var end = -1
        var o = raw.count - 22
        while o >= max(0, raw.count - 22 - 65535) {
          if try u32(o) == 0x0605_4b50 {
            end = o
            break
          }
          o -= 1
        }
        guard end >= 0 else { throw StopLineStore.Failure.badZip("no central directory") }
        let count = try u16(end + 10)
        var entry = try u32(end + 16)
        let fm = FileManager.default
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        for _ in 0..<count {
          guard try u32(entry) == 0x0201_4b50 else { throw StopLineStore.Failure.badZip("a broken central directory") }
          let method = try u16(entry + 10), packed = try u32(entry + 20), size = try u32(entry + 24)
          let nameLength = try u16(entry + 28), extra = try u16(entry + 30), comment = try u16(entry + 32)
          let local = try u32(entry + 42)
          guard entry + 46 + nameLength <= raw.count else { throw StopLineStore.Failure.badZip("past its end") }
          let name = String(decoding: raw[(entry + 46)..<(entry + 46 + nameLength)], as: UTF8.self)
          entry += 46 + nameLength + extra + comment

          let parts = name.split(separator: "/", omittingEmptySubsequences: true)
          guard !name.hasPrefix("/"), !parts.contains(".."), !parts.isEmpty else { throw StopLineStore.Failure.badZip("an unsafe name, \(name)") }
          let url = parts.reduce(dir) { $0.appending(path: String($1)) }
          if name.hasSuffix("/") {
            try fm.createDirectory(at: url, withIntermediateDirectories: true)
            continue
          }
          guard try u32(local) == 0x0403_4b50 else { throw StopLineStore.Failure.badZip("a broken entry, \(name)") }
          let start = try local + 30 + u16(local + 26) + u16(local + 28)
          guard start + packed <= raw.count else { throw StopLineStore.Failure.badZip("\(name) runs past its end") }
          let src = raw.baseAddress!.advanced(by: start).assumingMemoryBound(to: UInt8.self)
          var out = Data(count: size)
          switch method {
          case 0:
            guard packed == size else { throw StopLineStore.Failure.badZip("\(name)'s sizes differ") }
            out.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(start: src, count: size)) }
          case 8:
            // COMPRESSION_ZLIB is raw deflate (RFC 1951), as ZIP stores it
            let n =
              size == 0
              ? 0
              : out.withUnsafeMutableBytes {
                compression_decode_buffer($0.baseAddress!.assumingMemoryBound(to: UInt8.self), size, src, packed, nil, COMPRESSION_ZLIB)
              }
            guard n == size else { throw StopLineStore.Failure.badZip("\(name) inflates to \(n) bytes, not \(size)") }
          default:
            throw StopLineStore.Failure.badZip("\(name) is packed with method \(method)")
          }
          try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
          try out.write(to: url)
        }
      }
    }
  }
#endif
