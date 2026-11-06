import DioramaLink
import DioramaUI
import SwiftUI
import os

@main
struct DioramaApp: App {
  @State private var benchmark = StopLineBenchmark()

  var body: some Scene {
    WindowGroup { RootView().environment(benchmark) }
  }
}

/// Where the comma is: at the far end of the USB cable, or a Mac playing it (sim/commasim).
enum CommaAddress {
  static let key = "comma.host"
  #if targetEnvironment(simulator)
    /// The iOS Simulator reaches the Mac it runs on at localhost.
    static let standard = "127.0.0.1"
  #else
    static let standard = Wire.commaHost
  #endif
}

/// The screen stays on while the app is in front and has something to do: a comma streaming the drive (with Keep
/// Screen On, the default), or a benchmark running. A locked iPhone suspends the app, which drops the comma's link
/// and cuts a benchmark short. As jetlink's iPhone app does for its server and its Diorama tab.
enum KeepAwake {
  static let key = "keepScreenOn"

  @MainActor
  static func update(active: Bool, streaming: Bool, benchmarking: Bool) {
    let keep = UserDefaults.standard.object(forKey: key) as? Bool ?? true
    let on = active && (benchmarking || (keep && streaming))
    if UIApplication.shared.isIdleTimerDisabled != on {
      Logger(subsystem: "diorama", category: "app").info("keep the screen on: \(on ? "yes" : "no", privacy: .public)")
    }
    UIApplication.shared.isIdleTimerDisabled = on
  }
}

struct RootView: View {
  @AppStorage(CommaAddress.key) private var host = CommaAddress.standard
  @AppStorage(KeepAwake.key) private var keepScreenOn = true
  @State private var feed: CommaFeed?
  @State private var settings = false
  @Environment(\.scenePhase) private var phase
  @Environment(StopLineBenchmark.self) private var benchmark
  private let models = Models.shared

  var body: some View {
    NavigationStack {
      Group {
        // the live view stops while a benchmark runs, so the two never share the model's hardware
        if let feed, phase != .background, !benchmark.running {
          DioramaSceneView(feed: feed)
        } else {
          Color.clear
        }
      }
      .ignoresSafeArea(edges: .horizontal)
      .navigationTitle("Diorama")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        Button("Settings", systemImage: "gearshape") { settings = true }
      }
      .sheet(isPresented: $settings) { SettingsView(host: $host) }
    }
    .onAppear {
      connect()
      models.start()
    }
    .onChange(of: host) { connect() }
    .onChange(of: phase) {
      keepAwake()
      if phase == .active { Task { await models.refresh() } }
    }
    // a new model is taken up by the next connection; the first one at once, as there was none
    .onChange(of: models.version) { if feed?.streaming != true || feed?.model == nil { connect() } }
    .onChange(of: feed?.streaming) { keepAwake() }
    .onChange(of: benchmark.running) { keepAwake() }
    .onChange(of: keepScreenOn) { keepAwake() }
  }

  private func keepAwake() {
    KeepAwake.update(active: phase == .active, streaming: feed?.streaming ?? false, benchmarking: benchmark.running)
  }

  private func connect() {
    feed?.stop()
    feed = CommaFeed(host: host, name: UIDevice.current.name, model: models.stopLine)
  }
}

/// The stop-line model: the one downloaded last, and a check of Hugging Face for a newer one when the app starts and
/// each time it comes to the front. Without a network and without a model yet, the drive is drawn with no stop lines.
@MainActor
@Observable
final class Models {
  static let shared = Models()
  private(set) var stopLine: StopLineModel?
  private(set) var version: Int?
  private(set) var status = "loading"
  private var checking = false
  private var started = false

  /// Loads the installed model (compiling for the Neural Engine takes a moment), then checks for a newer one.
  func start() {
    guard !started else { return }
    started = true
    Task {
      let loaded = await Task.detached(priority: .userInitiated) { () -> (Int, StopLineModel)? in
        guard let (v, url) = StopLineStore.shared.installed(), let m = try? StopLineModel(url: url) else { return nil }
        return (v, m)
      }.value
      if let (v, m) = loaded { (stopLine, version) = (m, v) }
      await refresh()
    }
  }

  func refresh() async {
    guard started, !checking else { return }
    checking = true
    defer { checking = false }
    status = "checking for a newer model"
    let log = Logger(subsystem: "diorama", category: "model")
    do {
      let installed = try await Task.detached(priority: .utility) { try await StopLineStore.shared.update() }.value
      if let installed {
        status = "loading version \(installed)"
        let m = await Task.detached(priority: .userInitiated) { StopLineStore.shared.installed().flatMap { try? StopLineModel(url: $0.url) } }.value
        if let m { (stopLine, version) = (m, installed) }
        log.info("stop-line model version \(installed) installed")
      }
      status = "up to date"
    } catch {
      status = "could not check: \(error.localizedDescription)"
      log.error("stop-line model check: \(error.localizedDescription, privacy: .public)")
    }
  }
}

struct SettingsView: View {
  @Binding var host: String
  @AppStorage(KeepAwake.key) private var keepScreenOn = true
  @State private var draft = ""
  private let models = Models.shared
  @Environment(\.dismiss) private var dismiss

  var body: some View {
    NavigationStack {
      Form {
        Section {
          TextField("Address", text: $draft)
            .keyboardType(.numbersAndPunctuation)
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
          Button("Use the comma on the cable (\(Wire.commaHost))") { draft = Wire.commaHost }
        } header: {
          Text("Comma")
        } footer: {
          Text("A Mac playing the comma (sim/commasim) is at its own address; the iOS Simulator reaches it at 127.0.0.1.")
        }
        Section {
          Toggle("Keep Screen On", isOn: $keepScreenOn)
        } footer: {
          Text("While a comma streams the drive and Diorama is in front. A locked iPhone stops the drive until it is unlocked.")
        }
        Section {
          LabeledContent("Model", value: models.version.map { "\(StopLineModel.name) v\($0)" } ?? "not downloaded yet")
          LabeledContent("Update", value: models.status)
        } header: {
          Text("Stop-line model")
        } footer: {
          Text("Downloaded from Hugging Face (ernestchu/diorama) and checked for a newer version each time Diorama comes to the front. A new version is used from the next connection to the comma.")
        }
        Section {
          NavigationLink("Set Up Camera…") { DioramaCameraSetup() }
          NavigationLink {
            BenchmarkScreen()
          } label: {
            Label("Benchmark", systemImage: "stopwatch")
          }
        }
      }
      .navigationTitle("Settings")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .confirmationAction) {
          Button("Done") {
            let h = draft.trimmingCharacters(in: .whitespaces)
            if !h.isEmpty { host = h }
            dismiss()
          }
        }
      }
      .onAppear { draft = host }
    }
  }
}
