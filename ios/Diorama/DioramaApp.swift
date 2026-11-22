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

/// Light or dark: as the system has it, set, or Auto Detect, which follows the footage (FootageTone), dark for a
/// night drive and light for a day one. Until the footage has said, Auto Detect follows the system.
enum AppTheme: String, CaseIterable, Identifiable {
  case system, auto, light, dark
  static let key = "theme"
  static let standard = AppTheme.auto

  var id: Self { self }
  var label: String {
    switch self {
    case .system: "System"
    case .auto: "Auto Detect"
    case .light: "Light"
    case .dark: "Dark"
    }
  }

  /// The scheme to ask for, given what the footage last said (nil: nothing yet); nil follows the system.
  func scheme(footageDark: Bool?) -> ColorScheme? {
    switch self {
    case .system: nil
    case .auto: footageDark.map { $0 ? .dark : .light }
    case .light: .light
    case .dark: .dark
    }
  }
}

struct RootView: View {
  @AppStorage(CommaAddress.key) private var host = CommaAddress.standard
  @AppStorage(KeepAwake.key) private var keepScreenOn = true
  @AppStorage(AppTheme.key) private var theme = AppTheme.standard
  /// What the footage last said: dark or not, nil before it has. Kept across reconnections to the comma.
  @State private var footageDark: Bool?
  @State private var feed: CommaFeed?
  @State private var player = DioramaPlayer()
  @State private var settings = false
  /// Width over height of the whole screen, as the drive is drawn.
  @State private var aspect: CGFloat = 0.5
  @Environment(\.scenePhase) private var phase
  @Environment(StopLineBenchmark.self) private var benchmark
  private let models = Models.shared

  var body: some View {
    Group {
      // the live view stops while a benchmark runs, so the two never share the model's hardware
      if let feed, phase != .background, !benchmark.running {
        DioramaSceneView(feed: feed, player: player, footageTop: 64)
      } else {
        Color.clear
      }
    }
    .overlay(alignment: .topTrailing) {
      Button("Settings", systemImage: "gearshape") { settings = true }
        .labelStyle(.iconOnly)
        .font(.title2)
        .buttonStyle(.glass)
        .buttonBorderShape(.circle)
        .padding(14)
    }
    .onGeometryChange(for: CGFloat.self) { g in
      let inset = g.safeAreaInsets
      return (g.size.width + inset.leading + inset.trailing) / max(g.size.height + inset.top + inset.bottom, 1)
    } action: {
      aspect = $0
    }
    // the sheet too, as it keeps the scheme it was presented with
    .sheet(isPresented: $settings) {
      SettingsView(host: $host, player: player, feed: feed, aspect: aspect).preferredColorScheme(theme.scheme(footageDark: footageDark))
    }
    .preferredColorScheme(theme.scheme(footageDark: footageDark))
    .onChange(of: feed?.footageDark) { if let d = feed?.footageDark { footageDark = d } }
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
  let player: DioramaPlayer
  let feed: CommaFeed?
  let aspect: CGFloat
  @AppStorage(KeepAwake.key) private var keepScreenOn = true
  @AppStorage(DioramaSceneView.inputKey) private var showInput = false
  @AppStorage(DioramaSceneView.dioramaKey) private var showDiorama = true
  @AppStorage(DioramaSceneView.planSecondsKey) private var planSeconds = DioramaSceneView.planSecondsDefault
  @AppStorage(AppTheme.key) private var theme = AppTheme.standard
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
          Picker("Theme", selection: $theme) {
            ForEach(AppTheme.allCases) { Text($0.label).tag($0) }
          }
        } footer: {
          Text(
            "Auto Detect follows the footage's sky: dark at night, light by day. It goes by the last two minutes of footage, so a wall, an underpass or the sky coming in and out of view at sunset does not flip it. Until it has ten seconds of footage, it follows the system.")
        }
        Section {
          Toggle("Keep Screen On", isOn: $keepScreenOn)
        } footer: {
          Text("While a comma streams the drive and Diorama is in front. A locked iPhone stops the drive until it is unlocked.")
        }
        Section {
          DioramaStats(player: player, feed: feed)
          Toggle("Show footage & prediction", isOn: $showInput)
          Toggle("Show diorama prediction", isOn: $showDiorama)
            .disabled(!showInput)
        } header: {
          Text("Drive")
        } footer: {
          Text("The frame the models see, with the comma's prediction drawn as on its display. Diorama's own prediction, the stop line, is drawn over it too when that is on.")
        }
        Section {
          Stepper(value: $planSeconds, in: 0.5...10, step: 0.5) {
            LabeledContent("Planned Path", value: "\(planSeconds.formatted(.number.precision(.fractionLength(0...1)))) s")
          }
        } footer: {
          Text("How far ahead the diorama draws the comma's plan: this many seconds of driving, or 80 m, whichever comes first. It also stops behind the car ahead, and where the plan, having braked, speeds up again, as at a stop sign.")
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
          NavigationLink("Set Up Camera…") { DioramaCameraSetup(aspect: aspect) }
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
