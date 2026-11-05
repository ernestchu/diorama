import DioramaLink
import DioramaUI
import SwiftUI

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

struct RootView: View {
  @AppStorage(CommaAddress.key) private var host = CommaAddress.standard
  @State private var feed: CommaFeed?
  @State private var settings = false
  @Environment(\.scenePhase) private var phase
  @Environment(StopLineBenchmark.self) private var benchmark

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
    .onAppear { connect() }
    .onChange(of: host) { connect() }
  }

  private func connect() {
    feed?.stop()
    feed = CommaFeed(host: host, name: UIDevice.current.name, model: Models.stopLine)
  }
}

@MainActor
enum Models {
  /// Loaded once: compiling for the Neural Engine takes a moment.
  static let stopLine = StopLineModel.bundled()
}

struct SettingsView: View {
  @Binding var host: String
  @State private var draft = ""
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
        Section("Stop-line model") {
          LabeledContent("Model", value: Models.stopLine == nil ? "not in this build" : StopLineModel.name)
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
