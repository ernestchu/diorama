import SwiftUI

/// Diorama Bench: the stop-line models bundled in the app, timed on this iPhone with no Mac attached. The report is
/// a JSON file in the app's Documents (Files → On My iPhone → Diorama Bench), offered through the share sheet.
@main
struct BenchApp: App {
  @State private var run = BenchRun()

  var body: some Scene {
    WindowGroup { BenchView().environment(run) }
  }
}

struct BenchView: View {
  @Environment(BenchRun.self) private var run
  @State private var pacedSeconds = 180.0
  @State private var restSeconds = 60.0
  @State private var cpuOnly = true

  var body: some View {
    NavigationStack {
      Form {
        Section {
          ForEach(run.entries, id: \.name) { e in
            LabeledContent(e.name, value: String(format: "%.1fM · %.1f GMACs", e.params, e.gmacs))
          }
          if let error = run.loadError { Text(error).foregroundStyle(.red) }
        } header: {
          Text("\(run.entries.count) models")
        }
        Section {
          Picker("Paced at 20 Hz", selection: $pacedSeconds) {
            Text("30 s").tag(30.0)
            Text("1 min").tag(60.0)
            Text("3 min").tag(180.0)
            Text("5 min").tag(300.0)
          }
          Picker("Rest before each", selection: $restSeconds) {
            Text("none").tag(0.0)
            Text("30 s").tag(30.0)
            Text("1 min").tag(60.0)
            Text("2 min").tag(120.0)
          }
          Toggle("Also CPU only", isOn: $cpuOnly)
        } header: {
          Text("Settings")
        } footer: {
          Text(
            "After the rest, each model also waits until the iPhone is back to nominal temperature. About \(estimate) in all. Keep Diorama Bench in front: iOS stops an app in the background. The screen stays on by itself."
          )
        }
        .disabled(run.running)
        Section {
          if run.running {
            ProgressView(value: run.progress) { Text(run.phase).font(.footnote) }
            Button("Stop", role: .destructive) { run.cancel() }
          } else {
            Button("Run All") {
              run.start(BenchSettings(pacedSeconds: pacedSeconds, restSeconds: restSeconds, includeCPUOnly: cpuOnly))
            }
            .disabled(run.entries.isEmpty)
          }
          if case .failed(let why) = run.state { Text(why).foregroundStyle(.red) }
          if let url = run.reportURL {
            ShareLink(item: url) { Label(run.running ? "Share the report so far" : "Share the report", systemImage: "square.and.arrow.up") }
            Text(url.lastPathComponent).font(.footnote).foregroundStyle(.secondary)
          }
        }
        if !run.summaries.isEmpty {
          Section("Results") {
            ForEach(run.summaries, id: \.name) { s in
              VStack(alignment: .leading) {
                Text(s.name).font(.headline)
                Text(s.text).font(.footnote).foregroundStyle(.secondary)
              }
            }
          }
        }
      }
      .navigationTitle("Diorama Bench")
    }
    .onAppear {
      // a short check from the command line: xcrun devicectl device process launch ... -bench-quick
      if CommandLine.arguments.contains("-bench-quick"), !run.running {
        run.start(BenchSettings(pacedSeconds: 5, restSeconds: 0, maxCoolSeconds: 0, includeCPUOnly: true, warmup: 3, runs: 10, cpuOnlyRuns: 3))
      }
    }
  }

  private var estimate: String {
    let perModel = pacedSeconds + restSeconds + (cpuOnly ? 40 : 20)
    return "\(Int((perModel * Double(run.entries.count) / 60).rounded())) min"
  }
}
