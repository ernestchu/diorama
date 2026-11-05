import DioramaUI
import SwiftUI

/// Is this iPhone fast enough for the stop-line model, and does it stay fast enough? The model at the comma's pace,
/// a frame every 50 ms, on the phone alone. Laid out as jetlink's Benchmark (Zeph Leggett, MIT): the run, the
/// verdict, the totals, and the run ten seconds at a time.
struct BenchmarkScreen: View {
  @Environment(StopLineBenchmark.self) private var bench

  var body: some View {
    List {
      Section {
        runSection
      } footer: {
        Text("Run it with the iPhone charging and in its mount. The live view pauses while it runs.")
      }
      if let report = bench.report, !bench.running {
        Section("Verdict") { VerdictRow(report: report) }
        Section("Totals") { totals(report) }
        if report.windows.count > 1 {
          Section("Over Time, 10 s Each") {
            ForEach(report.windows) { w in WindowRow(window: w) }
          }
        }
      }
    }
    .navigationTitle("Benchmark")
    .toolbar {
      if let report = bench.report, !bench.running {
        ShareLink(item: report.text, preview: SharePreview("Diorama Benchmark"))
      }
    }
  }

  @ViewBuilder
  private var runSection: some View {
    VStack(alignment: .leading, spacing: 14) {
      LabeledContent("Stop-line model", value: UIDevice.current.model)
        .font(.headline)
      if bench.running {
        ProgressView(value: min(bench.elapsed / max(bench.total, 1), 1))
        HStack {
          Text("\(clock(bench.elapsed)) of \(clock(bench.total))")
          Spacer()
          Text("\(bench.frames.formatted()) frames")
        }
        .font(.subheadline.monospacedDigit())
        .foregroundStyle(.secondary)
        HStack(spacing: 0) {
          Figure(title: "P50", ms: bench.p50)
          Divider().frame(height: 32)
          Figure(title: "P99", ms: bench.p99)
        }
        Button(role: .destructive) {
          bench.cancel()
        } label: {
          Label("Cancel", systemImage: "stop.fill").frame(maxWidth: .infinity)
        }
        .buttonStyle(.glass)
        .controlSize(.large)
      } else {
        HStack(spacing: 12) {
          Button {
            bench.start(seconds: 60)
          } label: {
            // white like its title: on the prominent button the symbol would take the tint it sits on
            Label("1 Minute", systemImage: "play.fill").frame(maxWidth: .infinity).foregroundStyle(.white)
          }
          .buttonStyle(.glassProminent)
          Button {
            bench.start(seconds: 600)
          } label: {
            Label("10 Minutes", systemImage: "flame.fill").frame(maxWidth: .infinity)
          }
          .buttonStyle(.glass)
        }
        .controlSize(.large)
        .disabled(Models.stopLine == nil)
        if Models.stopLine == nil {
          Text("There is no stop-line model in this build.").font(.footnote).foregroundStyle(.orange)
        }
      }
      if let failure = bench.failure {
        Text(failure).font(.footnote).foregroundStyle(.red)
      }
    }
    .padding(.vertical, 4)
  }

  @ViewBuilder
  private func totals(_ r: StopLineBenchmark.Report) -> some View {
    LabeledContent("Frames", value: "\(r.frames.formatted()) in \(clock(r.seconds))")
    LabeledContent("Mean", value: ms(r.mean))
    LabeledContent("P50 / P99", value: "\(ms(r.p50)) / \(ms(r.p99))")
    LabeledContent("Slowest", value: ms(r.max))
    LabeledContent("Over 50 ms budget") {
      Text(r.over.formatted()).foregroundStyle(r.over > 0 ? .red : .secondary)
    }
    LabeledContent("Over 35 ms", value: r.tight.formatted())
    LabeledContent(
      "Temperature",
      value: r.thermalAtStart == r.thermalAtEnd ? r.thermalAtEnd.label.capitalized : "\(r.thermalAtStart.label.capitalized) to \(r.thermalAtEnd.label)")
  }
}

private struct Figure: View {
  let title: String
  let ms: Double?
  var body: some View {
    VStack(spacing: 2) {
      Text(ms.map { String(format: "%.1f ms", $0) } ?? "—").font(.title3.weight(.semibold).monospacedDigit())
      Text(title).font(.caption).foregroundStyle(.secondary)
    }
    .frame(maxWidth: .infinity)
  }
}

private struct VerdictRow: View {
  let report: StopLineBenchmark.Report
  var body: some View {
    let (title, symbol, tint, detail): (String, String, Color, String) =
      switch report.verdict {
      case .fast: ("Fast enough", "checkmark.seal.fill", .green, "Every frame well inside the 50 ms budget.")
      case .tight: ("Tight", "exclamationmark.triangle.fill", .orange, "P99 past 35 ms: little room beside other work.")
      case .slow: ("Too slow", "xmark.octagon.fill", .red, "Frames past the 50 ms budget: the stop line would lag the drive.")
      }
    VStack(alignment: .leading, spacing: 4) {
      Label(title + (report.cancelled ? " (stopped early)" : ""), systemImage: symbol).font(.headline).foregroundStyle(tint)
      Text(detail).font(.subheadline).foregroundStyle(.secondary)
    }
  }
}

private struct WindowRow: View {
  let window: StopLineBenchmark.Window
  var body: some View {
    HStack {
      Text("\(window.id * 10)–\(window.id * 10 + 10) s").monospacedDigit()
      Spacer()
      Text("\(ms(window.p50)) · \(ms(window.p99))").monospacedDigit().foregroundStyle(.secondary)
      if window.over > 0 { Text("\(window.over) over").foregroundStyle(.red) }
      Image(systemName: window.thermal.rawValue >= ProcessInfo.ThermalState.serious.rawValue ? "thermometer.high" : "thermometer.medium")
        .foregroundStyle(window.thermal.rawValue >= ProcessInfo.ThermalState.serious.rawValue ? .red : .secondary)
    }
    .font(.subheadline)
  }
}

private func ms(_ v: Double) -> String { String(format: "%.1f ms", v) }

private func clock(_ s: Double) -> String {
  let t = Int(s.rounded(.down))
  return String(format: "%d:%02d", t / 60, t % 60)
}
