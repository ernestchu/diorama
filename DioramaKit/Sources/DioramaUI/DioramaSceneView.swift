#if canImport(SwiftUI) && canImport(MetalKit)
  import DioramaCore
  import MetalKit
  import SwiftUI

  /// The drive as a live 3-D miniature, drawn natively in Metal from a feed's
  /// messages, edge to edge, with the HUD over it. The same on a Mac and an
  /// iPhone, and the same drawing as the page's. The player is the caller's, so
  /// that DioramaStats can show its timings elsewhere.
  public struct DioramaSceneView: View {
    /// Where the footage switch is kept: set it and every scene view follows.
    public static let inputKey = "diorama.input"
    /// Where the switch for the stop-line model's prediction on the footage is kept.
    public static let dioramaKey = "dioramaPrediction"
    /// Where the number of seconds of the plan to draw is kept.
    public static let planSecondsKey = "planSeconds"
    public static let planSecondsDefault = 2.0

    let feed: any DioramaFeed
    let player: DioramaPlayer
    /// Whether the frames are shown at all: off for a feed that never carries them.
    let modelInput: Bool
    /// How far below the safe area's top the footage sits, to clear what the caller puts over the scene.
    let footageTop: CGFloat
    @AppStorage(DioramaSceneView.inputKey) private var showInput = false
    @AppStorage(DioramaSceneView.dioramaKey) private var showDiorama = true
    @AppStorage(DioramaCamera.storageKey) private var camera = DioramaCamera.defaults
    @AppStorage(DioramaSceneView.planSecondsKey) private var planSeconds = DioramaSceneView.planSecondsDefault
    @Environment(\.colorScheme) private var colorScheme

    public init(feed: any DioramaFeed, player: DioramaPlayer, modelInput: Bool = true, footageTop: CGFloat = 0) {
      self.feed = feed
      self.player = player
      self.modelInput = modelInput
      self.footageTop = footageTop
    }

    public var body: some View {
      ZStack(alignment: .topLeading) {
        DioramaMetalView(player: player, dark: colorScheme == .dark, camera: camera, planSeconds: planSeconds, cutAtLead: true, followHeading: true)
          .ignoresSafeArea()
        if let hud = player.hud { DioramaHUD(hud: hud).padding(14) }
        if modelInput && showInput, let picture = player.input {
          DioramaInputView(picture: picture)
            .frame(maxWidth: 512)
            .containerRelativeFrame(.horizontal) { width, _ in min(width * 0.41, 512) }
            .padding(.horizontal, 14)
            .padding(.top, footageTop)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
            .allowsHitTesting(false)
        }
        TimelineView(.periodic(from: .now, by: 0.5)) { context in
          if player.waiting(at: context.date) {
            Text("Waiting for the comma…")
              .font(.system(size: 15))
              .foregroundStyle(Color(white: 0.73))
              .multilineTextAlignment(.center)
              .padding(24)
              .frame(maxWidth: .infinity, maxHeight: .infinity)
              .background(Color(red: 0.06, green: 0.063, blue: 0.07).opacity(0.6).ignoresSafeArea())
          }
        }
        .allowsHitTesting(false)
      }
      .onAppear { start() }
      .onDisappear { feed.stop() }
      .onChange(of: ObjectIdentifier(feed)) {
        // the server restarted, or the source changed: a new feed, and the scene starts over from its meta
        start()
      }
      .onChange(of: showInput) {
        if !showInput { player.clearInput() }
        start()
      }
      .onChange(of: showDiorama) { player.showDiorama = showDiorama }
    }

    private func start() {
      let input = modelInput && showInput
      player.showInput = input
      player.showDiorama = showDiorama
      feed.start(input: input) { [player] batch in await player.take(batch) }
    }
  }

  /// The speed, the lead, the view, and the acceleration as a bar, as the page's HUD.
  struct DioramaHUD: View {
    let hud: DioramaPlayer.HUD

    var body: some View {
      VStack(alignment: .leading, spacing: 0) {
        Text("\(hud.speed)")
          .font(.system(size: 44, weight: .semibold).monospacedDigit())
          .foregroundStyle(Color(white: 0.95))
        Text("MPH").font(.system(size: 12)).kerning(1).foregroundStyle(Color(white: 0.66))
        if let lead = hud.lead {
          Text("lead \(lead) m").font(.system(size: 12)).foregroundStyle(Color(white: 0.77)).padding(.top, 6)
        }
        Text(hud.mode).font(.system(size: 12)).kerning(0.7).foregroundStyle(Color(white: 0.55)).padding(.top, 6)
      }
      .padding(.vertical, 12)
      .padding(.leading, 16)
      .padding(.trailing, 36)
      .frame(minWidth: 150, alignment: .leading)
      .overlay(alignment: .trailing) { accel.padding(14) }
      .background(Color(red: 0.086, green: 0.09, blue: 0.1).opacity(0.86), in: .rect(cornerRadius: 12))
      .allowsHitTesting(false)
    }

    /// Up from the middle accelerating, down braking.
    private var accel: some View {
      GeometryReader { geo in
        let a = Diorama.clamp(hud.accel, -3.2, 3.2), h = geo.size.height
        let top = (0.5 - max(a, 0) / 3.2 * 0.5) * h, bottom = (0.5 + min(a, 0) / 3.2 * 0.5) * h
        let color = a > 0.2 ? Color(red: 0.35, green: 0.82, blue: 0.48) : a < -0.5 ? Color(red: 1, green: 0.35, blue: 0.32) : Color(white: 0.78)
        ZStack(alignment: .top) {
          Capsule().fill(Color(red: 0.23, green: 0.24, blue: 0.26))
          Capsule().fill(color).frame(height: max(h - top - bottom, 0)).offset(y: top)
        }
      }
      .frame(width: 6)
    }
  }

  /// The models' timings, and whether the comma is connected with the full message a tap away, as rows of a Form or
  /// a List in a navigation stack. The connection row is always there, on one line: a message that comes and goes
  /// as the link is dialed would move every row below it.
  public struct DioramaStats: View {
    let player: DioramaPlayer
    let feed: CommaFeed?

    public init(player: DioramaPlayer, feed: CommaFeed?) {
      self.player = player
      self.feed = feed
    }

    public var body: some View {
      // the driving model runs on the comma, which does not say how long it took
      row("Frames") { "\(Int(player.frameRate(at: $0).rounded())) Hz" }
      row("Stop-line model") { _ in ms(median(player.timing.seg)) }
      row("Stop-line rate") { _ in segRate(player.timing) }
      NavigationLink {
        DioramaModelsDetail(player: player, feed: feed)
      } label: {
        LabeledContent("Models", value: feed?.streaming == true ? "Connected" : "Not connected")
      }
    }

    private func row(_ name: String, _ value: @escaping (Date) -> String) -> some View {
      TimelineView(.periodic(from: .now, by: 0.5)) { context in
        LabeledContent(name) { Text(value(context.date)).monospacedDigit() }
      }
    }

    private func ms(_ v: Double?) -> String { v.map { String(format: "%.1f ms", $0) } ?? "–" }

    private func segRate(_ t: DioramaScene.Timing) -> String {
      let f = t.segFrames
      guard f.count > 1 else { return "–" }
      return String(format: "%.1f Hz", Diorama.fps * Double(f.count - 1) / Double(max(f[f.count - 1] - f[0], 1)))
    }

    private func median(_ a: [Double]) -> Double? { a.isEmpty ? nil : a.sorted()[a.count >> 1] }
  }

  /// The models' full state, in words: the message the comma's link last gave.
  struct DioramaModelsDetail: View {
    let player: DioramaPlayer
    let feed: CommaFeed?

    private var connected: Bool { feed?.streaming == true }

    private var message: String {
      guard let models = player.models else { return connected ? "Running" : "Not connected" }
      return models.prefix(1).uppercased() + models.dropFirst()
    }

    var body: some View {
      Form {
        LabeledContent("Comma", value: connected ? "Connected" : "Not connected")
        Section("Message") {
          Text(message).textSelection(.enabled)
        }
      }
      .navigationTitle("Models")
      #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
      #endif
    }
  }

  /// The frame the models see, each thing named above it.
  struct DioramaInputView: View {
    let picture: InputPicture

    var body: some View {
      VStack(spacing: 0) {
        Image(decorative: picture.image, scale: 1)
          .resizable()
          .interpolation(.none)
          .aspectRatio(2, contentMode: .fit)
          .overlay {
            Canvas { context, size in
              let scale = size.height / picture.height
              context.withCGContext { cg in
                cg.scaleBy(x: scale, y: scale)
                CommaPainter.draw(picture.comma, height: picture.height, in: cg)
              }
            }
          }
          .overlay {
            GeometryReader { geo in
              let s = geo.size.width / 512
              ZStack(alignment: .topLeading) {
                ForEach(Array(picture.labels.enumerated()), id: \.offset) { _, label in
                  Text(label.text)
                    .font(.system(size: 10 * s, weight: .semibold))
                    .foregroundStyle(.black)
                    .padding(.horizontal, 3 * s)
                    .frame(height: 13 * s)
                    .background(Color(red: Double(label.rgb.x) / 255, green: Double(label.rgb.y) / 255, blue: Double(label.rgb.z) / 255).opacity(0.85))
                    .fixedSize()
                    .offset(x: max(label.x, 0) * s, y: max(label.y - 13, 0) * s)
                }
              }
              .frame(width: geo.size.width, height: geo.size.height, alignment: .topLeading)
            }
          }
        Text(picture.caption)
          .font(.system(size: 11).monospacedDigit())
          .foregroundStyle(Color(white: 0.87))
          .lineLimit(1)
          .truncationMode(.tail)
          .padding(.horizontal, 8)
          .padding(.vertical, 3)
          .frame(maxWidth: .infinity, alignment: .leading)
          .background(.black)
      }
      .background(.black)
      .clipShape(.rect(cornerRadius: 8))
      .shadow(color: .black.opacity(0.35), radius: 6, y: 2)
    }
  }


  #if os(macOS)
    typealias PlatformViewRepresentable = NSViewRepresentable
  #else
    typealias PlatformViewRepresentable = UIViewRepresentable
  #endif

  /// The Metal view the renderer draws into: on demand, once per batch the
  /// player takes, and when its size or the theme changes.
  struct DioramaMetalView: PlatformViewRepresentable {
    let player: DioramaPlayer
    let dark: Bool
    var camera = DioramaCamera.defaults
    /// One view whatever the drive's view chooser says.
    var fixedView: Double?
    /// The still scene the camera is set over, instead of the player's drive.
    var preview = false
    /// How many seconds of the plan to draw, at most; nil draws all of it.
    var planSeconds: Double?
    /// Whether the plan stops behind the lead.
    var cutAtLead = false
    /// Whether the camera follows the car's heading through a spring.
    var followHeading = false

    func makeCoordinator() -> Coordinator { Coordinator(player: player) }

    #if os(macOS)
      func makeNSView(context: Context) -> MTKView { make(context) }
      func updateNSView(_ view: MTKView, context: Context) { update(view, context) }
    #else
      func makeUIView(context: Context) -> MTKView { make(context) }
      func updateUIView(_ view: MTKView, context: Context) { update(view, context) }
    #endif

    private func make(_ context: Context) -> MTKView {
      let view = MTKView(frame: .zero, device: context.coordinator.renderer?.device ?? MTLCreateSystemDefaultDevice())
      view.colorPixelFormat = DioramaRenderer.colorFormat
      view.depthStencilPixelFormat = DioramaRenderer.depthFormat
      view.sampleCount = DioramaRenderer.sampleCount
      view.enableSetNeedsDisplay = true
      view.isPaused = true
      view.delegate = context.coordinator
      context.coordinator.view = view
      update(view, context)
      return view
    }

    private func update(_ view: MTKView, _ context: Context) {
      let coordinator = context.coordinator
      coordinator.player = player
      player.redraw = { [weak coordinator] in coordinator?.setNeedsDisplay() }
      if coordinator.dark != dark {
        coordinator.dark = dark
        coordinator.renderer?.theme = dark ? .dark : .light
        coordinator.setNeedsDisplay()
      }
      if let renderer = coordinator.renderer,
        renderer.camera != camera || renderer.fixedView != fixedView || renderer.planSeconds != planSeconds || renderer.cutAtLead != cutAtLead
          || renderer.followHeading != followHeading || coordinator.preview != preview
      {
        renderer.camera = camera
        renderer.fixedView = fixedView
        renderer.planSeconds = planSeconds
        renderer.cutAtLead = cutAtLead
        renderer.followHeading = followHeading
        coordinator.preview = preview
        coordinator.setNeedsDisplay()
      }
    }

    @MainActor
    final class Coordinator: NSObject, MTKViewDelegate {
      var player: DioramaPlayer
      let renderer: DioramaRenderer?
      weak var view: MTKView?
      var dark: Bool?
      var preview = false

      init(player: DioramaPlayer) {
        self.player = player
        renderer = MTLCreateSystemDefaultDevice().flatMap { DioramaRenderer(device: $0) }
      }

      private var drawPending = false

      /// Draws once more, soon: once for however many asks come before it does.
      func setNeedsDisplay() {
        #if os(macOS)
          // An MTKView drawn on demand by needsDisplay stops drawing once its window is
          // closed and opened again, as the camera's window is: SwiftUI keeps the view,
          // and AppKit never asks it to draw again. Drawing it directly always does.
          guard !drawPending else { return }
          drawPending = true
          DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.drawPending = false
            self.view?.draw()
          }
        #else
          view?.setNeedsDisplay()
        #endif
      }

      /// A new size draws again: with no drive playing, nothing else would.
      nonisolated func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        MainActor.assumeIsolated { setNeedsDisplay() }
      }

      nonisolated func draw(in view: MTKView) {
        MainActor.assumeIsolated {
          guard let renderer, let pass = view.currentRenderPassDescriptor else { return }
          let frame = preview ? DioramaRenderer.preview(view: renderer.fixedView) : player.box.with { renderer.take($0) }
          renderer.draw(frame, to: pass, size: view.drawableSize) { commands in
            if let drawable = view.currentDrawable { commands.present(drawable) }
          }
        }
      }
    }
  }
#endif
