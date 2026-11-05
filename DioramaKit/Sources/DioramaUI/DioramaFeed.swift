#if canImport(SwiftUI)
  import Foundation
  import DioramaCore

  /// Where the scene's messages come from: the comma over the link (`CommaFeed`).
  @MainActor
  public protocol DioramaFeed: AnyObject {
    /// From the meta on, the frames the models see as well with `input`;
    /// `deliver` returns once the scene has taken a batch, and no more come
    /// before it does. A second start replaces the first.
    func start(input: Bool, deliver: @escaping @MainActor @Sendable ([Data]) async -> Void)
    func stop()
  }

#endif
