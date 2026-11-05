import Foundation

/// The files the scene draws with, bundled as this package's resources.
public enum DioramaPage {
  /// `name` from Resources, with no content type: only the renderer reads them.
  public static func file(_ name: String) -> (data: Data, type: String)? {
    guard let url = Bundle.module.url(forResource: name, withExtension: nil),
      let data = try? Data(contentsOf: url)
    else { return nil }
    return (data, "application/octet-stream")
  }
}
