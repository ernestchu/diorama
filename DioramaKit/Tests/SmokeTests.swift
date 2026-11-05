import Testing
@testable import DioramaCore

@Test func assetsAreBundled() { #expect(DioramaPage.file("diorama-assets.bin") != nil) }
