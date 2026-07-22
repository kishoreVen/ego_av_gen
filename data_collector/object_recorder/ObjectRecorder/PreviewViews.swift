import SwiftUI
import ARKit
import SceneKit
import AVFoundation

/// Live camera preview for ARKit mode, backed by the controller's own
/// ARSession. Runs the session (for preview) as soon as this view appears,
/// independent of whether a recording is in progress, and pauses it again
/// when the view is torn down (e.g. switching to Max FPS mode) so the two
/// modes don't fight over the camera.
struct ARPreviewContainer: UIViewRepresentable {
    let controller: ARKitCaptureController

    func makeCoordinator() -> ARKitCaptureController { controller }

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView()
        view.session = controller.session
        view.automaticallyUpdatesLighting = true
        controller.startPreview()
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}

    static func dismantleUIView(_ uiView: ARSCNView, coordinator: ARKitCaptureController) {
        coordinator.stopPreview()
    }
}

/// Live camera preview for Max-FPS mode, backed by the controller's
/// AVCaptureSession. Same start-on-appear / stop-on-teardown behavior as
/// ARPreviewContainer, for the same reason.
struct CameraPreviewContainer: UIViewRepresentable {
    let controller: AVFoundationCaptureController

    func makeCoordinator() -> AVFoundationCaptureController { controller }

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.videoPreviewLayer.session = controller.session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        controller.startPreview()
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}

    static func dismantleUIView(_ uiView: PreviewView, coordinator: AVFoundationCaptureController) {
        coordinator.stopPreview()
    }
}

final class PreviewView: UIView {
    override static var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var videoPreviewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }
}
