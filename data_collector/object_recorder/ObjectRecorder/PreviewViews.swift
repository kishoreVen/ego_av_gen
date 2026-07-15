import SwiftUI
import ARKit
import SceneKit
import AVFoundation

/// Live camera preview for ARKit mode, backed by the controller's own ARSession.
struct ARPreviewContainer: UIViewRepresentable {
    let controller: ARKitCaptureController

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView()
        view.session = controller.session
        view.automaticallyUpdatesLighting = true
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

/// Live camera preview for Max-FPS mode, backed by the controller's AVCaptureSession.
struct CameraPreviewContainer: UIViewRepresentable {
    let controller: AVFoundationCaptureController

    func makeUIView(context: Context) -> PreviewView {
        controller.configureIfNeeded()
        let view = PreviewView()
        view.videoPreviewLayer.session = controller.session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}
}

final class PreviewView: UIView {
    override static var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var videoPreviewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }
}
