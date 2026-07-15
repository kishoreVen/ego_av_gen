import SwiftUI
import AVFoundation

struct ContentView: View {
    @State private var mode: CaptureMode = .arKit
    @State private var cameraAuthorized = false
    @StateObject private var arController = ARKitCaptureController()
    @StateObject private var maxFPSController = AVFoundationCaptureController()

    private var isRecording: Bool {
        mode == .arKit ? arController.isRecording : maxFPSController.isRecording
    }

    private var status: String {
        mode == .arKit ? arController.status : maxFPSController.status
    }

    private var frameCount: Int {
        mode == .arKit ? arController.frameCount : maxFPSController.frameCount
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if cameraAuthorized {
                switch mode {
                case .arKit:
                    ARPreviewContainer(controller: arController).ignoresSafeArea()
                case .maxFPS:
                    CameraPreviewContainer(controller: maxFPSController).ignoresSafeArea()
                }
            }

            VStack {
                modePicker
                Spacer()
                statusPanel
            }
            .padding()
        }
        .onAppear(perform: requestCameraAccess)
        .statusBarHidden()
    }

    private var modePicker: some View {
        Picker("Mode", selection: $mode) {
            ForEach(CaptureMode.allCases) { mode in
                Text(mode.rawValue).tag(mode)
            }
        }
        .pickerStyle(.segmented)
        .disabled(isRecording)
        .padding(.top, 8)
    }

    private var statusPanel: some View {
        VStack(spacing: 8) {
            Text(CaptureMode.arKit == mode ? CaptureMode.arKit.subtitle : CaptureMode.maxFPS.subtitle)
                .font(.caption)
                .foregroundStyle(.white.opacity(0.7))
            Text("\(status)  \u{2022}  frames: \(frameCount)")
                .font(.footnote.monospaced())
                .foregroundStyle(.white)

            Button(action: toggleRecording) {
                Circle()
                    .fill(isRecording ? Color.red : Color.white)
                    .frame(width: 72, height: 72)
                    .overlay(
                        RoundedRectangle(cornerRadius: isRecording ? 6 : 36)
                            .fill(Color.red)
                            .frame(width: isRecording ? 28 : 64, height: isRecording ? 28 : 64)
                    )
            }
            .disabled(!cameraAuthorized)
            .padding(.bottom, 24)
        }
        .padding()
        .background(.black.opacity(0.4))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private func toggleRecording() {
        switch mode {
        case .arKit:
            arController.isRecording ? arController.stop() : arController.start()
        case .maxFPS:
            maxFPSController.isRecording ? maxFPSController.stop() : maxFPSController.start()
        }
    }

    private func requestCameraAccess() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            cameraAuthorized = true
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async { cameraAuthorized = granted }
            }
        default:
            cameraAuthorized = false
        }
    }
}

#Preview {
    ContentView()
}
