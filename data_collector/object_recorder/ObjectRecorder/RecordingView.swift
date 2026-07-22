import SwiftUI
import AVFoundation

/// Camera + recording screen for one already-started capture session.
/// Reached by picking "New Session" on the sessions list; ending the
/// session here pops back to that list.
struct RecordingView: View {
    let captureSession: CaptureSession
    var onSessionEnded: () -> Void

    @State private var mode: CaptureMode = .arKit
    @State private var cameraAuthorized = false
    @State private var isEndingSession = false
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
        .navigationBarBackButtonHidden(true)
        .toolbar(.hidden, for: .navigationBar)
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
            Text(sessionStatusText)
                .font(.footnote.monospaced())
                .foregroundStyle(.white)

            HStack(spacing: 32) {
                Button(action: endSession) {
                    Text("End Session")
                        .font(.subheadline.bold())
                        .foregroundStyle(.white)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 10)
                        .background(Color.orange)
                        .clipShape(Capsule())
                }
                .disabled(isEndingSession)

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
                .disabled(!cameraAuthorized || isEndingSession)
            }
            .padding(.bottom, 24)
        }
        .padding()
        .background(.black.opacity(0.4))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private var sessionStatusText: String {
        let label = captureSession.label.isEmpty ? "session" : captureSession.label
        return "\(label)  \u{2022}  \(captureSession.recordingCount) recording(s)  \u{2022}  \(status)  \u{2022}  frames: \(frameCount)"
    }

    private func endSession() {
        isEndingSession = true
        if isRecording {
            stopActiveRecording { self.isEndingSession = false; self.onSessionEnded() }
        } else {
            isEndingSession = false
            onSessionEnded()
        }
    }

    private func toggleRecording() {
        if isRecording {
            stopActiveRecording()
        } else {
            switch mode {
            case .arKit: arController.start(in: captureSession)
            case .maxFPS: maxFPSController.start(in: captureSession)
            }
        }
    }

    private func stopActiveRecording(onFinished: (() -> Void)? = nil) {
        switch mode {
        case .arKit: arController.stop(onFinished: onFinished)
        case .maxFPS: maxFPSController.stop(onFinished: onFinished)
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
