import AVFoundation
import Combine

struct HighFPSChoice {
    let format: AVCaptureDevice.Format
    let range: AVFrameRateRange
}

/// Picks the format + frame-rate range with the highest supported fps on the device.
func bestHighFPSFormat(for device: AVCaptureDevice) -> HighFPSChoice? {
    var best: HighFPSChoice?
    for format in device.formats {
        for range in format.videoSupportedFrameRateRanges {
            if best == nil || range.maxFrameRate > best!.range.maxFrameRate {
                best = HighFPSChoice(format: format, range: range)
            }
        }
    }
    return best
}

/// Plain single-camera AVCaptureSession tuned for the highest frame rate the
/// device supports. Trades away depth and true extrinsics (both require a
/// paired depth stream / multi-camera calibration) for raw fps. Per-frame
/// intrinsics are still delivered via camera intrinsic matrix delivery.
final class AVFoundationCaptureController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var isRecording = false
    @Published var status = "Idle"
    @Published var frameCount = 0
    @Published var activeFPS: Double = 0

    let session = AVCaptureSession()
    private let outputQueue = DispatchQueue(label: "objectrecorder.maxfps.output")

    private var writer: SessionWriter?
    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var frameIndex = 0
    private var sessionStarted = false
    private var configured = false
    private var isPreviewRunning = false

    func configureIfNeeded() {
        guard !configured else { return }
        configured = true

        session.beginConfiguration()
        session.sessionPreset = .inputPriority

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let choice = bestHighFPSFormat(for: device) else {
            status = "No suitable camera/format found"
            session.commitConfiguration()
            return
        }

        do {
            try device.lockForConfiguration()
            device.activeFormat = choice.format
            device.activeVideoMinFrameDuration = choice.range.minFrameDuration
            device.activeVideoMaxFrameDuration = choice.range.minFrameDuration
            device.unlockForConfiguration()
        } catch {
            status = "Format lock failed: \(error.localizedDescription)"
            session.commitConfiguration()
            return
        }

        if let input = try? AVCaptureDeviceInput(device: device), session.canAddInput(input) {
            session.addInput(input)
        }

        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: outputQueue)
        if session.canAddOutput(output) {
            session.addOutput(output)
        }
        if let connection = output.connection(with: .video) {
            if connection.isCameraIntrinsicMatrixDeliverySupported {
                connection.isCameraIntrinsicMatrixDeliveryEnabled = true
            }
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
        }

        session.commitConfiguration()
        activeFPS = choice.range.maxFrameRate
    }

    /// Starts the capture session for a live camera feed without recording
    /// anything. Safe to call repeatedly; a no-op once already running.
    func startPreview() {
        configureIfNeeded()
        guard !isPreviewRunning, !session.isRunning else { return }
        isPreviewRunning = true
        outputQueue.async { [session] in session.startRunning() }
    }

    /// Stops the capture session's camera feed. No-ops while recording so
    /// the mode picker (which drives this via view teardown) can't cut a
    /// recording short.
    func stopPreview() {
        guard isPreviewRunning, !isRecording else { return }
        isPreviewRunning = false
        outputQueue.async { [session] in session.stopRunning() }
    }

    func start(in captureSession: CaptureSession) {
        guard !isRecording else { return }
        startPreview()

        let name = captureSession.nextRecordingName(mode: "maxfps")
        writer = SessionWriter(parentURL: captureSession.url, name: name)
        captureSession.registerRecording(name: name, mode: "maxfps")
        frameIndex = 0
        sessionStarted = false
        assetWriter = nil
        isRecording = true
        status = "Recording (Max FPS)\u{2026}"
    }

    func stop(onFinished: (() -> Void)? = nil) {
        guard isRecording else { return }
        isRecording = false
        // Session keeps running (unlike before) so the live preview
        // continues between recordings within the same capture session.

        videoInput?.markAsFinished()
        assetWriter?.finishWriting { [weak self] in
            DispatchQueue.main.async {
                self?.writer?.finish()
                self?.status = "Saved: \(self?.writer?.sessionURL.lastPathComponent ?? "")"
                onFinished?()
            }
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard isRecording else { return }

        if assetWriter == nil {
            setupAssetWriter(sampleBuffer: sampleBuffer)
        }
        guard let assetWriter, let videoInput else { return }

        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        if !sessionStarted {
            sessionStarted = true
            assetWriter.startWriting()
            assetWriter.startSession(atSourceTime: pts)
        }
        if videoInput.isReadyForMoreMediaData {
            videoInput.append(sampleBuffer)
        }

        writer?.appendFrame([
            "frame_index": frameIndex,
            "timestamp": pts.seconds,
            "intrinsics": cameraIntrinsics(from: sampleBuffer)
        ])

        frameIndex += 1
        let count = frameIndex
        DispatchQueue.main.async { self.frameCount = count }
    }

    private func setupAssetWriter(sampleBuffer: CMSampleBuffer) {
        guard let formatDesc = CMSampleBufferGetFormatDescription(sampleBuffer) else { return }
        let dims = CMVideoFormatDescriptionGetDimensions(formatDesc)
        guard let url = writer?.videoURL else { return }
        try? FileManager.default.removeItem(at: url)

        guard let aw = try? AVAssetWriter(outputURL: url, fileType: .mov) else { return }
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: Int(dims.width),
            AVVideoHeightKey: Int(dims.height)
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = true
        if aw.canAdd(input) { aw.add(input) }

        assetWriter = aw
        videoInput = input

        writer?.updateManifest([
            "mode": "maxfps",
            "width": Int(dims.width),
            "height": Int(dims.height),
            "target_fps": activeFPS,
            "has_depth": false,
            "has_extrinsics": false,
            "video_codec": "h264",
            "note": "Max-FPS mode trades depth and extrinsics for frame rate; per-frame intrinsics are included when available."
        ])
    }
}
