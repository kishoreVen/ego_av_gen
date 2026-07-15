import ARKit
import AVFoundation
import Combine

/// Drives an ARSession in world-tracking mode and records, per frame:
/// video (capturedImage), intrinsics, extrinsics (camera pose), LiDAR
/// scene depth + confidence, light estimate, and device exposure/WB state.
final class ARKitCaptureController: NSObject, ObservableObject, ARSessionDelegate {
    @Published var isRecording = false
    @Published var status = "Idle"
    @Published var frameCount = 0

    let session = ARSession()
    let hasLiDAR = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)

    private var writer: SessionWriter?
    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var sessionStartTime: TimeInterval?
    private var frameIndex = 0
    private var didWriteDepthInfo = false

    /// Separate handle onto the physical back camera purely to read live
    /// exposure/white-balance/ISO state while ARKit owns the capture session.
    private let backCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)

    func start() {
        guard !isRecording, ARWorldTrackingConfiguration.isSupported else { return }

        let config = ARWorldTrackingConfiguration()
        if let bestFormat = ARWorldTrackingConfiguration.supportedVideoFormats.max(by: {
            $0.framesPerSecond < $1.framesPerSecond ||
            ($0.framesPerSecond == $1.framesPerSecond &&
             $0.imageResolution.width * $0.imageResolution.height < $1.imageResolution.width * $1.imageResolution.height)
        }) {
            config.videoFormat = bestFormat
        }
        if hasLiDAR {
            config.frameSemantics.insert(.sceneDepth)
        }
        config.worldAlignment = .gravity

        session.delegate = self
        session.run(config, options: [.resetTracking, .removeExistingAnchors])

        writer = SessionWriter(mode: "arkit")
        frameIndex = 0
        sessionStartTime = nil
        assetWriter = nil
        didWriteDepthInfo = false
        isRecording = true
        status = "Recording (ARKit)\u{2026}"
    }

    func stop() {
        guard isRecording else { return }
        isRecording = false
        session.pause()

        videoInput?.markAsFinished()
        assetWriter?.finishWriting { [weak self] in
            DispatchQueue.main.async {
                self?.writer?.finish()
                self?.status = "Saved: \(self?.writer?.sessionURL.lastPathComponent ?? "")"
            }
        }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard isRecording else { return }

        let pixelBuffer = frame.capturedImage
        if assetWriter == nil {
            setupAssetWriter(pixelBuffer: pixelBuffer)
        }
        guard let assetWriter, let videoInput, let pixelBufferAdaptor else { return }

        if sessionStartTime == nil {
            sessionStartTime = frame.timestamp
            assetWriter.startWriting()
            assetWriter.startSession(atSourceTime: .zero)
        }

        let pts = CMTime(seconds: frame.timestamp - sessionStartTime!, preferredTimescale: 1_000_000)
        if videoInput.isReadyForMoreMediaData {
            pixelBufferAdaptor.append(pixelBuffer, withPresentationTime: pts)
        }

        var depthPath: String?
        var confidencePath: String?
        if let sceneDepth = frame.sceneDepth {
            depthPath = writer?.writeRawPixelBuffer(sceneDepth.depthMap, prefix: "depth", frameIndex: frameIndex)
            if let confidenceMap = sceneDepth.confidenceMap {
                confidencePath = writer?.writeRawPixelBuffer(confidenceMap, prefix: "confidence", frameIndex: frameIndex)
            }
            if !didWriteDepthInfo {
                didWriteDepthInfo = true
                writer?.updateManifest([
                    "depth_width": CVPixelBufferGetWidth(sceneDepth.depthMap),
                    "depth_height": CVPixelBufferGetHeight(sceneDepth.depthMap),
                    "depth_format": "float32_meters",
                    "confidence_format": "uint8_0to2"
                ])
            }
        }

        writer?.appendFrame([
            "frame_index": frameIndex,
            "timestamp": frame.timestamp,
            "intrinsics": matrix3x3ToArray(frame.camera.intrinsics),
            "extrinsics": matrix4x4ToArray(frame.camera.transform),
            "depth_file": depthPath,
            "confidence_file": confidencePath,
            "light_estimate": lightEstimateDict(frame.lightEstimate),
            "exposure": exposureDict()
        ])

        frameIndex += 1
        let count = frameIndex
        DispatchQueue.main.async { self.frameCount = count }
    }

    private func lightEstimateDict(_ estimate: ARLightEstimate?) -> [String: Any]? {
        guard let estimate else { return nil }
        var dict: [String: Any] = [
            "ambient_intensity": estimate.ambientIntensity,
            "ambient_color_temperature": estimate.ambientColorTemperature
        ]
        if let directional = estimate as? ARDirectionalLightEstimate {
            dict["primary_light_direction"] = [
                directional.primaryLightDirection.x,
                directional.primaryLightDirection.y,
                directional.primaryLightDirection.z
            ]
            dict["primary_light_intensity"] = directional.primaryLightIntensity
            let coeffData = directional.sphericalHarmonicsCoefficients
            let floatCount = coeffData.count / MemoryLayout<Float32>.size
            var coeffs = [Float32](repeating: 0, count: floatCount)
            coeffData.withUnsafeBytes { raw in
                coeffs = Array(raw.bindMemory(to: Float32.self))
            }
            dict["spherical_harmonics"] = coeffs
        }
        return dict
    }

    private func exposureDict() -> [String: Any]? {
        guard let device = backCamera else { return nil }
        let gains = device.deviceWhiteBalanceGains
        return [
            "iso": device.iso,
            "exposure_duration_seconds": CMTimeGetSeconds(device.exposureDuration),
            "lens_position": device.lensPosition,
            "white_balance_gains": [
                "red": gains.redGain,
                "green": gains.greenGain,
                "blue": gains.blueGain
            ]
        ]
    }

    private func setupAssetWriter(pixelBuffer: CVPixelBuffer) {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard let url = writer?.videoURL else { return }
        try? FileManager.default.removeItem(at: url)

        guard let aw = try? AVAssetWriter(outputURL: url, fileType: .mov) else { return }
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = true

        let adaptorAttrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: CVPixelBufferGetPixelFormatType(pixelBuffer)
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: adaptorAttrs)

        if aw.canAdd(input) { aw.add(input) }
        assetWriter = aw
        videoInput = input
        pixelBufferAdaptor = adaptor

        writer?.updateManifest([
            "mode": "arkit",
            "width": width,
            "height": height,
            "has_lidar_depth": hasLiDAR,
            "video_codec": "hevc"
        ])
    }
}
