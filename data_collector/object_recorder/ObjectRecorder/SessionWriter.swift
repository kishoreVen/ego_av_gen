import Foundation
import CoreVideo

/// Owns one recording session's output folder: video.mov, manifest.json,
/// frames.jsonl (one JSON object per captured frame) and raw/ (binary
/// per-frame payloads such as depth maps or confidence maps).
final class SessionWriter {
    let sessionURL: URL
    private let rawDir: URL
    private var manifest: [String: Any] = [:]
    private let framesFileHandle: FileHandle

    init(mode: String) {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let name = "session_\(mode)_\(Self.timestampString())"
        sessionURL = docs.appendingPathComponent(name)
        rawDir = sessionURL.appendingPathComponent("raw")
        try? FileManager.default.createDirectory(at: rawDir, withIntermediateDirectories: true)

        let framesURL = sessionURL.appendingPathComponent("frames.jsonl")
        FileManager.default.createFile(atPath: framesURL.path, contents: nil)
        framesFileHandle = FileHandle(forWritingAtPath: framesURL.path)!
    }

    var videoURL: URL { sessionURL.appendingPathComponent("video.mov") }

    /// Merges the given fields into manifest.json and persists it immediately.
    func updateManifest(_ fields: [String: Any]) {
        for (key, value) in fields { manifest[key] = value }
        let url = sessionURL.appendingPathComponent("manifest.json")
        if let data = try? JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: url)
        }
    }

    /// Appends one JSON line describing a captured frame. Nil values are dropped.
    func appendFrame(_ fields: [String: Any?]) {
        let clean = fields.compactMapValues { $0 }
        guard var data = try? JSONSerialization.data(withJSONObject: clean) else { return }
        data.append(0x0A)
        framesFileHandle.write(data)
    }

    /// Dumps a CVPixelBuffer's raw bytes (row by row, ignoring stride padding)
    /// to raw/<prefix>_<frameIndex>.bin and returns the relative path.
    func writeRawPixelBuffer(_ pixelBuffer: CVPixelBuffer, prefix: String, frameIndex: Int) -> String {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let bytesPerElement = bytesPerRow / max(width, 1)
        let tightRowBytes = width * bytesPerElement

        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return "" }

        var payload = Data(capacity: tightRowBytes * height)
        for row in 0..<height {
            let rowPointer = base.advanced(by: row * bytesPerRow)
            payload.append(Data(bytes: rowPointer, count: tightRowBytes))
        }

        let filename = "\(prefix)_\(String(format: "%06d", frameIndex)).bin"
        let url = rawDir.appendingPathComponent(filename)
        try? payload.write(to: url)
        return "raw/\(filename)"
    }

    func finish() {
        try? framesFileHandle.close()
    }

    private static func timestampString() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return formatter.string(from: Date())
    }
}
