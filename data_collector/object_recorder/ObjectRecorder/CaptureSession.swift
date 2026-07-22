import Foundation

/// Owns one "object capture session": a parent folder that can contain
/// multiple recordings (ARKit and/or Max FPS, any mix, any order) of the
/// same physical object. Each recording gets its own numbered subfolder;
/// session_manifest.json lists them so the whole folder can be zipped and
/// uploaded as a single unit when the session ends.
final class CaptureSession {
    let url: URL
    /// Sanitized version of the name passed at init, empty if none was given.
    let label: String
    private(set) var recordingCount = 0
    private var recordings: [[String: String]] = []

    /// - Parameter name: optional label (e.g. the object being captured),
    ///   folded into the session folder name and stored in
    ///   session_manifest.json for the sessions list to read back.
    ///   Non-filesystem-safe characters are replaced.
    init(name: String? = nil) {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        label = Self.sanitize(name ?? "")
        let prefix = label.isEmpty ? "" : "\(label)_"
        url = docs.appendingPathComponent("session_\(prefix)\(Self.timestampString())")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        writeManifest()
    }

    private static func sanitize(_ raw: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        let cleaned = raw.trimmingCharacters(in: .whitespacesAndNewlines).unicodeScalars.map {
            allowed.contains($0) ? Character($0) : "_"
        }
        return String(cleaned).trimmingCharacters(in: CharacterSet(charactersIn: "_"))
    }

    /// Reserves the next recording's subfolder name for the given mode.
    func nextRecordingName(mode: String) -> String {
        recordingCount += 1
        return "\(String(format: "%02d", recordingCount))_\(mode)_\(Self.timestampString())"
    }

    /// Registers a reserved recording so it shows up in session_manifest.json.
    func registerRecording(name: String, mode: String) {
        recordings.append(["name": name, "mode": mode])
        writeManifest()
    }

    private func writeManifest() {
        let manifest: [String: Any] = ["label": label, "recordings": recordings]
        if let data = try? JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: url.appendingPathComponent("session_manifest.json"))
        }
    }

    private static func timestampString() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return formatter.string(from: Date())
    }
}

/// Identity-based conformance so a CaptureSession can drive SwiftUI
/// navigation (`navigationDestination(item:)`).
extension CaptureSession: Hashable {
    static func == (lhs: CaptureSession, rhs: CaptureSession) -> Bool { lhs === rhs }
    func hash(into hasher: inout Hasher) { hasher.combine(ObjectIdentifier(self)) }
}
