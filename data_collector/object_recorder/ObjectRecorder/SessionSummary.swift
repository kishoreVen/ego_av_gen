import Foundation

/// Read-only view of a session already written to disk, used by the
/// sessions list. Distinct from `CaptureSession`, which creates a new
/// folder on init and is only for an in-progress session.
struct SessionSummary: Identifiable, Hashable {
    struct Recording: Decodable, Hashable {
        let name: String
        let mode: String
    }

    let url: URL
    let label: String
    let createdAt: Date
    let recordings: [Recording]

    var id: URL { url }
    var recordingCount: Int { recordings.count }
    var displayName: String { label.isEmpty ? "Untitled session" : label }

    /// Scans Documents for session_* folders, newest first.
    static func loadAll() -> [SessionSummary] {
        let fm = FileManager.default
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        guard let entries = try? fm.contentsOfDirectory(
            at: docs, includingPropertiesForKeys: [.creationDateKey, .isDirectoryKey]
        ) else {
            return []
        }

        let summaries: [SessionSummary] = entries.compactMap { url in
            guard url.lastPathComponent.hasPrefix("session_"),
                  (try? url.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true else {
                return nil
            }
            let createdAt = (try? url.resourceValues(forKeys: [.creationDateKey]))?.creationDate ?? .distantPast
            let manifestURL = url.appendingPathComponent("session_manifest.json")
            guard let data = try? Data(contentsOf: manifestURL) else {
                return SessionSummary(url: url, label: "", createdAt: createdAt, recordings: [])
            }
            struct Manifest: Decodable { let label: String; let recordings: [Recording] }
            let manifest = try? JSONDecoder().decode(Manifest.self, from: data)
            return SessionSummary(
                url: url,
                label: manifest?.label ?? "",
                createdAt: createdAt,
                recordings: manifest?.recordings ?? []
            )
        }
        return summaries.sorted { $0.createdAt > $1.createdAt }
    }
}
