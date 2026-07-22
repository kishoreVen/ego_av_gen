import SwiftUI

/// Read-only look at a past session's recordings. Sessions can't be
/// resumed for recording — start a new one from the sessions list instead.
struct SessionDetailView: View {
    let session: SessionSummary

    var body: some View {
        List {
            Section {
                LabeledContent("Created", value: session.createdAt.formatted(date: .abbreviated, time: .standard))
                LabeledContent("Recordings", value: "\(session.recordingCount)")
            }

            Section("Recordings") {
                ForEach(session.recordings, id: \.name) { recording in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(recording.name)
                            .font(.body.monospaced())
                        Text(recording.mode)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .navigationTitle(session.displayName)
    }
}
