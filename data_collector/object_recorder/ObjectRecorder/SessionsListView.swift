import SwiftUI

/// App's home screen: every past capture session, newest first, plus a way
/// to start a new one.
struct SessionsListView: View {
    @State private var sessions: [SessionSummary] = []
    @State private var newSessionName = ""
    @State private var activeCaptureSession: CaptureSession?

    var body: some View {
        List {
            Section {
                HStack {
                    TextField("Object name", text: $newSessionName)
                        .textFieldStyle(.roundedBorder)
                        .autocorrectionDisabled()
                    Button("New Session", action: startNewSession)
                        .buttonStyle(.borderedProminent)
                        .disabled(trimmedSessionName.isEmpty)
                }
            }

            Section("Sessions") {
                if sessions.isEmpty {
                    Text("No sessions yet")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(sessions) { session in
                        NavigationLink(value: session) {
                            SessionRow(session: session)
                        }
                    }
                    .onDelete(perform: deleteSessions)
                }
            }
        }
        .navigationTitle("Object Recorder")
        .navigationDestination(item: $activeCaptureSession) { session in
            RecordingView(captureSession: session) {
                activeCaptureSession = nil
                reload()
            }
        }
        .navigationDestination(for: SessionSummary.self) { session in
            SessionDetailView(session: session)
        }
        .onAppear(perform: reload)
    }

    private var trimmedSessionName: String {
        newSessionName.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func startNewSession() {
        guard !trimmedSessionName.isEmpty else { return }
        activeCaptureSession = CaptureSession(name: trimmedSessionName)
        newSessionName = ""
    }

    private func reload() {
        sessions = SessionSummary.loadAll()
    }

    private func deleteSessions(at offsets: IndexSet) {
        for index in offsets {
            try? FileManager.default.removeItem(at: sessions[index].url)
        }
        sessions.remove(atOffsets: offsets)
    }
}

private struct SessionRow: View {
    let session: SessionSummary

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(session.displayName)
                .font(.headline)
            Text("\(session.recordingCount) recording(s)  \u{2022}  \(session.createdAt.formatted(date: .abbreviated, time: .shortened))")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }
}

#Preview {
    NavigationStack {
        SessionsListView()
    }
}
