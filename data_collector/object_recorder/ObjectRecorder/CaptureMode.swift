import Foundation

enum CaptureMode: String, CaseIterable, Identifiable {
    case arKit = "ARKit"
    case maxFPS = "Max FPS"

    var id: String { rawValue }

    var subtitle: String {
        switch self {
        case .arKit:
            return "Pose + intrinsics + LiDAR depth (~60 fps)"
        case .maxFPS:
            return "Highest fps, intrinsics only, no depth"
        }
    }
}
