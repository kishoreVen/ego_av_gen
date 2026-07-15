import simd
import CoreMedia

func matrix3x3ToArray(_ m: simd_float3x3) -> [[Float]] {
    [
        [m.columns.0.x, m.columns.1.x, m.columns.2.x],
        [m.columns.0.y, m.columns.1.y, m.columns.2.y],
        [m.columns.0.z, m.columns.1.z, m.columns.2.z]
    ]
}

func matrix4x4ToArray(_ m: simd_float4x4) -> [[Float]] {
    [
        [m.columns.0.x, m.columns.1.x, m.columns.2.x, m.columns.3.x],
        [m.columns.0.y, m.columns.1.y, m.columns.2.y, m.columns.3.y],
        [m.columns.0.z, m.columns.1.z, m.columns.2.z, m.columns.3.z],
        [m.columns.0.w, m.columns.1.w, m.columns.2.w, m.columns.3.w]
    ]
}

/// Reads the per-frame intrinsic matrix attached to a video sample buffer when
/// `isCameraIntrinsicMatrixDeliveryEnabled` is set on the capture connection.
func cameraIntrinsics(from sampleBuffer: CMSampleBuffer) -> [[Float]]? {
    guard let attachment = CMGetAttachment(
        sampleBuffer,
        key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix,
        attachmentModeOut: nil
    ) as? Data else { return nil }

    let matrix = attachment.withUnsafeBytes { $0.load(as: matrix_float3x3.self) }
    return matrix3x3ToArray(matrix)
}
