# Object Recorder

Very simple iOS app for collecting object-capture data (video + camera
calibration + depth) on iPhone. Built for iPhone 15 Pro Max (LiDAR) but
degrades gracefully on non-LiDAR devices.

The app opens on a **sessions list** — every past capture session, newest
first. From there, name a new session (e.g. the object you're about to
capture — required, so sessions stay identifiable) and tap **New Session**
to jump into the camera screen;
record as many ARKit and/or Max FPS clips as you want (switching modes
freely between them), then tap **End Session** to return to the list. The
name is folded into the session's folder name and shown in the list so you
can tell sessions apart. Tapping a past session in the list opens a
read-only detail view of its recordings — sessions can't be resumed, only
started fresh. Swipe a session left to delete it (and its recordings) from
the phone.

Two capture modes, picked with the segmented control at the top:

- **ARKit** — video + per-frame intrinsics + extrinsics (full 6-DoF camera
  pose in world space) + LiDAR scene depth + depth confidence + light
  estimate + exposure/white-balance. Capped around 60 fps by ARKit.
- **Max FPS** — plain `AVCaptureSession` on the wide camera, running at the
  highest fps format the device supports (e.g. up to 240 fps slow-motion
  formats). Only per-frame intrinsics are recorded — depth and true
  extrinsics require a paired depth stream that isn't available at these
  frame rates, so this mode trades them away for raw fps.

## Build & run

Requires a physical device (camera/ARKit/LiDAR aren't available in the
simulator).

```bash
cd data_collector/object_recorder
xcodegen generate      # regenerates ObjectRecorder.xcodeproj from project.yml
open ObjectRecorder.xcodeproj
```

In Xcode, set your Development Team under Signing & Capabilities, select
your device, and run.

## Output format

**New Session** creates one timestamped session folder (prefixed with the
name you gave it) in the app's Documents directory (visible via the Files
app → On My iPhone → Object Recorder, or `xcrun devicectl` / Xcode's
Devices window); every recording made before the matching **End Session**
lands in its own numbered subfolder inside it:

```
session_<name>_<yyyyMMdd_HHmmss>/
  session_manifest.json          # list of this session's recordings (name, mode)
  01_arkit_<yyyyMMdd_HHmmss>/
    video.mov         # HEVC (ARKit mode) or H.264 (Max FPS mode)
    manifest.json      # mode, resolution, fps, depth/extrinsics availability
    frames.jsonl        # one JSON object per captured frame (see below)
    raw/                # binary depth/confidence dumps referenced from frames.jsonl
  02_maxfps_<yyyyMMdd_HHmmss>/
    video.mov
    manifest.json
    frames.jsonl
```

`frames.jsonl` — one line per frame, tightly packed row-major matrices:

- `frame_index`, `timestamp`
- `intrinsics`: 3x3 camera intrinsic matrix
- ARKit mode only:
  - `extrinsics`: 4x4 camera transform (pose) in world space
  - `depth_file` / `confidence_file`: paths into `raw/`, Float32 meters /
    UInt8 (0/1/2 = low/medium/high) respectively, tightly packed
    row-major, dimensions given by `depth_width`/`depth_height` in
    `manifest.json`
  - `light_estimate`: ambient intensity/color temperature, plus (on LiDAR
    devices) primary light direction/intensity and the 9x3 spherical
    harmonics coefficients
  - `exposure`: ISO, exposure duration, lens position, white balance gains
    — read live off the physical camera device while ARKit owns the
    capture session

## Getting sessions off the phone

The app enables `UIFileSharingEnabled` + `LSSupportsOpeningDocumentsInPlace`,
so its Documents folder is exposed without any extra app code:

- **Files app / Google Drive**: Files app → *On My iPhone → Object
  Recorder* → select a `session_...` folder → Share or Move → if the
  Google Drive app is installed it appears as a destination.
- **USB to Mac**: plug in the phone, open Finder → select the device →
  **Files** tab → drag session folders out of "Object Recorder" directly.
  This is the more reliable path for large raw depth binaries.

## Notes

- The live camera feed starts as soon as camera access is granted (not just
  once you hit record), so you can frame the shot before starting a
  recording; switching the mode picker between ARKit/Max FPS swaps which
  camera session is active. No preview overlay beyond the raw feed — kept
  intentionally minimal.
- Recording state (mode, start/stop, active session, typed-in name) resets
  between app runs; nothing is persisted across launches except the written
  session folders. The sessions list is rebuilt from those folders (and
  their `session_manifest.json`) each time it appears, so it's always in
  sync with what's actually on disk.
