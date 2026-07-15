# iOS Data Viewer

Python loader + browser-based 3D viewer for sessions recorded by the
[Object Recorder](../../data_collector/object_recorder) iOS app.

## Pulling a session from the phone

```bash
xcrun devicectl device copy from \
  --device <device-id> \
  --domain-type appDataContainer \
  --domain-identifier com.datacollector.objectrecorder \
  --source "/Documents" \
  --destination <local-folder>
```

(`xcrun devicectl list devices` to find `<device-id>`.) This pulls every
`session_<mode>_<timestamp>/` folder from the app's Documents directory.

## Python loader

`loader.py` reads a session folder into numpy-friendly structures:

```python
from loader import Session

session = Session("session_arkit_20260714_213402")
for frame in session.frames:
    frame.intrinsics   # 3x3 np.ndarray or None
    frame.extrinsics   # 4x4 camera-to-world np.ndarray, ARKit mode only
    depth = session.depth_map(frame)        # (H, W) float32 meters, or None
    confidence = session.confidence_map(frame)  # (H, W) uint8 0/1/2, or None

for bgr_frame in session.video_frames():    # decoded video.mov, in frame order
    ...
```

`geometry.py` has `unproject_depth(...)`, which turns a depth map + its
frame's intrinsics/extrinsics into world-space points, following ARKit's
coordinate convention (camera-local +Z toward the viewer, i.e. forward is
-Z; `extrinsics` is `ARFrame.camera.transform`, camera-to-world).

## Browser viewer

```bash
python export_viewer_data.py /path/to/session_arkit_20260714_213402
python serve.py
# open http://127.0.0.1:8765/
```

`export_viewer_data.py` writes everything the browser needs into
`viewer/data/<session_name>/`:
- **video.mp4** — video.mov transcoded to H.264 (requires `ffmpeg` on
  PATH; falls back to a verbatim copy with a warning if missing). The
  app records HEVC in ARKit mode, which Chromium-based browsers without
  licensed codec support cannot decode — H.264 plays everywhere.
- **depth/depth_\<frame_index\>.png** — a colorized heatmap per depth
  frame (red/warm = near, blue/cool = far), for every frame, not just a
  sampled subset, so frame-by-frame scrubbing is smooth.
- **trajectory.json** — one entry per posed frame: position, rotation,
  timestamp (seconds relative to video start), and the matching
  depth_image path when available.
- **positions.f32 / colors.u8** — a downsampled point cloud, unprojected
  from depth and colored by sampling the *nearest-by-timestamp* video
  frame (not same-index — `frames.jsonl` entries and encoded video
  frames aren't guaranteed 1:1 if the encoder ever falls behind).

Re-running it for a different session just adds another entry to the
dropdown in the viewer (tracked in `viewer/data/index.json`) — nothing is
overwritten across sessions.

Useful flags:
- `--max-points` (default 400k) — hard cap, randomly subsampled beyond it
- `--frame-stride` (default 3) — use every Nth depth frame for the point cloud
- `--pixel-stride` (default 2) — sample every Nth pixel within each depth map
- `--min-confidence` (default 2) — drop low/medium-confidence LiDAR samples
- `--max-depth` (default 3.0m) — drop far-range, noisier depth samples; also
  the normalization range for the depth heatmap colors

The viewer (`viewer/index.html` + `viewer/app.js`) is plain three.js,
vendored locally under `viewer/vendor/`, with two tabs:

- **3D Reconstruction** — the point cloud, camera path (red line), and
  camera frustums (blue); orbit/zoom with the mouse, toggle
  trajectory/frustums, adjust point size.
- **Video & Depth** — the actual video, playable/scrubbable, with the
  colorized depth heatmap for the nearest frame updating live as you
  play or seek.

It needs a local HTTP server (`serve.py`) both because `fetch()` of local
files is blocked under `file://`, and because `<video>` playback needs
HTTP range-request support to seek — `serve.py` implements that (Python's
stock `http.server` doesn't). Runs fully offline otherwise.

Max-FPS mode sessions have no depth/extrinsics (see the app's README), so
the 3D tab and depth heatmap will be empty for those — only the video tab
is meaningful.

**Verifying the depth colors**: `_colorize_depth` in `export_viewer_data.py`
is unit-testable in isolation (feed it a synthetic depth array) if the
near/far color mapping ever looks wrong again — a real desk scene often
has surfaces at surprising relative distances (e.g. a nearby wall can be
closer than a foreground object at a steep camera angle), which can look
like a bug when it isn't.
