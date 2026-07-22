# Dataset catalog

Every dataset here subclasses `DatasetBase` (`brain_factory/scaffold/dataset_base.py`)
and returns a dict per item — the "packet" a training loop sees out of a
`DataLoader`. This doc lists, per dataset, exactly what's in that dict.

## DummyDataset

`brain_factory/dataset/dummy_dataset.py` — random tensors, for exercising the
training framework end-to-end without real data.

| key | type/shape | notes |
|---|---|---|
| `input` | `float32` `(input_dim,)` | `torch.randn`, `input_dim` defaults to 16 |
| `target` | `int` (0 or 1) | `torch.randint(0, 2, (1,)).item()` |

## DroidDataset

`brain_factory/dataset/droid_dataset.py` — DROID (r2d2_faceblur) robot
episodes read from RLDS TFRecords, flattened to one item per step across all
loaded episodes.

| key | type/shape | notes |
|---|---|---|
| `wrist_image` | `float32` `(H, W, 3)`, `[0,1]` | decoded from JPEG, resized to `image_size` if set |
| `exterior_image_1` | `float32` `(H, W, 3)`, `[0,1]` | same |
| `exterior_image_2` | `float32` `(H, W, 3)`, `[0,1]` | same |
| `joint_position` | `float32` `(7,)` | |
| `cartesian_position` | `float32` `(6,)` | |
| `gripper_position` | `float32` `(1,)` | |
| `action` | `float32` `(7,)` | |
| `language_instruction` | `str` | decoded from the episode's bytes field |
| `is_first` | `bool` | |
| `is_last` | `bool` | |

## ObjectRecorderSessionDataset

`brain_factory/dataset/object_recorder_dataset.py` — one ObjectRecorder iOS
app capture session (`data_collector/object_recorder`), flattening frames
across all of the session's recordings (ARKit and/or Max FPS) into a single
indexable sequence.

A session is captures of one physical object, so every item carries
`object_name` identifying which one.

Keys present on every item:

| key | type/shape | notes |
|---|---|---|
| `image` | `float32` `(H, W, 3)`, `[0,1]` | decoded from `video.mov`, matched to this frame by presentation **timestamp** (not `frame_index` — video and metadata frames aren't guaranteed 1:1). Native resolution unless `image_size` is passed to the constructor. |
| `intrinsics` | `float32` `(3, 3)` | zero-filled if this frame didn't carry one (observed on some Max FPS frames) |
| `has_intrinsics` | `bool` | whether `intrinsics` above is real or zero-filled |
| `object_name` | `str` | the session's name (`label` in `session_manifest.json`, falling back to the session folder name for older sessions) — the object being captured, same for every item in the dataset instance. Overridable via the constructor's `object_name` param, so a training recipe config can set it explicitly (e.g. `object_name: aa_battery` under `dataset:`) rather than trusting whatever's on disk. |
| `mode` | `str` | `"arkit"` or `"maxfps"` |
| `recording_name` | `str` | e.g. `"01_arkit"` — which recording in the session this frame is from |
| `frame_index` | `int64` | frame's index within its recording |
| `timestamp` | `float64` | raw ARKit/AVFoundation uptime for this frame |
| `light_estimate` | `str` (JSON, `""` if absent) | ambient intensity/color temp, plus directional light + spherical harmonics on LiDAR devices |
| `exposure` | `str` (JSON, `""` if absent) | ISO, exposure duration, lens position, white-balance gains |

Keys included only if the dataset instance has at least one ARKit recording:

| key | type/shape | notes |
|---|---|---|
| `extrinsics` | `float32` `(4, 4)` | camera pose; zero-filled if absent |
| `has_extrinsics` | `bool` | |

Keys included only if the dataset instance has at least one LiDAR-depth
recording:

| key | type/shape | notes |
|---|---|---|
| `depth` | `float32` `(depth_h, depth_w)` | meters; zero-filled if absent |
| `has_depth` | `bool` | |
| `confidence` | `uint8` `(depth_h, depth_w)` | 0/1/2 = low/med/high; zero-filled if absent |
| `has_confidence` | `bool` | |

Notes:
- `light_estimate`/`exposure` are JSON strings rather than tensors/dicts so a
  mixed-mode batch doesn't break `DataLoader`'s default collate on the
  None-vs-dict split — `json.loads()` them when non-empty.
- ARKit and Max FPS recordings can be different native image shapes (e.g.
  landscape 1920x1440 vs portrait 1440x1920) — pass `image_size=(h, w)` to
  batch across mixed modes, or filter with `modes=["arkit"]` /
  `modes=["maxfps"]` to keep a batch's images natively uniform.
- Resizing via `image_size` does **not** rescale `intrinsics` to match —
  geometric math (projection, unprojection) against a resized image needs
  intrinsics scaled by the same factor; this isn't done automatically yet.
- Each `__getitem__` seeks into a compressed video, so it's not fast for
  large-scale training iteration — no frame-cache pass exists yet
  (`cached_get_item` currently just calls `pre_cache_get_item`, same as the
  other datasets above).
