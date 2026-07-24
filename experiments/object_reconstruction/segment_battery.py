"""Run SAM3 text-grounded segmentation on aa_battery sessions.

Processes every video frame. Outputs per session:
  aa_battery/<session>/masks/frame_<N>.png       — binary mask (255=battery, 0=bg)
  aa_battery/<session>/masks_debug/frame_<N>.jpg — RGB overlay for inspection

Usage:
    python experiments/object_reconstruction/segment_battery.py
    python experiments/object_reconstruction/segment_battery.py --session 01_arkit
    python experiments/object_reconstruction/segment_battery.py --confidence 0.3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "ios_data_viewer"))
sys.path.insert(0, str(ROOT / "third_party" / "sam3"))

from loader import Session  # noqa: E402

DATA_DIR = ROOT / "data" / "object_reconstruction" / "aa_battery"
PROMPT = "a battery"


def make_debug_overlay(bgr: np.ndarray, masks: list[np.ndarray], scores: list[float]) -> np.ndarray:
    out = bgr.copy()
    colours = [(0, 255, 0), (0, 128, 255), (255, 0, 128), (255, 255, 0)]
    for i, (mask, score) in enumerate(zip(masks, scores)):
        colour = colours[i % len(colours)]
        overlay = np.zeros_like(bgr)
        overlay[mask] = colour
        out = cv2.addWeighted(out, 1.0, overlay, 0.45, 0)
        ys, xs = np.where(mask)
        if len(xs):
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(out, f"{score:.2f}", (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    return out


def merge_masks(masks: list[np.ndarray], h: int, w: int) -> np.ndarray:
    """Union all predicted masks into a single uint8 binary image (255=battery)."""
    if not masks:
        return np.zeros((h, w), dtype=np.uint8)
    combined = np.zeros((h, w), dtype=bool)
    for m in masks:
        combined |= m
    return (combined * 255).astype(np.uint8)


def segment_session(session_dir: Path, model) -> None:
    session = Session(session_dir)
    masks_dir = session_dir / "masks"
    debug_dir = session_dir / "masks_debug"
    masks_dir.mkdir(exist_ok=True)
    debug_dir.mkdir(exist_ok=True)

    # Count total frames first for progress display
    cap = cv2.VideoCapture(str(session.video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print(f"  {session_dir.name}: {total} video frames @ {fps:.1f}fps, "
          f"mode={session.mode}, {session.width}x{session.height}")

    debug_stride = max(1, total // 10)  # ~10 debug samples per session

    for frame_idx, (t, bgr) in enumerate(session.video_frames()):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = model.generate(image=Image.fromarray(rgb), prompt=PROMPT)
        masks = result["masks"]
        scores = result["scores"]

        h, w = bgr.shape[:2]
        name = f"frame_{frame_idx:06d}"
        binary = merge_masks(masks, h, w)
        cv2.imwrite(str(masks_dir / f"{name}.png"), binary)

        if frame_idx % debug_stride == 0:
            cv2.imwrite(str(debug_dir / f"{name}.jpg"), make_debug_overlay(bgr, masks, scores))

        best = f"{max(scores):.3f}" if scores else "none"
        print(f"    [{frame_idx+1}/{total}] {name} (t={t:.3f}s): score={best}", flush=True)

    print(f"  → {masks_dir}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default=None,
                        help="Run only this session folder (e.g. 01_arkit). Default: all.")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="SAM3 confidence threshold (default 0.3).")
    parser.add_argument("--device", default="cuda",
                        help="Torch device (default cuda).")
    args = parser.parse_args()

    from model_router.models.sam import SAMModel

    print(f"Loading SAM3 on {args.device}...")
    model = SAMModel(device=args.device, confidence_threshold=args.confidence)
    model.load()
    print(f"SAM3 ready. Prompt: '{PROMPT}'\n")

    if args.session:
        sessions = [DATA_DIR / args.session]
    else:
        sessions = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())

    for session_dir in sessions:
        segment_session(session_dir, model)

    print("Done.")


if __name__ == "__main__":
    main()
