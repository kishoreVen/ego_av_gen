"""
Generate a self-contained interactive HTML viewer for the DROID dataset.

Usage:
    python -m brain_factory.dataset.viewer.generate \
        --data-dir data/droid_100 \
        --split train \
        --max-episodes 3 \
        --output brain_factory/dataset/viewer/droid_viewer.html

Opens the output .html file directly in any browser — no server needed.
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ── TFRecord feature layout (same schema as droid_dataset.py) ───────────────

_FLOAT_FIELDS: Dict[str, int] = {
    "steps/action":                          7,
    "steps/observation/joint_position":      7,
    "steps/observation/cartesian_position":  6,
    "steps/observation/gripper_position":    1,
}
_IMAGE_KEYS = [
    "steps/observation/wrist_image_left",
    "steps/observation/exterior_image_1_left",
    "steps/observation/exterior_image_2_left",
]
_INT64_KEYS  = ["steps/is_first", "steps/is_last"]
_STRING_KEYS = ["steps/language_instruction"]

ACTION_LABELS    = ["jv0", "jv1", "jv2", "jv3", "jv4", "jv5", "grip"]
JOINT_POS_LABELS = ["jp0", "jp1", "jp2", "jp3", "jp4", "jp5", "jp6"]
CART_POS_LABELS  = ["cx",  "cy",  "cz",  "cr",  "cp",  "cyw"]


# ── Data extraction ──────────────────────────────────────────────────────────

def _resolve_shard_dir(data_dir: str) -> Path:
    p = Path(data_dir)
    if (p / "dataset_info.json").exists():
        return p
    candidates = sorted(p.glob("*/dataset_info.json"))
    if not candidates:
        raise FileNotFoundError(f"No dataset_info.json found under {data_dir}")
    return candidates[0].parent


def _thumb_b64(jpeg_bytes: bytes, size: Tuple[int, int], quality: int) -> str:
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB").resize(size, Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load_episodes(
    data_dir: str,
    split: str = "train",
    max_episodes: int = 3,
    max_steps: Optional[int] = None,
    thumb_size: Tuple[int, int] = (320, 180),
    thumb_quality: int = 60,
) -> List[Dict[str, Any]]:
    import tensorflow as tf

    shard_dir = _resolve_shard_dir(data_dir)
    shards = sorted(glob.glob(str(shard_dir / f"*-{split}.tfrecord-*")))
    if not shards:
        raise FileNotFoundError(f"No shards for split '{split}' under {shard_dir}")

    spec: Dict[str, Any] = {
        "episode_metadata/file_path": tf.io.FixedLenFeature([], tf.string, ""),
    }
    for key in _FLOAT_FIELDS:
        spec[key] = tf.io.VarLenFeature(tf.float32)
    for key in _IMAGE_KEYS + _STRING_KEYS:
        spec[key] = tf.io.VarLenFeature(tf.string)
    for key in _INT64_KEYS:
        spec[key] = tf.io.VarLenFeature(tf.int64)

    episodes: List[Dict[str, Any]] = []
    for ep_idx, raw in enumerate(tf.data.TFRecordDataset(shards)):
        if ep_idx >= max_episodes:
            break
        p = tf.io.parse_single_example(raw, spec)
        n = int(tf.sparse.to_dense(p["steps/is_first"]).shape[0])
        if max_steps is not None:
            n = min(n, max_steps)

        # floats: (n_steps, dim)
        floats: Dict[str, list] = {}
        for key, dim in _FLOAT_FIELDS.items():
            arr = tf.sparse.to_dense(p[key]).numpy().reshape(-1, dim)[:n]
            floats[key] = arr.tolist()

        # int64 flags
        ints: Dict[str, list] = {}
        for key in _INT64_KEYS:
            ints[key] = tf.sparse.to_dense(p[key]).numpy()[:n].tolist()

        # strings
        lang = [s.decode("utf-8") for s in tf.sparse.to_dense(p["steps/language_instruction"]).numpy()[:n]]

        # images → thumbnails
        imgs: Dict[str, List[str]] = {k: [] for k in ("wrist", "ext1", "ext2")}
        for key, alias in zip(_IMAGE_KEYS, ("wrist", "ext1", "ext2")):
            jpegs = tf.sparse.to_dense(p[key]).numpy()[:n]
            for jpeg in jpegs:
                imgs[alias].append(_thumb_b64(jpeg, thumb_size, thumb_quality))

        file_path = p["episode_metadata/file_path"].numpy().decode("utf-8")
        episodes.append({
            "ep_idx":    ep_idx,
            "file_path": file_path,
            "n_steps":   n,
            "wrist":     imgs["wrist"],
            "ext1":      imgs["ext1"],
            "ext2":      imgs["ext2"],
            "action":    floats["steps/action"],
            "joint_pos": floats["steps/observation/joint_position"],
            "cart_pos":  floats["steps/observation/cartesian_position"],
            "gripper":   floats["steps/observation/gripper_position"],
            "is_first":  ints["steps/is_first"],
            "is_last":   ints["steps/is_last"],
            "language":  lang,
        })
        pct = (ep_idx + 1) / max_episodes * 100
        print(f"  [{pct:3.0f}%] episode {ep_idx}: {n} steps — {Path(file_path).name}", file=sys.stderr)

    return episodes


# ── HTML generation ───────────────────────────────────────────────────────────

_PALETTE_7 = [
    "#ff6b6b", "#ffa94d", "#ffd43b", "#69db7c", "#4dabf7", "#da77f2", "#f783ac",
]
_PALETTE_6 = _PALETTE_7[:6]

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DROID Dataset Viewer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:      #0f0f13;
    --surface: #1a1a22;
    --border:  #2e2e3a;
    --text:    #e0e0ee;
    --muted:   #6a6a88;
    --accent:  #7c6aff;
    --tag-bg:  #252530;
  }
  body { background: var(--bg); color: var(--text); font-family: ui-monospace, monospace; font-size: 13px; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

  /* ── header ── */
  header { display: flex; align-items: center; gap: 16px; padding: 10px 16px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
  header h1 { font-size: 14px; font-weight: 600; letter-spacing: .04em; color: var(--accent); }
  .ep-tabs { display: flex; gap: 6px; flex-wrap: wrap; }
  .ep-tab { padding: 3px 12px; border-radius: 4px; border: 1px solid var(--border); background: var(--surface); cursor: pointer; color: var(--muted); transition: .15s; }
  .ep-tab.active { border-color: var(--accent); color: var(--text); background: var(--tag-bg); }
  .ep-tab:hover:not(.active) { border-color: var(--muted); color: var(--text); }

  /* ── step bar ── */
  .step-bar { display: flex; align-items: center; gap: 10px; padding: 6px 16px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
  .step-bar label { color: var(--muted); white-space: nowrap; }
  #step-slider { flex: 1; accent-color: var(--accent); cursor: pointer; }
  #step-counter { min-width: 72px; text-align: right; }
  button.play-btn { padding: 3px 14px; border-radius: 4px; border: 1px solid var(--border); background: var(--surface); color: var(--text); cursor: pointer; transition: .15s; }
  button.play-btn:hover { border-color: var(--accent); }
  button.play-btn.playing { border-color: var(--accent); color: var(--accent); }

  /* ── main body ── */
  .main { flex: 1; display: grid; grid-template-columns: 1fr auto; min-height: 0; overflow: hidden; }

  /* ── left pane: cameras + charts ── */
  .left-pane { display: flex; flex-direction: column; gap: 8px; padding: 10px 10px 10px 16px; overflow-y: auto; min-height: 0; }
  .cameras { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
  .cam-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
  .cam-card figcaption { padding: 4px 8px; font-size: 11px; color: var(--muted); border-bottom: 1px solid var(--border); }
  .cam-card img { width: 100%; display: block; aspect-ratio: 16/9; object-fit: cover; }
  .chart-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px 8px; }
  .chart-wrap h3 { font-size: 11px; color: var(--muted); margin-bottom: 6px; letter-spacing: .06em; text-transform: uppercase; }
  .chart-wrap canvas { width: 100% !important; height: 120px !important; }

  /* ── right pane: info ── */
  .info-pane { width: 240px; border-left: 1px solid var(--border); padding: 12px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }
  .info-section h4 { font-size: 10px; color: var(--muted); letter-spacing: .1em; text-transform: uppercase; margin-bottom: 6px; }
  .info-row { display: flex; justify-content: space-between; gap: 4px; margin-bottom: 3px; line-height: 1.6; }
  .info-row .key { color: var(--muted); }
  .info-row .val { text-align: right; word-break: break-all; color: var(--text); }
  .tag { display: inline-block; padding: 1px 7px; border-radius: 3px; font-size: 11px; background: var(--tag-bg); border: 1px solid var(--border); }
  .tag.on  { border-color: #69db7c; color: #69db7c; }
  .tag.off { color: var(--muted); }
  #lang-box { font-size: 12px; color: var(--text); line-height: 1.5; padding: 6px 8px; background: var(--tag-bg); border-radius: 4px; border: 1px solid var(--border); }
  .vec-grid { display: grid; grid-template-columns: auto 1fr; gap: 2px 8px; }
  .vec-label { color: var(--muted); }
  .vec-val   { font-variant-numeric: tabular-nums; }

  scrollbar-width: thin;
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>

<header>
  <h1>DROID Viewer</h1>
  <div class="ep-tabs" id="ep-tabs"></div>
</header>

<div class="step-bar">
  <label>Step</label>
  <input type="range" id="step-slider" min="0" value="0">
  <span id="step-counter">0 / 0</span>
  <button class="play-btn" id="play-btn">&#9654; Play</button>
</div>

<div class="main">
  <div class="left-pane">
    <div class="cameras">
      <figure class="cam-card"><figcaption>Wrist</figcaption><img id="img-wrist" alt="wrist"></figure>
      <figure class="cam-card"><figcaption>Exterior 1</figcaption><img id="img-ext1" alt="ext1"></figure>
      <figure class="cam-card"><figcaption>Exterior 2</figcaption><img id="img-ext2" alt="ext2"></figure>
    </div>
    <div class="chart-wrap">
      <h3>Action (joint velocities + gripper)</h3>
      <canvas id="chart-action"></canvas>
    </div>
    <div class="chart-wrap">
      <h3>Joint positions</h3>
      <canvas id="chart-joint"></canvas>
    </div>
    <div class="chart-wrap">
      <h3>Cartesian position</h3>
      <canvas id="chart-cart"></canvas>
    </div>
  </div>

  <div class="info-pane">
    <div class="info-section">
      <h4>Instruction</h4>
      <div id="lang-box">—</div>
    </div>
    <div class="info-section">
      <h4>Step</h4>
      <div class="info-row"><span class="key">index</span><span class="val" id="info-step">—</span></div>
      <div class="info-row"><span class="key">is_first</span><span class="val" id="info-first">—</span></div>
      <div class="info-row"><span class="key">is_last</span><span class="val" id="info-last">—</span></div>
    </div>
    <div class="info-section">
      <h4>Action</h4>
      <div class="vec-grid" id="vec-action"></div>
    </div>
    <div class="info-section">
      <h4>Joint position</h4>
      <div class="vec-grid" id="vec-joint"></div>
    </div>
    <div class="info-section">
      <h4>Cartesian position</h4>
      <div class="vec-grid" id="vec-cart"></div>
    </div>
    <div class="info-section">
      <h4>Gripper</h4>
      <div class="info-row"><span class="key">position</span><span class="val" id="info-grip">—</span></div>
    </div>
    <div class="info-section">
      <h4>Episode</h4>
      <div class="info-row"><span class="key">steps</span><span class="val" id="info-nsteps">—</span></div>
      <div style="margin-top:4px; color: var(--muted); font-size:11px; word-break:break-all;" id="info-path">—</div>
    </div>
  </div>
</div>

<script>
const EPISODES = __EPISODES_JSON__;
const ACTION_LABELS    = __ACTION_LABELS__;
const JOINT_POS_LABELS = __JOINT_POS_LABELS__;
const CART_POS_LABELS  = __CART_POS_LABELS__;
const PALETTE_7 = __PALETTE_7__;
const PALETTE_6 = __PALETTE_6__;

// ── state ────────────────────────────────────────────────────────────────────
let currentEp   = 0;
let currentStep = 0;
let playTimer   = null;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const slider    = document.getElementById("step-slider");
const counter   = document.getElementById("step-counter");
const playBtn   = document.getElementById("play-btn");
const imgWrist  = document.getElementById("img-wrist");
const imgExt1   = document.getElementById("img-ext1");
const imgExt2   = document.getElementById("img-ext2");

// ── chart helpers ─────────────────────────────────────────────────────────────
const cursorPlugin = {
  id: "cursor",
  afterDatasetsDraw(chart) {
    const { ctx, chartArea: ca, scales: { x } } = chart;
    const px = x.getPixelForValue(currentStep);
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(px, ca.top);
    ctx.lineTo(px, ca.bottom);
    ctx.strokeStyle = "rgba(255,255,255,0.45)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();
  }
};
Chart.register(cursorPlugin);

const BASE_CHART_OPTS = (labels) => ({
  type: "line",
  data: { labels, datasets: [] },
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { labels: { color: "#6a6a88", boxWidth: 10, font: { size: 10 } } },
      tooltip: { enabled: true }
    },
    scales: {
      x: {
        type: "linear",
        ticks: { color: "#6a6a88", maxTicksLimit: 8, font: { size: 10 } },
        grid: { color: "#2e2e3a" }
      },
      y: {
        ticks: { color: "#6a6a88", maxTicksLimit: 5, font: { size: 10 } },
        grid: { color: "#2e2e3a" }
      }
    }
  }
});

function makeChart(canvasId, labels) {
  const ctx = document.getElementById(canvasId).getContext("2d");
  return new Chart(ctx, BASE_CHART_OPTS(Array.from({ length: 1 }, (_, i) => i)));
}

const chartAction = makeChart("chart-action");
const chartJoint  = makeChart("chart-joint");
const chartCart   = makeChart("chart-cart");

function buildDatasets(matrix, labels, palette) {
  return labels.map((lbl, i) => ({
    label: lbl,
    data: matrix.map((row, t) => ({ x: t, y: row[i] })),
    borderColor: palette[i],
    borderWidth: 1.5,
    pointRadius: 0,
    tension: 0.2
  }));
}

function loadEpisode(epIdx) {
  const ep = EPISODES[epIdx];
  currentEp   = epIdx;
  currentStep = 0;
  slider.max  = ep.n_steps - 1;
  slider.value = 0;

  // charts
  chartAction.data.datasets = buildDatasets(ep.action,    ACTION_LABELS,    PALETTE_7);
  chartJoint.data.datasets  = buildDatasets(ep.joint_pos, JOINT_POS_LABELS, PALETTE_7);
  chartCart.data.datasets   = buildDatasets(ep.cart_pos,  CART_POS_LABELS,  PALETTE_6);
  chartAction.update("none");
  chartJoint.update("none");
  chartCart.update("none");

  // episode info
  document.getElementById("info-nsteps").textContent = ep.n_steps;
  document.getElementById("info-path").textContent   = ep.file_path.split("/").slice(-3).join("/");

  updateStep(0);
  highlightTab(epIdx);
}

function updateStep(step) {
  const ep = EPISODES[currentEp];
  currentStep  = step;
  slider.value = step;
  counter.textContent = `${step} / ${ep.n_steps - 1}`;

  // images
  imgWrist.src = "data:image/jpeg;base64," + ep.wrist[step];
  imgExt1.src  = "data:image/jpeg;base64," + ep.ext1[step];
  imgExt2.src  = "data:image/jpeg;base64," + ep.ext2[step];

  // info panel
  document.getElementById("lang-box").textContent    = ep.language[step] || "—";
  document.getElementById("info-step").textContent   = step;
  document.getElementById("info-first").innerHTML    = flag(ep.is_first[step]);
  document.getElementById("info-last").innerHTML     = flag(ep.is_last[step]);
  document.getElementById("info-grip").textContent   = fmt(ep.gripper[step][0]);

  renderVec("vec-action", ep.action[step],    ACTION_LABELS,    PALETTE_7);
  renderVec("vec-joint",  ep.joint_pos[step], JOINT_POS_LABELS, PALETTE_7);
  renderVec("vec-cart",   ep.cart_pos[step],  CART_POS_LABELS,  PALETTE_6);

  // redraw cursor
  chartAction.update("none");
  chartJoint.update("none");
  chartCart.update("none");
}

function flag(val) {
  const on = Boolean(val);
  return `<span class="tag ${on ? "on" : "off"}">${on ? "true" : "false"}</span>`;
}

function fmt(v) { return typeof v === "number" ? v.toFixed(4) : String(v); }

function renderVec(divId, row, labels, palette) {
  const div = document.getElementById(divId);
  div.innerHTML = labels.map((lbl, i) =>
    `<span class="vec-label" style="color:${palette[i]}">${lbl}</span>` +
    `<span class="vec-val">${fmt(row[i])}</span>`
  ).join("");
}

// ── episode tabs ──────────────────────────────────────────────────────────────
const tabsEl = document.getElementById("ep-tabs");
EPISODES.forEach((ep, i) => {
  const btn = document.createElement("button");
  btn.className = "ep-tab";
  btn.textContent = `ep ${i}`;
  btn.title = ep.file_path;
  btn.onclick = () => { stopPlay(); loadEpisode(i); };
  tabsEl.appendChild(btn);
});

function highlightTab(idx) {
  tabsEl.querySelectorAll(".ep-tab").forEach((b, i) => b.classList.toggle("active", i === idx));
}

// ── slider ────────────────────────────────────────────────────────────────────
slider.addEventListener("input", () => { stopPlay(); updateStep(parseInt(slider.value, 10)); });

// ── play / pause ──────────────────────────────────────────────────────────────
function startPlay() {
  if (playTimer) return;
  playBtn.textContent = "⏸ Pause";
  playBtn.classList.add("playing");
  playTimer = setInterval(() => {
    const ep = EPISODES[currentEp];
    const next = currentStep + 1;
    if (next >= ep.n_steps) { stopPlay(); return; }
    updateStep(next);
  }, 80);
}
function stopPlay() {
  if (!playTimer) return;
  clearInterval(playTimer);
  playTimer = null;
  playBtn.textContent = "▶ Play";
  playBtn.classList.remove("playing");
}
playBtn.onclick = () => playTimer ? stopPlay() : startPlay();

// ── keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener("keydown", e => {
  const ep = EPISODES[currentEp];
  if (e.key === "ArrowRight") { stopPlay(); updateStep(Math.min(currentStep + 1, ep.n_steps - 1)); }
  if (e.key === "ArrowLeft")  { stopPlay(); updateStep(Math.max(currentStep - 1, 0)); }
  if (e.key === " ") { e.preventDefault(); playTimer ? stopPlay() : startPlay(); }
});

// ── init ──────────────────────────────────────────────────────────────────────
if (EPISODES.length > 0) loadEpisode(0);
</script>
</body>
</html>
"""


def generate_html(episodes: List[Dict[str, Any]]) -> str:
    subs = {
        "__EPISODES_JSON__":    json.dumps(episodes, separators=(",", ":")),
        "__ACTION_LABELS__":    json.dumps(ACTION_LABELS),
        "__JOINT_POS_LABELS__": json.dumps(JOINT_POS_LABELS),
        "__CART_POS_LABELS__":  json.dumps(CART_POS_LABELS),
        "__PALETTE_7__":        json.dumps(_PALETTE_7),
        "__PALETTE_6__":        json.dumps(_PALETTE_6),
    }
    html = _HTML_TEMPLATE
    for key, val in subs.items():
        html = html.replace(key, val)
    return html


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir",      default="data/droid_100")
    parser.add_argument("--split",         default="train")
    parser.add_argument("--max-episodes",  type=int, default=3)
    parser.add_argument("--max-steps",     type=int, default=None, help="Cap steps per episode (None = all)")
    parser.add_argument("--thumb-width",   type=int, default=320)
    parser.add_argument("--thumb-height",  type=int, default=180)
    parser.add_argument("--thumb-quality", type=int, default=60)
    parser.add_argument("--output",        default="brain_factory/dataset/viewer/droid_viewer.html")
    args = parser.parse_args()

    print(f"Loading {args.max_episodes} episode(s) from {args.data_dir} …", file=sys.stderr)
    episodes = load_episodes(
        data_dir     = args.data_dir,
        split        = args.split,
        max_episodes = args.max_episodes,
        max_steps    = args.max_steps,
        thumb_size   = (args.thumb_width, args.thumb_height),
        thumb_quality= args.thumb_quality,
    )
    html = generate_html(episodes)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    size_mb = out.stat().st_size / 1_048_576
    print(f"Wrote {out}  ({size_mb:.1f} MB)", file=sys.stderr)
    print(f"Open in browser: open {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
