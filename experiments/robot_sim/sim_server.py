"""
Nero arm + gripper MuJoCo simulation server.

Streams rendered frames as MJPEG so any browser or <img> tag can display them.
Also exposes a minimal JSON API for reading/writing joint targets.

Usage:
    MUJOCO_GL=egl python experiments/robot_sim/sim_server.py

Then open http://localhost:5202 in your browser (or use brain_tv SimViewer tab).

Endpoints:
    GET  /          - standalone HTML viewer
    GET  /stream    - multipart/x-mixed-replace MJPEG stream
    GET  /state     - JSON: {t, qpos, ctrl, camera}
    POST /ctrl      - JSON body {j1..j7, gripper}  (all optional, rad/m)
    POST /camera    - JSON body {name: "front"|"side"|"iso"}
    POST /keyframe  - JSON body {name: "home"|"ready"}
"""

import os
import sys
import json
import time
import math
import argparse
import threading
import io
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import numpy as np
import cv2
import mujoco

# ── CLI args ─────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--scene", default=None, help="Path to MJCF .xml scene file")
_parser.add_argument("--port",  type=int, default=5202)
_args, _ = _parser.parse_known_args()

# ── Config ──────────────────────────────────────────────────────────────────
_default_scene = os.path.join(os.path.dirname(__file__),
                               "../../robots/scenes/nero_sim.xml")
SCENE_XML   = os.path.abspath(_args.scene or _default_scene)
PORT        = _args.port
WIDTH       = 720
HEIGHT      = 480
SIM_DT      = 0.002   # seconds per physics step
RENDER_HZ   = 30      # target rendered frames per second
DEMO_PERIOD = 8.0     # seconds per demo cycle

JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper_act"]
CAMERAS     = ["front", "side", "iso"]

# ── Shared state (guarded by _lock) ─────────────────────────────────────────
_lock          = threading.Lock()
_latest_jpg    = b""
_current_ctrl  = np.zeros(8)
_current_qpos  = np.zeros(10)
_sim_time      = 0.0
_camera_name   = "iso"
_demo_mode     = True   # autonomous sinusoidal demo until user sends /ctrl


def _demo_ctrl(t: float) -> np.ndarray:
    """Sinusoidal demo trajectory keeping the arm above the floor.

    With eulerseq=XYZ kinematics:
      j1 (yaw):   safe full range ±2.7
      j2 (shoulder): 0→1.3  (0=up, >0=reach out; negative goes underground)
      j3 (elbow):    0→1.0  (positive bends safely)
      j4-j7:         small oscillations around zero
      gripper:       slow open/close (0=closed, 0.1=fully open)
    """
    T  = 2 * math.pi * t / DEMO_PERIOD
    ctrl = np.zeros(8)
    # Base yaw sweep ±90°
    ctrl[0] =  0.9  * math.sin(T)
    # Shoulder: from 0.3 (nearly upright) to 1.3 (reaching out), never negative
    ctrl[1] =  0.8  + 0.5 * math.sin(T + 0.7)     # range [0.3 .. 1.3]
    # Elbow: 0 to 0.9 oscillation
    ctrl[2] =  0.45 + 0.45 * math.sin(T * 0.8 + 1.2)
    # Forearm roll
    ctrl[3] =  0.4  * math.sin(T * 1.1 + 2.0)
    # Wrist pitch
    ctrl[4] =  0.5  * math.sin(T * 0.9 + 3.0)
    # Wrist roll
    ctrl[5] =  0.3  * math.sin(T * 1.3 + 0.5)
    # Wrist yaw
    ctrl[6] =  0.5  * math.sin(T * 0.7 + 1.8)
    # Gripper open/close at a slower pace (0=closed, 0.1=open)
    ctrl[7] =  0.05 * (1.0 + math.sin(T * 0.6))
    return ctrl


def _sim_thread():
    global _latest_jpg, _current_ctrl, _current_qpos, _sim_time

    model    = mujoco.MjModel.from_xml_path(os.path.abspath(SCENE_XML))
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    mujoco.mj_resetDataKeyframe(model, data, 0)  # start from "home" (stable upright)
    mujoco.mj_forward(model, data)
    # Warmup: let the position controllers settle before the render loop begins.
    # Avoids transient NaN in QACC during the first few steps.
    for _ in range(200):
        data.ctrl[:] = np.zeros(8)
        mujoco.mj_step(model, data)
    data.time = 0.0

    frame_interval = 1.0 / RENDER_HZ
    last_render    = 0.0
    sim_time       = data.time

    while True:
        loop_start = time.monotonic()

        # Read desired control + camera
        with _lock:
            cam    = _camera_name
            demo   = _demo_mode
            target = _current_ctrl.copy()

        if demo:
            target = _demo_ctrl(sim_time)

        data.ctrl[:] = target

        # Advance physics (a few steps per render frame for speed)
        steps = max(1, int(frame_interval / SIM_DT))
        for _ in range(steps):
            mujoco.mj_step(model, data)
        sim_time = data.time

        # Render frame
        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()

        # Encode JPEG
        ok, buf = cv2.imencode(".jpg", pixels[:, :, ::-1],
                               [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            with _lock:
                _latest_jpg   = buf.tobytes()
                _current_qpos = data.qpos.copy()
                _sim_time     = sim_time
                if demo:
                    _current_ctrl = target.copy()

        # Pace to RENDER_HZ
        elapsed = time.monotonic() - loop_start
        wait    = frame_interval - elapsed
        if wait > 0:
            time.sleep(wait)


# ── HTTP handler ─────────────────────────────────────────────────────────────
_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Nero Sim</title>
<style>
  body  {{ background:#0d1117; color:#e6edf3; font-family:monospace; margin:0; }}
  header {{ padding:12px 24px; background:#161b22; border-bottom:1px solid #30363d; }}
  h1    {{ font-size:16px; margin:0; }}
  main  {{ display:flex; flex-direction:column; align-items:center; padding:24px; gap:16px; }}
  img   {{ border:1px solid #30363d; border-radius:6px; max-width:100%; }}
  .controls {{ display:flex; gap:8px; flex-wrap:wrap; justify-content:center; }}
  button {{ padding:6px 14px; background:#21262d; color:#e6edf3;
            border:1px solid #30363d; border-radius:4px; cursor:pointer; font-family:monospace; }}
  button:hover {{ background:#30363d; }}
  .cam-active {{ border-color:#58a6ff !important; color:#58a6ff; }}
</style>
</head>
<body>
<header><h1>Nero Arm &amp; Gripper — MuJoCo Simulation</h1></header>
<main>
  <img src="/stream" alt="sim stream"/>
  <div class="controls">
    <button onclick="setCamera('front')" id="btn-front">Front</button>
    <button onclick="setCamera('side')"  id="btn-side" >Side</button>
    <button onclick="setCamera('iso')"   id="btn-iso" class="cam-active">Iso</button>
    <button onclick="setKeyframe('home')">Reset Home</button>
    <button onclick="setKeyframe('ready')">Ready Pose</button>
  </div>
  <div style="color:#8b949e;font-size:12px">Demo mode: sinusoidal joint motion · POST /ctrl to take over</div>
</main>
<script>
  function setCamera(name) {{
    fetch('/camera', {{method:'POST', headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{name}})
    }});
    document.querySelectorAll('[id^=btn-]').forEach(b => b.classList.remove('cam-active'));
    document.getElementById('btn-' + name).classList.add('cam-active');
  }}
  function setKeyframe(name) {{
    fetch('/keyframe', {{method:'POST', headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{name}})
    }});
  }}
</script>
</body>
</html>
"""


class SimHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence access logs
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path == "/stream":
            self._stream()
        elif self.path == "/state":
            self._state()
        else:
            self._index()

    def do_POST(self):
        length  = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/ctrl":
            self._set_ctrl(payload)
        elif self.path == "/camera":
            self._set_camera(payload)
        elif self.path == "/keyframe":
            self._set_keyframe(payload)
        else:
            self.send_response(404); self.end_headers()

    # ── Handlers ──────────────────────────────────────────────────────────
    def _index(self):
        body = _HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type",
                          "multipart/x-mixed-replace; boundary=frame")
        self._cors()
        self.end_headers()
        try:
            while True:
                with _lock:
                    frame = _latest_jpg
                if frame:
                    self.wfile.write(
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" +
                        frame + b"\r\n"
                    )
                    self.wfile.flush()
                time.sleep(1.0 / RENDER_HZ)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _state(self):
        with _lock:
            data = {
                "t":      round(_sim_time, 4),
                "qpos":   _current_qpos.tolist(),
                "ctrl":   _current_ctrl.tolist(),
                "camera": _camera_name,
                "demo":   _demo_mode,
            }
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _set_ctrl(self, payload):
        global _demo_mode, _current_ctrl
        ctrl = np.zeros(8)
        for i, name in enumerate(JOINT_NAMES):
            short = name.replace("_act", "")
            if name in payload:   ctrl[i] = float(payload[name])
            elif short in payload: ctrl[i] = float(payload[short])
        with _lock:
            _current_ctrl = ctrl
            _demo_mode    = False
        self.send_response(200)
        self._cors()
        self.end_headers()

    def _set_camera(self, payload):
        global _camera_name
        name = payload.get("name", "iso")
        if name in CAMERAS:
            with _lock:
                _camera_name = name
        self.send_response(200)
        self._cors()
        self.end_headers()

    def _set_keyframe(self, payload):
        # Reload model + reset to keyframe is done in sim thread on next cycle.
        # Simplest approach: just reset ctrl targets to keyframe values.
        global _demo_mode, _current_ctrl
        name = payload.get("name", "home")
        kf_ctrl = {
            "home":  np.zeros(8),
            "ready": np.array([0, 1.0, 0.5, 0, 0, 0, 0, 0.05]),
        }.get(name, np.zeros(8))
        with _lock:
            _current_ctrl = kf_ctrl
            _demo_mode    = False
        self.send_response(200)
        self._cors()
        self.end_headers()


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading scene: {os.path.abspath(SCENE_XML)}")
    # Warm up: verify model loads before starting server
    _test = mujoco.MjModel.from_xml_path(os.path.abspath(SCENE_XML))
    print(f"Model OK: nq={_test.nq} nu={_test.nu} nbody={_test.nbody}")
    del _test

    t = threading.Thread(target=_sim_thread, daemon=True)
    t.start()
    print(f"Sim thread started, waiting for first frame...")
    for _ in range(50):
        with _lock:
            if _latest_jpg:
                break
        time.sleep(0.1)

    server = ThreadingHTTPServer(("0.0.0.0", PORT), SimHandler)
    print(f"\nNero sim server running at http://localhost:{PORT}")
    print(f"  /         standalone HTML viewer")
    print(f"  /stream   MJPEG stream (embed in <img src=...>)")
    print(f"  /state    JSON state")
    print(f"  /ctrl     POST joint targets")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
