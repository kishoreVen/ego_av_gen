import { useState, useEffect, useRef, useCallback } from 'react';

const SIM_BASE  = 'http://localhost:5202';
const API_BASE  = 'http://localhost:5199';
const CAMERAS   = ['front', 'side', 'iso'] as const;
type Camera     = typeof CAMERAS[number];

interface SimState {
  t: number;
  qpos: number[];
  ctrl: number[];
  camera: Camera;
  demo: boolean;
}

interface RobotDef {
  id: string;
  name: string;
  scene: string;
}

export function SimViewer() {
  const [status,      setStatus]      = useState<'starting' | 'running' | 'error'>('starting');
  const [state,       setState]       = useState<SimState | null>(null);
  const [camera,      setCamera]      = useState<Camera>('iso');
  const [showSliders, setShowSliders] = useState(false);
  const [ctrl,        setCtrl]        = useState<number[]>(Array(8).fill(0));
  const [launching,   setLaunching]   = useState(false);
  const [robots,      setRobots]      = useState<RobotDef[]>([]);
  const [selectedRobot, setSelectedRobot] = useState<string>('');
  const pollRef   = useRef<number | null>(null);
  const imgRef    = useRef<HTMLImageElement | null>(null);
  const imgKeyRef = useRef<number>(0);

  // Fetch available robots from Express API
  useEffect(() => {
    fetch(`${API_BASE}/api/sim/robots`)
      .then(r => r.json())
      .then((list: RobotDef[]) => {
        setRobots(list);
        if (list.length > 0) setSelectedRobot(list[0].id);
      })
      .catch(() => {});
  }, []);

  // Check if sim is reachable and start polling state
  useEffect(() => {
    let cancelled = false;

    async function checkStatus() {
      try {
        const r = await fetch(`${SIM_BASE}/state`, { signal: AbortSignal.timeout(2000) });
        if (!r.ok) throw new Error('not ok');
        const s: SimState = await r.json();
        if (!cancelled) {
          setStatus('running');
          setState(s);
          setCamera(s.camera);
          setCtrl(s.ctrl);
        }
      } catch {
        if (!cancelled) setStatus('error');
      }
    }

    checkStatus();
    pollRef.current = window.setInterval(checkStatus, 2000);
    return () => {
      cancelled = true;
      if (pollRef.current !== null) clearInterval(pollRef.current);
    };
  }, []);

  const launchSim = useCallback(async () => {
    setLaunching(true);
    const robot = robots.find(r => r.id === selectedRobot);
    await fetch(`${API_BASE}/api/sim/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(robot ? { scene: robot.scene } : {}),
    }).catch(() => {});
    setTimeout(() => setLaunching(false), 3000);
  }, [robots, selectedRobot]);

  const launch3DViewer = useCallback(async () => {
    const robot = robots.find(r => r.id === selectedRobot);
    await fetch(`${API_BASE}/api/sim/viewer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(robot ? { scene: robot.scene } : {}),
    }).catch(() => {});
  }, [robots, selectedRobot]);

  const switchCamera = useCallback(async (name: Camera) => {
    setCamera(name);
    await fetch(`${SIM_BASE}/camera`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    }).catch(() => {});
  }, []);

  const loadKeyframe = useCallback(async (name: string) => {
    const kf: Record<string, number[]> = {
      home:  Array(8).fill(0),
      ready: [0, 1.0, 0.5, 0, 0, 0, 0, 0.05],
    };
    setCtrl(kf[name] ?? Array(8).fill(0));
    await fetch(`${SIM_BASE}/keyframe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    }).catch(() => {});
  }, []);

  const sendCtrl = useCallback(async (newCtrl: number[]) => {
    setCtrl(newCtrl);
    const names = ['j1','j2','j3','j4','j5','j6','j7','gripper_act'];
    const body: Record<string, number> = {};
    names.forEach((n, i) => { body[n] = newCtrl[i]; });
    await fetch(`${SIM_BASE}/ctrl`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).catch(() => {});
  }, []);

  // Joint metadata for sliders
  const joints = [
    { name: 'joint1',  label: 'J1 – Base Yaw',      min: -2.70526, max: 2.70526 },
    { name: 'joint2',  label: 'J2 – Shoulder',       min: -1.74,    max: 1.74    },
    { name: 'joint3',  label: 'J3 – Elbow',          min: -2.75,    max: 2.75    },
    { name: 'joint4',  label: 'J4 – Forearm Roll',   min: -1.01,    max: 2.14    },
    { name: 'joint5',  label: 'J5 – Wrist Pitch',    min: -2.75,    max: 2.75    },
    { name: 'joint6',  label: 'J6 – Wrist Roll',     min: -0.73,    max: 0.95    },
    { name: 'joint7',  label: 'J7 – Wrist Yaw',      min: -1.5708,  max: 1.5708  },
    { name: 'gripper', label: 'Gripper (0=close)',    min: 0,        max: 0.1     },
  ];

  return (
    <div className="sim-viewer">
      <div className="sim-header">
        <div className="sim-header-left">
          <h2 className="sim-title">Robot Simulation — MuJoCo</h2>
          <div className="sim-robot-row">
            {robots.length > 0 ? (
              <select
                className="sim-select"
                value={selectedRobot}
                onChange={e => setSelectedRobot(e.target.value)}
              >
                {robots.map(r => (
                  <option key={r.id} value={r.id}>{r.name}</option>
                ))}
              </select>
            ) : (
              <span className="sim-hint">No scenes in robots/scenes/</span>
            )}
            <span className={`sim-badge sim-badge-${status}`}>
              {status === 'starting' && 'Connecting…'}
              {status === 'running'  && (state?.demo ? 'Demo mode' : 'Manual control')}
              {status === 'error'    && 'Sim offline'}
            </span>
            {status === 'error' && (
              <button className="sim-btn" onClick={launchSim} disabled={launching}>
                {launching ? 'Launching…' : 'Launch Sim'}
              </button>
            )}
          </div>
        </div>
        <div className="sim-controls">
          {CAMERAS.map(c => (
            <button
              key={c}
              className={`sim-btn ${camera === c ? 'active' : ''}`}
              onClick={() => switchCamera(c)}
            >
              {c}
            </button>
          ))}
          <button className="sim-btn" onClick={() => loadKeyframe('home')}>Home</button>
          <button className="sim-btn" onClick={() => loadKeyframe('ready')}>Ready</button>
          <button
            className={`sim-btn ${showSliders ? 'active' : ''}`}
            onClick={() => setShowSliders(s => !s)}
          >
            Joints
          </button>
          <button className="sim-btn" onClick={launch3DViewer} title="Open native MuJoCo viewer (WSLg)">
            3D View
          </button>
        </div>
      </div>

      <div className="sim-body">
        <div className="sim-stream-wrap">
          {status === 'error' && (
            <div className="sim-offline">
              <div className="sim-offline-icon">⚡</div>
              <p>Simulation server not running</p>
              <code>MUJOCO_GL=egl python experiments/robot_sim/sim_server.py</code>
            </div>
          )}
          {/* Always keep img in DOM so stream reconnects automatically.
              Polling is the sole source of truth for online/offline status. */}
          <img
            ref={imgRef}
            key={imgKeyRef.current}
            src={`${SIM_BASE}/stream`}
            alt="MuJoCo simulation"
            className="sim-stream"
            style={{ display: status === 'error' ? 'none' : 'block' }}
            onError={() => {
              // Bump key after a short delay to force the browser to retry the stream.
              // Do NOT touch status — the poll controls that.
              setTimeout(() => {
                imgKeyRef.current += 1;
                if (imgRef.current) {
                  imgRef.current.src = `${SIM_BASE}/stream`;
                }
              }, 2000);
            }}
          />
        </div>

        {showSliders && (
          <div className="sim-sliders">
            <div className="sim-sliders-title">Joint Targets (rad / m)</div>
            {joints.map((j, i) => (
              <div key={j.name} className="sim-slider-row">
                <span className="sim-slider-label">{j.label}</span>
                <input
                  type="range"
                  min={j.min}
                  max={j.max}
                  step={0.01}
                  value={ctrl[i] ?? 0}
                  onChange={e => {
                    const next = [...ctrl];
                    next[i] = parseFloat(e.target.value);
                    sendCtrl(next);
                  }}
                  className="sim-range"
                />
                <span className="sim-slider-val">{(ctrl[i] ?? 0).toFixed(3)}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {state && (
        <div className="sim-footer">
          <span>t = {state.t.toFixed(2)} s</span>
          {state.qpos.slice(0, 7).map((q, i) => (
            <span key={i}>J{i + 1}={q.toFixed(2)}</span>
          ))}
          <span>grip={state.qpos[7]?.toFixed(3)}</span>
        </div>
      )}
    </div>
  );
}
