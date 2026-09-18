import { Router } from 'express';
import { spawn, ChildProcess } from 'child_process';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const router = Router();
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT   = path.resolve(__dirname, '..', '..', '..');
const VENV_PY     = path.join(REPO_ROOT, '.venv', 'bin', 'python');
const SIM_SCRIPT  = path.join(REPO_ROOT, 'experiments', 'robot_sim', 'sim_server.py');
const SCENES_DIR  = path.join(REPO_ROOT, 'robots', 'scenes');
const SIM_PORT    = 5202;

let proc: ChildProcess | null = null;
let viewerProc: ChildProcess | null = null;
let log: string[] = [];

function appendLog(line: string) {
  log.push(line);
  if (log.length > 200) log.shift();
}

async function isSimAlive(): Promise<boolean> {
  try {
    const res = await fetch(`http://localhost:${SIM_PORT}/state`, {
      signal: AbortSignal.timeout(1000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

// List available MJCF scene files under robots/scenes/
router.get('/robots', (_req, res) => {
  let robots: { id: string; name: string; scene: string }[] = [];
  try {
    const files = fs.readdirSync(SCENES_DIR).filter(f => f.endsWith('.xml'));
    robots = files.map(f => ({
      id:    path.basename(f, '.xml'),
      name:  path.basename(f, '.xml').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
      scene: path.join(SCENES_DIR, f),
    }));
  } catch {
    // scenes dir missing — return empty list
  }
  res.json(robots);
});

router.get('/status', (_req, res) => {
  res.json({ running: proc !== null && !proc.killed, pid: proc?.pid ?? null, log });
});

router.post('/start', async (req, res) => {
  // If the sim is already responding, no need to spawn another one.
  if (await isSimAlive()) {
    return res.json({ ok: true, already: true });
  }

  // Kill stale tracked proc if any
  if (proc && !proc.killed) {
    proc.kill('SIGTERM');
    proc = null;
    await new Promise(r => setTimeout(r, 500));
  }

  const scene: string | undefined = (req.body as { scene?: string }).scene;
  const args = [SIM_SCRIPT];
  if (scene) args.push('--scene', scene);

  log = [];
  appendLog('[brain_tv] Starting sim server…');
  proc = spawn(VENV_PY, args, {
    env: { ...process.env, MUJOCO_GL: 'egl' },
    cwd: REPO_ROOT,
  });
  proc.stdout?.on('data', d => d.toString().split('\n').filter(Boolean).forEach(appendLog));
  proc.stderr?.on('data', d => d.toString().split('\n').filter(Boolean).forEach(appendLog));
  proc.on('exit', (code) => {
    appendLog(`[brain_tv] Sim server exited (code=${code})`);
    proc = null;
  });
  res.json({ ok: true, pid: proc.pid });
});

router.post('/stop', (_req, res) => {
  if (proc && !proc.killed) {
    proc.kill('SIGTERM');
    appendLog('[brain_tv] Sent SIGTERM to sim server');
  }
  res.json({ ok: true });
});

// Launch the native MuJoCo interactive viewer (requires WSLg / X11)
router.post('/viewer', (req, res) => {
  if (viewerProc && !viewerProc.killed) {
    return res.json({ ok: true, already: true, pid: viewerProc.pid });
  }
  const scene: string | undefined = (req.body as { scene?: string }).scene
    ?? path.join(SCENES_DIR, 'nero_sim.xml');

  const script = `
import mujoco, mujoco.viewer, sys
model = mujoco.MjModel.from_xml_path(sys.argv[1])
data  = mujoco.MjData(model)
mujoco.viewer.launch(model, data)
`.trim();

  viewerProc = spawn(VENV_PY, ['-c', script, scene], {
    env: { ...process.env, DISPLAY: process.env.DISPLAY ?? ':0' },
    cwd: REPO_ROOT,
    detached: false,
  });
  viewerProc.on('exit', () => { viewerProc = null; });
  res.json({ ok: true, pid: viewerProc.pid });
});

export { router as simRouter };
