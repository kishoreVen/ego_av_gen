import { Router } from 'express';
import { spawn, ChildProcess } from 'child_process';
import path from 'path';
import fs from 'fs';
import os from 'os';
import { getRunPath } from '../utils.js';

export const tensorboardRouter = Router();

let tbProcess: ChildProcess | null = null;
let tbPort: number | null = null;
let tbRunKey: string | null = null;
let tbSymlinkDir: string | null = null;

function cleanupSymlinkDir() {
  if (tbSymlinkDir && fs.existsSync(tbSymlinkDir)) {
    fs.rmSync(tbSymlinkDir, { recursive: true, force: true });
    tbSymlinkDir = null;
  }
}

// Clean up any stale bfw-tb-* dirs from previous crashed sessions
function cleanupStaleDirs() {
  const tmpDir = os.tmpdir();
  try {
    for (const entry of fs.readdirSync(tmpDir, { withFileTypes: true })) {
      if (entry.isDirectory() && entry.name.startsWith('bfw-tb-')) {
        fs.rmSync(path.join(tmpDir, entry.name), { recursive: true, force: true });
      }
    }
  } catch {
    // /tmp read failed, not critical
  }
}
cleanupStaleDirs();

tensorboardRouter.post('/start', (req, res) => {
  const { experiment, timestamp, compareRuns } = req.body as {
    experiment: string;
    timestamp: string;
    compareRuns?: { experiment: string; timestamp: string }[];
  };

  // Build a stable key from all runs being viewed
  const allRuns = [
    { experiment, timestamp },
    ...(compareRuns ?? []),
  ];
  const runKey = allRuns
    .map((r) => `${r.experiment}/${r.timestamp}`)
    .sort()
    .join('|');

  // Already running for this exact set of runs
  if (tbProcess && tbRunKey === runKey && tbPort) {
    res.json({ port: tbPort, status: 'already_running' });
    return;
  }

  // Kill existing
  if (tbProcess) {
    tbProcess.kill();
    tbProcess = null;
  }
  cleanupSymlinkDir();

  const port = 6006 + Math.floor(Math.random() * 100);

  let logdir: string;

  if (allRuns.length === 1) {
    // Single run — point directly at its tb dir
    const runPath = getRunPath(experiment, timestamp);
    logdir = path.join(runPath, 'tb');
  } else {
    // Multiple runs — create a temp dir with symlinks so TB discovers them
    // as separate runs via recursive directory walking
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'bfw-tb-'));
    tbSymlinkDir = tmpDir;
    for (const r of allRuns) {
      const runPath = getRunPath(r.experiment, r.timestamp);
      const tbDir = path.join(runPath, 'tb');
      const linkName = `${r.experiment}_${r.timestamp}`;
      try {
        fs.symlinkSync(tbDir, path.join(tmpDir, linkName));
      } catch {
        // If symlink fails (e.g. tb dir doesn't exist), skip
      }
    }
    logdir = tmpDir;
  }

  tbProcess = spawn(
    'tensorboard',
    ['--logdir', logdir, '--port', String(port), '--bind_all'],
    { stdio: 'pipe' }
  );

  tbPort = port;
  tbRunKey = runKey;

  tbProcess.on('error', () => {
    tbProcess = null;
    tbPort = null;
    tbRunKey = null;
    cleanupSymlinkDir();
  });

  tbProcess.on('exit', () => {
    if (tbRunKey === runKey) {
      tbProcess = null;
      tbPort = null;
      tbRunKey = null;
      cleanupSymlinkDir();
    }
  });

  // Give TB a moment to start
  setTimeout(() => {
    res.json({ port, status: 'started' });
  }, 2000);
});

tensorboardRouter.post('/stop', (_req, res) => {
  if (tbProcess) {
    tbProcess.kill();
    tbProcess = null;
    tbPort = null;
    tbRunKey = null;
  }
  cleanupSymlinkDir();
  res.json({ status: 'stopped' });
});

tensorboardRouter.get('/status', (_req, res) => {
  res.json({
    running: tbProcess !== null,
    port: tbPort,
    runKey: tbRunKey,
    symlinkDir: tbSymlinkDir,
  });
});
