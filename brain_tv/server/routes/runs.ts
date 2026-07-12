import { Router } from 'express';
import { execFile } from 'child_process';
import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';
import { listRuns, getRunPath, walkDir, formatBytes } from '../utils.js';

const SCRIPTS_DIR = path.join(import.meta.dirname, '..', 'scripts');

export const runsRouter = Router();

// List all runs
runsRouter.get('/', (_req, res) => {
  res.json(listRuns());
});

// Get config for a run
runsRouter.get('/:experiment/:timestamp/config', (req, res) => {
  const runPath = getRunPath(req.params.experiment, req.params.timestamp);
  const configPath = path.join(runPath, '.hydra', 'config.yaml');
  const overridesPath = path.join(runPath, '.hydra', 'overrides.yaml');

  if (!fs.existsSync(configPath)) {
    res.status(404).json({ error: 'Config not found' });
    return;
  }

  const config = yaml.load(fs.readFileSync(configPath, 'utf-8'));
  let overrides: unknown = null;
  if (fs.existsSync(overridesPath)) {
    overrides = yaml.load(fs.readFileSync(overridesPath, 'utf-8'));
  }

  res.json({ config, overrides });
});

// Get metrics for a run
runsRouter.get('/:experiment/:timestamp/metrics', (req, res) => {
  const runPath = getRunPath(req.params.experiment, req.params.timestamp);
  const metricsPath = path.join(runPath, 'metrics.jsonl');

  if (!fs.existsSync(metricsPath)) {
    res.json([]);
    return;
  }

  const content = fs.readFileSync(metricsPath, 'utf-8').trim();
  if (!content) {
    res.json([]);
    return;
  }

  const metrics = content
    .split('\n')
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    })
    .filter(Boolean);

  res.json(metrics);
});

// List checkpoints for a run
runsRouter.get('/:experiment/:timestamp/checkpoints', (req, res) => {
  const runPath = getRunPath(req.params.experiment, req.params.timestamp);
  const checkpointsDir = path.join(runPath, 'checkpoints');

  if (!fs.existsSync(checkpointsDir)) {
    res.json([]);
    return;
  }

  const checkpoints = fs
    .readdirSync(checkpointsDir, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => {
      const cpPath = path.join(checkpointsDir, d.name);
      const files = fs.readdirSync(cpPath).map((f) => {
        const stat = fs.statSync(path.join(cpPath, f));
        return { name: f, size: formatBytes(stat.size) };
      });
      const stepMatch = d.name.match(/step_(\d+)/);
      return {
        name: d.name,
        step: stepMatch ? parseInt(stepMatch[1]) : null,
        path: cpPath,
        files,
      };
    })
    .sort((a, b) => (a.step ?? 0) - (b.step ?? 0));

  res.json(checkpoints);
});

// Get weight metadata for a checkpoint
runsRouter.get('/:experiment/:timestamp/checkpoints/:name/weights', (req, res) => {
  const runPath = getRunPath(req.params.experiment, req.params.timestamp);
  const cpDir = path.join(runPath, 'checkpoints', req.params.name);
  const modelPath = path.join(cpDir, 'model.safetensors');

  if (!modelPath.startsWith(runPath) || !fs.existsSync(modelPath)) {
    res.status(404).json({ error: 'model.safetensors not found' });
    return;
  }

  const scriptPath = path.join(SCRIPTS_DIR, 'read_weights.py');
  execFile(
    'python3',
    [scriptPath, modelPath, '--mode', 'metadata'],
    { maxBuffer: 10 * 1024 * 1024 },
    (error, stdout, stderr) => {
      if (error) {
        res.status(500).json({ error: `Failed to read weights: ${stderr}` });
        return;
      }
      try {
        res.json(JSON.parse(stdout));
      } catch {
        res.status(500).json({ error: 'Failed to parse weight metadata' });
      }
    }
  );
});

// Get heatmap data for a specific weight tensor
runsRouter.get(
  '/:experiment/:timestamp/checkpoints/:name/weights/*tensorName',
  (req, res) => {
    const runPath = getRunPath(req.params.experiment, req.params.timestamp);
    const cpDir = path.join(runPath, 'checkpoints', req.params.name);
    const modelPath = path.join(cpDir, 'model.safetensors');

    if (!modelPath.startsWith(runPath) || !fs.existsSync(modelPath)) {
      res.status(404).json({ error: 'model.safetensors not found' });
      return;
    }

    const scriptPath = path.join(SCRIPTS_DIR, 'read_weights.py');
    execFile(
      'python3',
      [scriptPath, modelPath, '--mode', 'tensor', '--name', [req.params.tensorName].flat().join('/')],
      { maxBuffer: 50 * 1024 * 1024 },
      (error, stdout, stderr) => {
        if (error) {
          res.status(500).json({ error: `Failed to read tensor: ${stderr}` });
          return;
        }
        try {
          res.json(JSON.parse(stdout));
        } catch {
          res.status(500).json({ error: 'Failed to parse tensor data' });
        }
      }
    );
  }
);

// List outputs/visualizations for a run
runsRouter.get('/:experiment/:timestamp/outputs', (req, res) => {
  const runPath = getRunPath(req.params.experiment, req.params.timestamp);
  const vizDir = path.join(runPath, 'visualizations');
  res.json(walkDir(vizDir));
});

// Serve static files from a run directory
runsRouter.get('/:experiment/:timestamp/files/*filepath', (req, res) => {
  const runPath = getRunPath(req.params.experiment, req.params.timestamp);
  const relativePath = [req.params.filepath].flat().join('/');
  const filePath = path.join(runPath, relativePath);

  // Security: prevent path traversal
  if (!filePath.startsWith(runPath)) {
    res.status(403).json({ error: 'Forbidden' });
    return;
  }

  if (!fs.existsSync(filePath)) {
    res.status(404).json({ error: 'File not found' });
    return;
  }

  res.sendFile(filePath);
});

// Get model architecture graph
runsRouter.get('/:experiment/:timestamp/architecture', (req, res) => {
  const runPath = getRunPath(req.params.experiment, req.params.timestamp);
  const archPath = path.join(runPath, 'architecture.json');

  if (!fs.existsSync(archPath)) {
    res.status(404).json({ error: 'Architecture graph not found' });
    return;
  }

  try {
    const data = JSON.parse(fs.readFileSync(archPath, 'utf-8'));
    res.json(data);
  } catch {
    res.status(500).json({ error: 'Failed to parse architecture.json' });
  }
});

// Get pipeline log
runsRouter.get('/:experiment/:timestamp/log', (req, res) => {
  const runPath = getRunPath(req.params.experiment, req.params.timestamp);
  const logPath = path.join(runPath, 'pipeline.log');

  if (!fs.existsSync(logPath)) {
    res.status(404).json({ error: 'Log not found' });
    return;
  }

  // Return last 500 lines
  const content = fs.readFileSync(logPath, 'utf-8');
  const lines = content.split('\n');
  const tail = lines.slice(-500).join('\n');
  res.type('text/plain').send(tail);
});
