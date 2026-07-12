import { Router } from 'express';
import { spawn, ChildProcess } from 'child_process';
import fs from 'fs';
import path from 'path';
import { PROJECT_ROOT } from '../utils.js';

export const actionsRouter = Router();

const PROJECTS_CONFIG_DIR = path.join(
  PROJECT_ROOT,
  'brain_factory/config/projects'
);

interface RecipeTreeNode {
  name: string;
  type: 'dir' | 'file';
  path?: string; // recipe path for files (e.g. "training/flow_matching")
  children?: RecipeTreeNode[];
}

function buildRecipeTree(dir: string, prefix: string = ''): RecipeTreeNode[] {
  if (!fs.existsSync(dir)) return [];
  const nodes: RecipeTreeNode[] = [];

  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const dirs = entries.filter((e) => e.isDirectory()).sort((a, b) => a.name.localeCompare(b.name));
  const files = entries
    .filter((e) => e.isFile() && (e.name.endsWith('.yaml') || e.name.endsWith('.yml')))
    .sort((a, b) => a.name.localeCompare(b.name));

  for (const d of dirs) {
    const childPrefix = prefix ? `${prefix}/${d.name}` : d.name;
    nodes.push({
      name: d.name,
      type: 'dir',
      children: buildRecipeTree(path.join(dir, d.name), childPrefix),
    });
  }

  for (const f of files) {
    const name = f.name.replace(/\.ya?ml$/, '');
    nodes.push({
      name,
      type: 'file',
      path: prefix ? `${prefix}/${name}` : name,
    });
  }

  return nodes;
}

// List available recipes as a recursive tree. All project configs live
// under the single brain_factory/config/projects/<name>/... tree, so this
// is a plain walk — no per-project merging needed.
actionsRouter.get('/recipes', (_req, res) => {
  res.json(buildRecipeTree(PROJECTS_CONFIG_DIR));
});

interface ProcessInfo {
  id: string;
  type: 'train' | 'inference';
  recipe: string;
  process: ChildProcess;
  output: string[];
  startedAt: string;
  status: 'running' | 'exited';
  exitCode: number | null;
}

const activeProcesses = new Map<string, ProcessInfo>();

actionsRouter.post('/train', (req, res) => {
  const { recipe, overrides = [], checkpointPath } = req.body;
  const id = `train_${Date.now()}`;

  const args = ['-m', 'brain_factory.main', `projects=${recipe}`];
  if (checkpointPath) {
    args.push(`projects.config.resume_checkpoint.path=${checkpointPath}`);
  }
  args.push(...overrides);

  const proc = spawn('python', args, {
    cwd: PROJECT_ROOT,
    stdio: 'pipe',
    env: { ...process.env },
  });

  const info: ProcessInfo = {
    id,
    type: 'train',
    recipe,
    process: proc,
    output: [],
    startedAt: new Date().toISOString(),
    status: 'running',
    exitCode: null,
  };

  const pushLine = (line: string) => {
    info.output.push(line);
    if (info.output.length > 2000) info.output.shift();
  };

  proc.stdout?.on('data', (data: Buffer) => pushLine(data.toString()));
  proc.stderr?.on('data', (data: Buffer) => pushLine(data.toString()));
  proc.on('exit', (code) => {
    info.status = 'exited';
    info.exitCode = code;
  });

  activeProcesses.set(id, info);
  res.json({ id, status: 'started' });
});

actionsRouter.post('/inference', (req, res) => {
  const { recipe, checkpointPath, overrides = [] } = req.body;
  const id = `infer_${Date.now()}`;

  const args = [
    '-m',
    'brain_factory.main',
    `projects=${recipe}`,
    `projects.config.model_checkpoint.path=${checkpointPath}`,
    ...overrides,
  ];

  const proc = spawn('python', args, {
    cwd: PROJECT_ROOT,
    stdio: 'pipe',
    env: { ...process.env },
  });

  const info: ProcessInfo = {
    id,
    type: 'inference',
    recipe,
    process: proc,
    output: [],
    startedAt: new Date().toISOString(),
    status: 'running',
    exitCode: null,
  };

  const pushLine = (line: string) => {
    info.output.push(line);
    if (info.output.length > 2000) info.output.shift();
  };

  proc.stdout?.on('data', (data: Buffer) => pushLine(data.toString()));
  proc.stderr?.on('data', (data: Buffer) => pushLine(data.toString()));
  proc.on('exit', (code) => {
    info.status = 'exited';
    info.exitCode = code;
  });

  activeProcesses.set(id, info);
  res.json({ id, status: 'started' });
});

// List active/recent processes
actionsRouter.get('/processes', (_req, res) => {
  const list = Array.from(activeProcesses.values()).map(
    ({ id, type, recipe, startedAt, status, exitCode }) => ({
      id,
      type,
      recipe,
      startedAt,
      status,
      exitCode,
    })
  );
  res.json(list);
});

// Get process output (streaming tail)
actionsRouter.get('/processes/:id', (req, res) => {
  const info = activeProcesses.get(req.params.id);
  if (!info) {
    res.status(404).json({ error: 'Process not found' });
    return;
  }
  const since = parseInt(String(req.query.since ?? '0'));
  res.json({
    id: info.id,
    status: info.status,
    exitCode: info.exitCode,
    output: info.output.slice(since),
    totalLines: info.output.length,
  });
});

// Kill a process
actionsRouter.post('/processes/:id/kill', (req, res) => {
  const info = activeProcesses.get(req.params.id);
  if (!info) {
    res.status(404).json({ error: 'Process not found' });
    return;
  }
  info.process.kill();
  res.json({ status: 'killed' });
});
