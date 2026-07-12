import path from 'path';
import fs from 'fs';

export const BRAIN_FACTORY_OUT = path.resolve(
  import.meta.dirname,
  '../../brain_factory_out'
);

export const PROJECT_ROOT = path.resolve(import.meta.dirname, '../..');

export interface RunInfo {
  experiment: string;
  timestamp: string;
  hasMetrics: boolean;
  hasCheckpoints: boolean;
  hasVisualizations: boolean;
  hasConfig: boolean;
  lastStep: number | null;
  lastLoss: number | null;
  lastCheckpoint: string | null;
}

export function listRuns(): RunInfo[] {
  const runs: RunInfo[] = [];
  if (!fs.existsSync(BRAIN_FACTORY_OUT)) return runs;

  const experiments = fs
    .readdirSync(BRAIN_FACTORY_OUT, { withFileTypes: true })
    .filter((d) => d.isDirectory());

  for (const exp of experiments) {
    const expPath = path.join(BRAIN_FACTORY_OUT, exp.name);
    const timestamps = fs
      .readdirSync(expPath, { withFileTypes: true })
      .filter((d) => d.isDirectory());

    for (const ts of timestamps) {
      const runPath = path.join(expPath, ts.name);
      const metricsPath = path.join(runPath, 'metrics.jsonl');
      const checkpointsPath = path.join(runPath, 'checkpoints');
      const vizPath = path.join(runPath, 'visualizations');
      const configPath = path.join(runPath, '.hydra', 'config.yaml');

      let lastStep: number | null = null;
      let lastLoss: number | null = null;

      if (fs.existsSync(metricsPath)) {
        try {
          const content = fs.readFileSync(metricsPath, 'utf-8').trim();
          if (content) {
            const lines = content.split('\n');
            const lastLine = JSON.parse(lines[lines.length - 1]);
            lastStep = lastLine.step ?? null;
            lastLoss = lastLine['loss/total'] ?? null;
          }
        } catch {
          // ignore parse errors
        }
      }

      let lastCheckpoint: string | null = null;
      if (fs.existsSync(checkpointsPath)) {
        const cpDirs = fs
          .readdirSync(checkpointsPath, { withFileTypes: true })
          .filter((d) => d.isDirectory())
          .map((d) => d.name)
          .sort((a, b) => {
            const stepA = parseInt(a.match(/(\d+)/)?.[1] ?? '0');
            const stepB = parseInt(b.match(/(\d+)/)?.[1] ?? '0');
            return stepB - stepA;
          });
        if (cpDirs.length > 0) lastCheckpoint = cpDirs[0];
      }

      runs.push({
        experiment: exp.name,
        timestamp: ts.name,
        hasMetrics: fs.existsSync(metricsPath),
        hasCheckpoints: lastCheckpoint !== null,
        hasVisualizations:
          fs.existsSync(vizPath) && fs.readdirSync(vizPath).length > 0,
        hasConfig: fs.existsSync(configPath),
        lastStep,
        lastLoss,
        lastCheckpoint,
      });
    }
  }

  // Sort newest first
  runs.sort((a, b) => b.timestamp.localeCompare(a.timestamp));
  return runs;
}

export function getRunPath(experiment: string, timestamp: string): string {
  return path.join(BRAIN_FACTORY_OUT, experiment, timestamp);
}

export function walkDir(
  dir: string,
  base: string = ''
): { path: string; size: number; isDir: boolean }[] {
  const results: { path: string; size: number; isDir: boolean }[] = [];
  if (!fs.existsSync(dir)) return results;

  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const relPath = base ? `${base}/${entry.name}` : entry.name;
    if (entry.isDirectory()) {
      results.push({ path: relPath, size: 0, isDir: true });
      results.push(...walkDir(path.join(dir, entry.name), relPath));
    } else {
      const stat = fs.statSync(path.join(dir, entry.name));
      results.push({ path: relPath, size: stat.size, isDir: false });
    }
  }
  return results;
}

export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}
