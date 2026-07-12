import { useState, useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { useApi } from '../hooks/useApi';
import type { MetricEntry } from '../types';

const COLORS = [
  '#58a6ff',
  '#3fb950',
  '#d29922',
  '#f85149',
  '#a371f7',
  '#79c0ff',
  '#56d364',
  '#e3b341',
];

const DASH_PATTERNS = ['', '8 4', '4 4'];

interface MetricsChartProps {
  basePath: string;
  compareBasePaths?: string[];
  compareLabels?: string[];
  compareColors?: string[];
}

export function MetricsChart({
  basePath,
  compareBasePaths,
  compareLabels,
  compareColors,
}: MetricsChartProps) {
  const isComparing =
    compareBasePaths && compareLabels && compareColors && compareBasePaths.length > 0;

  if (isComparing) {
    return (
      <CompareMetricsChart
        basePath={basePath}
        compareBasePaths={compareBasePaths}
        compareLabels={compareLabels}
        compareColors={compareColors}
      />
    );
  }

  return <SingleMetricsChart basePath={basePath} />;
}

function SingleMetricsChart({ basePath }: { basePath: string }) {
  const {
    data: metrics,
    loading,
    error,
  } = useApi<MetricEntry[]>(`${basePath}/metrics`, 5000);

  const allKeys = useMemo(() => {
    if (!metrics || metrics.length === 0) return [];
    const keys = new Set<string>();
    for (const entry of metrics) {
      for (const key of Object.keys(entry)) {
        if (key !== 'step') keys.add(key);
      }
    }
    return Array.from(keys).sort();
  }, [metrics]);

  const keyGroups = useMemo(() => {
    const groups = new Map<string, string[]>();
    for (const key of allKeys) {
      const prefix = key.includes('/') ? key.split('/')[0] : 'other';
      const list = groups.get(prefix) ?? [];
      list.push(key);
      groups.set(prefix, list);
    }
    return groups;
  }, [allKeys]);

  const [activeKeys, setActiveKeys] = useState<Set<string>>(
    () => new Set(allKeys.filter((k) => k.startsWith('loss/')))
  );

  useMemo(() => {
    if (activeKeys.size === 0 && allKeys.length > 0) {
      const lossKeys = allKeys.filter((k) => k.startsWith('loss/'));
      if (lossKeys.length > 0) {
        setActiveKeys(new Set(lossKeys));
      } else {
        setActiveKeys(new Set([allKeys[0]]));
      }
    }
  }, [allKeys, activeKeys.size]);

  const toggleKey = (key: string) => {
    setActiveKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  if (loading) return <div className="loading">Loading metrics...</div>;
  if (error) return <div className="error-msg">{error}</div>;
  if (!metrics || metrics.length === 0) {
    return (
      <div className="empty-state">
        <h3>No metrics recorded</h3>
        <p>This run has no metrics.jsonl file yet.</p>
      </div>
    );
  }

  return (
    <div className="chart-container">
      <div className="chart-title">
        Training Metrics ({metrics.length} steps logged)
      </div>

      <div className="chart-keys">
        {Array.from(keyGroups.entries()).map(([group, keys]) => (
          <span key={group} style={{ display: 'contents' }}>
            {keys.map((key) => (
              <button
                key={key}
                className={`chart-key-toggle ${activeKeys.has(key) ? 'active' : ''}`}
                onClick={() => toggleKey(key)}
              >
                {key}
              </button>
            ))}
          </span>
        ))}
      </div>

      {activeKeys.size > 0 && (
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={metrics}>
            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
            <XAxis
              dataKey="step"
              stroke="#8b949e"
              fontSize={12}
              tickFormatter={(v) => String(v)}
            />
            <YAxis stroke="#8b949e" fontSize={12} />
            <Tooltip
              contentStyle={{
                background: '#161b22',
                border: '1px solid #30363d',
                borderRadius: '6px',
                fontSize: '13px',
                fontFamily: 'monospace',
              }}
              labelStyle={{ color: '#e6edf3' }}
            />
            <Legend />
            {Array.from(activeKeys).map((key, i) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={COLORS[i % COLORS.length]}
                dot={false}
                strokeWidth={2}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

function CompareMetricsChart({
  basePath,
  compareBasePaths,
  compareLabels,
  compareColors,
}: Required<Omit<MetricsChartProps, 'compareColors'>> & {
  compareColors: string[];
}) {
  const allPaths = [basePath, ...compareBasePaths];

  const primaryMetrics = useApi<MetricEntry[]>(`${allPaths[0]}/metrics`, 5000);
  const compare1Metrics = useApi<MetricEntry[]>(
    allPaths[1] ? `${allPaths[1]}/metrics` : ''
  );
  const compare2Metrics = useApi<MetricEntry[]>(
    allPaths[2] ? `${allPaths[2]}/metrics` : ''
  );

  const allMetrics = [primaryMetrics, compare1Metrics, compare2Metrics].slice(
    0,
    allPaths.length
  );

  // Collect all metric keys across all runs
  const allKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const m of allMetrics) {
      if (!m.data) continue;
      for (const entry of m.data) {
        for (const key of Object.keys(entry)) {
          if (key !== 'step') keys.add(key);
        }
      }
    }
    return Array.from(keys).sort();
  }, [allMetrics.map((m) => m.data).join(',')]);

  // Per-run metric keys (which keys does each run have?)
  const perRunKeys = useMemo(() => {
    return allMetrics.map((m) => {
      const keys = new Set<string>();
      if (m.data) {
        for (const entry of m.data) {
          for (const key of Object.keys(entry)) {
            if (key !== 'step') keys.add(key);
          }
        }
      }
      return keys;
    });
  }, [allMetrics.map((m) => m.data).join(',')]);

  const [activeKeys, setActiveKeys] = useState<Set<string>>(new Set());
  const [activeRuns, setActiveRuns] = useState<Set<number>>(
    () => new Set(allPaths.map((_, i) => i))
  );

  useMemo(() => {
    if (activeKeys.size === 0 && allKeys.length > 0) {
      const lossKeys = allKeys.filter((k) => k.startsWith('loss/'));
      if (lossKeys.length > 0) {
        setActiveKeys(new Set(lossKeys));
      } else {
        setActiveKeys(new Set([allKeys[0]]));
      }
    }
  }, [allKeys, activeKeys.size]);

  // Keep activeRuns in sync if number of runs changes
  useMemo(() => {
    setActiveRuns((prev) => {
      if (prev.size === 0) return new Set(allPaths.map((_, i) => i));
      return prev;
    });
  }, [allPaths.length]);

  const toggleKey = (key: string) => {
    setActiveKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const toggleRun = (runIdx: number) => {
    setActiveRuns((prev) => {
      const next = new Set(prev);
      if (next.has(runIdx)) next.delete(runIdx);
      else next.add(runIdx);
      return next;
    });
  };

  const soloRun = (runIdx: number) => {
    setActiveRuns((prev) => {
      // If this run is the only one active, re-enable all
      if (prev.size === 1 && prev.has(runIdx)) {
        return new Set(allPaths.map((_, i) => i));
      }
      return new Set([runIdx]);
    });
  };

  // Merge all runs' data into a unified dataset keyed by step
  // Each metric key is prefixed with run index to avoid collisions
  const { mergedData, lineConfigs } = useMemo(() => {
    const stepMap = new Map<
      number,
      Record<string, number>
    >();

    for (let runIdx = 0; runIdx < allMetrics.length; runIdx++) {
      if (!activeRuns.has(runIdx)) continue;
      const data = allMetrics[runIdx].data;
      if (!data) continue;
      for (const entry of data) {
        const step = entry.step;
        if (!stepMap.has(step)) stepMap.set(step, { step });
        const row = stepMap.get(step)!;
        for (const [key, val] of Object.entries(entry)) {
          if (key === 'step') continue;
          row[`r${runIdx}__${key}`] = val;
        }
      }
    }

    const merged = Array.from(stepMap.values()).sort((a, b) => a.step - b.step);

    // Build line configs for active keys + active runs
    const lines: {
      dataKey: string;
      stroke: string;
      dashArray: string;
      label: string;
      runIdx: number;
    }[] = [];

    for (const key of activeKeys) {
      for (let runIdx = 0; runIdx < allPaths.length; runIdx++) {
        if (!activeRuns.has(runIdx)) continue;
        const dataKey = `r${runIdx}__${key}`;
        if (perRunKeys[runIdx]?.has(key)) {
          const shortLabel = compareLabels[runIdx].split('/').pop() ?? compareLabels[runIdx];
          lines.push({
            dataKey,
            stroke: compareColors[runIdx],
            dashArray: DASH_PATTERNS[runIdx],
            label: `${shortLabel}: ${key}`,
            runIdx,
          });
        }
      }
    }

    return { mergedData: merged, lineConfigs: lines };
  }, [allMetrics.map((m) => m.data).join(','), activeKeys, activeRuns, allPaths.length]);

  const anyLoading = allMetrics.some((m) => m.loading);
  if (anyLoading)
    return <div className="loading">Loading metrics for comparison...</div>;

  const anyData = allMetrics.some((m) => m.data && m.data.length > 0);
  if (!anyData) {
    return (
      <div className="empty-state">
        <h3>No metrics recorded</h3>
        <p>None of the selected runs have metrics data.</p>
      </div>
    );
  }

  return (
    <div className="chart-container">
      <div className="chart-title">Metrics Comparison</div>

      {/* Run toggles */}
      <div className="compare-run-toggles">
        <span className="compare-run-toggles-label">Runs:</span>
        {compareLabels.map((label, i) => {
          const shortLabel = label.split('/').pop() ?? label;
          const isActive = activeRuns.has(i);
          return (
            <button
              key={i}
              className={`compare-run-toggle ${isActive ? 'active' : ''}`}
              style={{
                borderColor: isActive ? compareColors[i] : 'var(--border)',
                color: isActive ? compareColors[i] : 'var(--text-dim)',
              }}
              onClick={() => toggleRun(i)}
              onDoubleClick={() => soloRun(i)}
              title={`Click to toggle, double-click to solo\n${label}`}
            >
              <span
                className="compare-badge-dot"
                style={{
                  background: isActive ? compareColors[i] : 'var(--text-dim)',
                }}
              />
              {shortLabel}
              {DASH_PATTERNS[i] && (
                <span className="compare-run-dash-hint">
                  {DASH_PATTERNS[i] === '8 4' ? '- -' : '...'}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Metric key toggles */}
      <div className="chart-keys">
        {allKeys.map((key) => (
          <button
            key={key}
            className={`chart-key-toggle ${activeKeys.has(key) ? 'active' : ''}`}
            onClick={() => toggleKey(key)}
          >
            {key}
          </button>
        ))}
      </div>

      {lineConfigs.length > 0 && (
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={mergedData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
            <XAxis
              dataKey="step"
              stroke="#8b949e"
              fontSize={12}
              tickFormatter={(v) => String(v)}
            />
            <YAxis stroke="#8b949e" fontSize={12} />
            <Tooltip
              contentStyle={{
                background: '#161b22',
                border: '1px solid #30363d',
                borderRadius: '6px',
                fontSize: '13px',
                fontFamily: 'monospace',
              }}
              labelStyle={{ color: '#e6edf3' }}
              formatter={(value: number, name: string) => {
                const line = lineConfigs.find((l) => l.dataKey === name);
                return [value?.toFixed(6) ?? '-', line?.label ?? name];
              }}
            />
            <Legend
              formatter={(value: string) => {
                const line = lineConfigs.find((l) => l.dataKey === value);
                return line?.label ?? value;
              }}
            />
            {lineConfigs.map((line) => (
              <Line
                key={line.dataKey}
                type="monotone"
                dataKey={line.dataKey}
                stroke={line.stroke}
                strokeDasharray={line.dashArray || undefined}
                dot={false}
                strokeWidth={2}
                connectNulls={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}

      {lineConfigs.length === 0 && (
        <div className="empty-state" style={{ padding: 40 }}>
          <p>Select at least one run and one metric to display.</p>
        </div>
      )}
    </div>
  );
}
