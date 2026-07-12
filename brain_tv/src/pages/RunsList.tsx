import { useState, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { useApi } from '../hooks/useApi';
import { ProcessesPanel } from '../components/ProcessesPanel';
import type { RunInfo } from '../types';

function matchesSearch(run: RunInfo, query: string): boolean {
  if (!query) return true;

  const tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
  const haystack = [
    run.experiment,
    run.timestamp,
    run.lastStep !== null ? `step:${run.lastStep}` : '',
    run.lastCheckpoint ?? '',
    run.hasMetrics ? 'metrics' : '',
    run.hasCheckpoints ? 'ckpt checkpoint' : '',
    run.hasVisualizations ? 'viz visualizations' : '',
    run.hasConfig ? 'config cfg' : '',
    run.lastLoss !== null ? `loss:${run.lastLoss}` : '',
  ]
    .join(' ')
    .toLowerCase();

  return tokens.every((token) => {
    // Support key:value filters
    if (token.startsWith('exp:') || token.startsWith('experiment:')) {
      const val = token.split(':')[1];
      return run.experiment.toLowerCase().includes(val);
    }
    if (token.startsWith('step>')) {
      const val = parseInt(token.slice(5));
      return run.lastStep !== null && run.lastStep > val;
    }
    if (token.startsWith('step<')) {
      const val = parseInt(token.slice(5));
      return run.lastStep !== null && run.lastStep < val;
    }
    if (token.startsWith('loss<')) {
      const val = parseFloat(token.slice(5));
      return run.lastLoss !== null && run.lastLoss < val;
    }
    if (token.startsWith('loss>')) {
      const val = parseFloat(token.slice(5));
      return run.lastLoss !== null && run.lastLoss > val;
    }
    if (token === 'has:ckpt' || token === 'has:checkpoint') {
      return run.hasCheckpoints;
    }
    if (token === 'has:metrics') {
      return run.hasMetrics;
    }
    if (token === 'has:viz') {
      return run.hasVisualizations;
    }
    return haystack.includes(token);
  });
}

const HOME_TABS = [
  { id: 'runs', label: 'Runs' },
  { id: 'processes', label: 'Processes' },
] as const;

type HomeTab = (typeof HOME_TABS)[number]['id'];

export function RunsList() {
  const [activeTab, setActiveTab] = useState<HomeTab>('runs');

  return (
    <div>
      <div className="tabs" style={{ marginBottom: 20 }}>
        {HOME_TABS.map((tab) => (
          <button
            key={tab.id}
            className={`tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'runs' && <RunsTable />}
      {activeTab === 'processes' && <ProcessesPanel />}
    </div>
  );
}

function RunsTable() {
  const { data: runs, loading, error } = useApi<RunInfo[]>(
    '/api/runs',
    5000
  );
  const [search, setSearch] = useState('');

  const filtered = useMemo(() => {
    if (!runs) return [];
    return runs.filter((r) => matchesSearch(r, search));
  }, [runs, search]);

  const grouped = useMemo(() => {
    const map = new Map<string, RunInfo[]>();
    for (const run of filtered) {
      const list = map.get(run.experiment) ?? [];
      list.push(run);
      map.set(run.experiment, list);
    }
    return map;
  }, [filtered]);

  if (loading) return <div className="loading">Loading runs...</div>;
  if (error) return <div className="error-msg">{error}</div>;
  if (!runs || runs.length === 0) {
    return (
      <div className="empty-state">
        <h3>No runs found</h3>
        <p>Start a training run and it will appear here.</p>
      </div>
    );
  }

  return (
    <div>
      <div className="runs-header">
        <h2>Training Runs</h2>
        <span className="runs-count">
          {filtered.length === runs.length
            ? `${runs.length} runs`
            : `${filtered.length} / ${runs.length} runs`}
        </span>
      </div>

      <div className="search-bar">
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search: flow_match, has:ckpt, step>100, loss<0.5, exp:default ..."
          className="search-input"
        />
        {search && (
          <button className="search-clear" onClick={() => setSearch('')}>
            &times;
          </button>
        )}
      </div>

      {filtered.length === 0 ? (
        <div className="empty-state" style={{ padding: '40px' }}>
          <h3>No matching runs</h3>
          <p>Try a different search query.</p>
        </div>
      ) : (
        <table className="runs-table">
          <thead>
            <tr>
              <th>Experiment</th>
              <th>Timestamp</th>
              <th>Steps</th>
              <th>Last Loss</th>
              <th>Checkpoint</th>
              <th>Data</th>
            </tr>
          </thead>
          <tbody>
            {Array.from(grouped.entries()).map(([experiment, expRuns]) =>
              expRuns.map((run, i) => (
                <tr key={`${run.experiment}/${run.timestamp}`}>
                  {i === 0 && (
                    <td rowSpan={expRuns.length} className="exp-cell">
                      {experiment}
                    </td>
                  )}
                  <td>
                    <Link
                      to={`/run/${run.experiment}/${run.timestamp}`}
                      className="mono"
                    >
                      {run.timestamp}
                    </Link>
                  </td>
                  <td className="mono">
                    {run.lastStep !== null
                      ? run.lastStep.toLocaleString()
                      : '-'}
                  </td>
                  <td className="mono">
                    {run.lastLoss !== null ? run.lastLoss.toFixed(6) : '-'}
                  </td>
                  <td className="mono">
                    {run.lastCheckpoint ? (
                      <span className="badge badge-yellow">
                        {run.lastCheckpoint}
                      </span>
                    ) : (
                      <span style={{ color: 'var(--text-dim)' }}>-</span>
                    )}
                  </td>
                  <td>
                    <span className="badge-row">
                      {run.hasConfig && (
                        <span className="badge badge-green">cfg</span>
                      )}
                      {run.hasMetrics && (
                        <span className="badge badge-green">metrics</span>
                      )}
                      {run.hasVisualizations && (
                        <span className="badge badge-yellow">viz</span>
                      )}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      )}
    </div>
  );
}
