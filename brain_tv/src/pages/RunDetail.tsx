import { useState, useCallback } from 'react';
import { useParams, useSearchParams, Link } from 'react-router-dom';
import { ConfigViewer } from '../components/ConfigViewer';
import { TensorBoardEmbed } from '../components/TensorBoardEmbed';
import { OutputsGallery } from '../components/OutputsGallery';
import { MetricsChart } from '../components/MetricsChart';
import { CheckpointList } from '../components/CheckpointList';
import { ActionPanel } from '../components/ActionPanel';
import { ArchitectureViewer } from '../components/ArchitectureViewer';
import { RunPicker } from '../components/RunPicker';

const TABS = [
  { id: 'config', label: 'Config' },
  { id: 'architecture', label: 'Architecture' },
  { id: 'metrics', label: 'Metrics & Checkpoints' },
  { id: 'tensorboard', label: 'TensorBoard' },
  { id: 'outputs', label: 'Outputs' },
  { id: 'actions', label: 'Actions' },
] as const;

type TabId = (typeof TABS)[number]['id'];

const RUN_COLORS = ['var(--accent)', 'var(--green)', 'var(--yellow)'];
const RUN_COLOR_NAMES = ['#58a6ff', '#3fb950', '#d29922'];

interface CompareRun {
  experiment: string;
  timestamp: string;
}

export function RunDetail() {
  const { experiment, timestamp } = useParams<{
    experiment: string;
    timestamp: string;
  }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const [activeTab, setActiveTab] = useState<TabId>('config');
  const [showPicker, setShowPicker] = useState(false);

  // Parse compare runs from URL
  const compareRuns: CompareRun[] = (() => {
    const param = searchParams.get('compare');
    if (!param) return [];
    return param
      .split(',')
      .map((s) => {
        const parts = s.split('/');
        if (parts.length >= 2) {
          return { experiment: parts[0], timestamp: parts.slice(1).join('/') };
        }
        return null;
      })
      .filter(Boolean) as CompareRun[];
  })();

  const updateCompareParams = useCallback(
    (runs: CompareRun[]) => {
      if (runs.length === 0) {
        searchParams.delete('compare');
      } else {
        searchParams.set(
          'compare',
          runs.map((r) => `${r.experiment}/${r.timestamp}`).join(',')
        );
      }
      setSearchParams(searchParams, { replace: true });
    },
    [searchParams, setSearchParams]
  );

  const addCompareRun = (exp: string, ts: string) => {
    if (compareRuns.length >= 2) return; // max 2 additional (3 total)
    const next = [...compareRuns, { experiment: exp, timestamp: ts }];
    updateCompareParams(next);
    setShowPicker(false);
  };

  const removeCompareRun = (index: number) => {
    const next = compareRuns.filter((_, i) => i !== index);
    updateCompareParams(next);
  };

  if (!experiment || !timestamp) {
    return <div className="error-msg">Invalid run parameters</div>;
  }

  const basePath = `/api/runs/${experiment}/${timestamp}`;
  const isComparing = compareRuns.length > 0;

  // Build the list of all runs (primary + comparisons) for components
  const allRuns = [
    { experiment, timestamp, basePath, label: `${experiment}/${timestamp}` },
    ...compareRuns.map((r) => ({
      ...r,
      basePath: `/api/runs/${r.experiment}/${r.timestamp}`,
      label: `${r.experiment}/${r.timestamp}`,
    })),
  ];

  const excludeKeys = allRuns.map((r) => `${r.experiment}/${r.timestamp}`);

  return (
    <div>
      <div className="run-breadcrumb">
        <Link to="/">Runs</Link>
        <span>/</span>
        <span>{experiment}</span>
        <span>/</span>
        <span>{timestamp}</span>
      </div>

      <div className="run-header-row">
        <h1 className="run-title">
          {experiment} / {timestamp}
        </h1>
        <div className="compare-controls">
          {compareRuns.length < 2 && (
            <div style={{ position: 'relative' }}>
              <button
                className="btn btn-primary btn-sm"
                onClick={() => setShowPicker(!showPicker)}
              >
                + Compare with...
              </button>
              {showPicker && (
                <RunPicker
                  excludeKeys={excludeKeys}
                  onSelect={addCompareRun}
                  onClose={() => setShowPicker(false)}
                />
              )}
            </div>
          )}
        </div>
      </div>

      {isComparing && (
        <div className="compare-badges">
          <span
            className="compare-badge"
            style={{ borderColor: RUN_COLORS[0] }}
          >
            <span
              className="compare-badge-dot"
              style={{ background: RUN_COLORS[0] }}
            />
            <span className="compare-badge-label">
              {experiment}/{timestamp}
            </span>
          </span>
          {compareRuns.map((run, i) => (
            <span
              key={`${run.experiment}/${run.timestamp}`}
              className="compare-badge"
              style={{ borderColor: RUN_COLORS[i + 1] }}
            >
              <span
                className="compare-badge-dot"
                style={{ background: RUN_COLORS[i + 1] }}
              />
              <span className="compare-badge-label">
                {run.experiment}/{run.timestamp}
              </span>
              <button
                className="compare-badge-remove"
                onClick={() => removeCompareRun(i)}
              >
                &times;
              </button>
            </span>
          ))}
        </div>
      )}

      <div className="tabs">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="tab-content">
        {activeTab === 'config' && (
          <ConfigViewer
            basePath={basePath}
            compareBasePaths={
              isComparing
                ? compareRuns.map(
                    (r) => `/api/runs/${r.experiment}/${r.timestamp}`
                  )
                : undefined
            }
            compareLabels={
              isComparing
                ? allRuns.map((r) => `${r.experiment}/${r.timestamp}`)
                : undefined
            }
            compareColors={isComparing ? RUN_COLOR_NAMES.slice(0, allRuns.length) : undefined}
          />
        )}

        {activeTab === 'metrics' && (
          <div>
            <MetricsChart
              basePath={basePath}
              compareBasePaths={
                isComparing
                  ? compareRuns.map(
                      (r) => `/api/runs/${r.experiment}/${r.timestamp}`
                    )
                  : undefined
              }
              compareLabels={
                isComparing
                  ? allRuns.map((r) => `${r.experiment}/${r.timestamp}`)
                  : undefined
              }
              compareColors={isComparing ? RUN_COLOR_NAMES.slice(0, allRuns.length) : undefined}
            />
            {!isComparing && <CheckpointList basePath={basePath} />}
          </div>
        )}

        {activeTab === 'tensorboard' && (
          <TensorBoardEmbed
            experiment={experiment}
            timestamp={timestamp}
            compareRuns={isComparing ? compareRuns : undefined}
          />
        )}

        {activeTab === 'outputs' && (
          <OutputsGallery
            basePath={basePath}
            compareBasePaths={
              isComparing
                ? compareRuns.map(
                    (r) => `/api/runs/${r.experiment}/${r.timestamp}`
                  )
                : undefined
            }
            compareLabels={
              isComparing
                ? allRuns.map((r) => `${r.experiment}/${r.timestamp}`)
                : undefined
            }
            compareColors={isComparing ? RUN_COLOR_NAMES.slice(0, allRuns.length) : undefined}
          />
        )}

        {activeTab === 'architecture' && (
          <ArchitectureViewer basePath={basePath} />
        )}

        {activeTab === 'actions' && (
          <ActionPanel
            experiment={experiment}
            timestamp={timestamp}
            basePath={basePath}
          />
        )}
      </div>
    </div>
  );
}
