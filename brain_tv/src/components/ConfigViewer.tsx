import { type ReactNode, useState, useMemo, useEffect } from 'react';
import { useApi } from '../hooks/useApi';

interface ConfigData {
  config: Record<string, unknown>;
  overrides: string[] | null;
}

interface ConfigViewerProps {
  basePath: string;
  compareBasePaths?: string[];
  compareLabels?: string[];
  compareColors?: string[];
}

// Flatten a nested config object to dot-path → value map
function flattenConfig(
  obj: unknown,
  prefix: string = ''
): Map<string, unknown> {
  const result = new Map<string, unknown>();
  if (obj === null || obj === undefined || typeof obj !== 'object') {
    result.set(prefix, obj);
    return result;
  }
  if (Array.isArray(obj)) {
    result.set(prefix, obj);
    return result;
  }
  const entries = Object.entries(obj as Record<string, unknown>);
  if (entries.length === 0) {
    result.set(prefix, obj);
    return result;
  }
  for (const [key, value] of entries) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (
      value !== null &&
      typeof value === 'object' &&
      !Array.isArray(value) &&
      Object.keys(value as object).length > 0
    ) {
      for (const [k, v] of flattenConfig(value, path)) {
        result.set(k, v);
      }
    } else {
      result.set(path, value);
    }
  }
  return result;
}

function valueToString(value: unknown): string {
  if (value === null || value === undefined) return 'null';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function valuesAllEqual(values: unknown[]): boolean {
  if (values.length <= 1) return true;
  const first = valueToString(values[0]);
  return values.every((v) => valueToString(v) === first);
}

function renderYaml(
  obj: unknown,
  indent: number = 0,
  collapsed: Set<string>,
  toggleCollapse: (path: string) => void,
  path: string = '',
  diffPaths?: Set<string>
): ReactNode[] {
  const elements: ReactNode[] = [];
  const pad = '  '.repeat(indent);

  if (obj === null || obj === undefined) {
    elements.push(
      <span key={path || 'null'}>
        <span className="config-null">null</span>
        {'\n'}
      </span>
    );
    return elements;
  }

  if (typeof obj !== 'object') {
    let className = 'config-str';
    let display = String(obj);
    if (typeof obj === 'number') className = 'config-num';
    else if (typeof obj === 'boolean') className = 'config-bool';
    else display = `"${display}"`;

    elements.push(
      <span key={path} className={className}>
        {display}
      </span>
    );
    return elements;
  }

  if (Array.isArray(obj)) {
    if (obj.length === 0) {
      elements.push(<span key={path}>{'[]'}</span>);
      return elements;
    }
    for (let i = 0; i < obj.length; i++) {
      const itemPath = `${path}[${i}]`;
      elements.push(
        <span key={`${itemPath}-prefix`}>
          {pad}- {' '}
        </span>
      );
      elements.push(
        ...renderYaml(
          obj[i],
          indent + 1,
          collapsed,
          toggleCollapse,
          itemPath,
          diffPaths
        )
      );
      elements.push(<span key={`${itemPath}-nl`}>{'\n'}</span>);
    }
    return elements;
  }

  const entries = Object.entries(obj as Record<string, unknown>);
  for (const [key, value] of entries) {
    const keyPath = path ? `${path}.${key}` : key;
    const isObject =
      value !== null && typeof value === 'object' && !Array.isArray(value);
    const isCollapsed = collapsed.has(keyPath);
    const isDiff = diffPaths?.has(keyPath);

    // Check if any child of this key has a diff
    const hasChildDiff =
      diffPaths &&
      Array.from(diffPaths).some(
        (p) => p.startsWith(keyPath + '.') || p.startsWith(keyPath + '[')
      );

    if (isObject && Object.keys(value as object).length > 0) {
      elements.push(
        <span
          key={`${keyPath}-key`}
          className={hasChildDiff ? 'config-diff-line' : ''}
        >
          {pad}
          <span
            className="config-key config-toggle"
            onClick={() => toggleCollapse(keyPath)}
          >
            {isCollapsed ? '+ ' : '- '}
            {key}
          </span>
          :{'\n'}
        </span>
      );
      if (!isCollapsed) {
        elements.push(
          ...renderYaml(
            value,
            indent + 1,
            collapsed,
            toggleCollapse,
            keyPath,
            diffPaths
          )
        );
      }
    } else {
      elements.push(
        <span
          key={`${keyPath}-key`}
          className={isDiff ? 'config-diff-line' : ''}
        >
          {pad}
          <span className="config-key">{key}</span>:{' '}
        </span>
      );
      elements.push(
        ...renderYaml(
          value,
          indent + 1,
          collapsed,
          toggleCollapse,
          keyPath,
          diffPaths
        )
      );
      elements.push(<span key={`${keyPath}-nl`}>{'\n'}</span>);
    }
  }

  return elements;
}

// Hook to fetch config data
function useConfigData(basePath: string) {
  return useApi<ConfigData>(`${basePath}/config`);
}

// Single-run config viewer (unchanged behavior)
function SingleConfigView({ basePath }: { basePath: string }) {
  const { data, loading, error } = useConfigData(basePath);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  const toggleCollapse = (path: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  };

  if (loading) return <div className="loading">Loading config...</div>;
  if (error) return <div className="error-msg">{error}</div>;
  if (!data) return null;

  return (
    <div>
      <div className="config-section">
        <h3>Resolved Config</h3>
        <pre className="config-tree">
          {renderYaml(data.config, 0, collapsed, toggleCollapse)}
        </pre>
      </div>

      {data.overrides && (
        <div className="config-section">
          <h3>Overrides</h3>
          <pre className="config-tree">
            {Array.isArray(data.overrides)
              ? data.overrides.length > 0
                ? data.overrides.join('\n')
                : '(none)'
              : JSON.stringify(data.overrides, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

// Multi-run comparison view
function CompareConfigView({
  basePath,
  compareBasePaths,
  compareLabels,
  compareColors,
}: Required<Omit<ConfigViewerProps, 'compareColors'>> & {
  compareColors: string[];
}) {
  const allPaths = [basePath, ...compareBasePaths];
  const primaryData = useConfigData(allPaths[0]);
  const compare1Data = useConfigData(allPaths[1] ?? '');
  const compare2Data = useConfigData(allPaths[2] ?? '');

  const allData = [primaryData, compare1Data, compare2Data].slice(
    0,
    allPaths.length
  );

  const [viewMode, setViewMode] = useState<'side-by-side' | 'diff'>(
    'side-by-side'
  );
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  const toggleCollapse = (path: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  };

  // Compute flattened configs and diff paths
  const { flatConfigs, diffPaths, allKeyPaths } = useMemo(() => {
    const configs = allData
      .map((d) => d.data?.config)
      .filter(Boolean) as Record<string, unknown>[];
    if (configs.length < 2) return { flatConfigs: [], diffPaths: new Set<string>(), allKeyPaths: [] as string[] };

    const flat = configs.map((c) => flattenConfig(c));
    const allKeys = new Set<string>();
    for (const f of flat) {
      for (const k of f.keys()) allKeys.add(k);
    }
    const diffs = new Set<string>();
    for (const key of allKeys) {
      const values = flat.map((f) => (f.has(key) ? f.get(key) : undefined));
      if (!valuesAllEqual(values)) {
        diffs.add(key);
      }
    }
    return {
      flatConfigs: flat,
      diffPaths: diffs,
      allKeyPaths: Array.from(allKeys).sort(),
    };
  }, [allData.map((d) => d.data).join(',')]);

  const anyLoading = allData.some((d) => d.loading);
  const errors = allData
    .map((d, i) => (d.error ? `${compareLabels[i]}: ${d.error}` : null))
    .filter(Boolean);

  if (anyLoading)
    return <div className="loading">Loading configs for comparison...</div>;

  return (
    <div>
      <div className="compare-view-controls">
        <button
          className={`btn btn-sm ${viewMode === 'side-by-side' ? 'btn-primary' : ''}`}
          onClick={() => setViewMode('side-by-side')}
        >
          Side by Side
        </button>
        <button
          className={`btn btn-sm ${viewMode === 'diff' ? 'btn-primary' : ''}`}
          onClick={() => setViewMode('diff')}
        >
          Diff View
        </button>
        <span className="compare-diff-summary">
          {diffPaths.size === 0
            ? 'Configs are identical'
            : `${diffPaths.size} difference${diffPaths.size === 1 ? '' : 's'}`}
        </span>
      </div>

      {errors.length > 0 && (
        <div className="error-msg" style={{ marginBottom: 12 }}>
          {errors.join('; ')}
        </div>
      )}

      {viewMode === 'side-by-side' ? (
        <div
          className="compare-columns"
          style={{
            gridTemplateColumns: `repeat(${allPaths.length}, 1fr)`,
          }}
        >
          {allData.map((d, i) => (
            <div key={i} className="compare-column">
              <div
                className="compare-column-header"
                style={{ borderBottomColor: compareColors[i] }}
              >
                <span
                  className="compare-badge-dot"
                  style={{ background: compareColors[i] }}
                />
                {compareLabels[i]}
              </div>
              {d.data?.config ? (
                <pre className="config-tree config-tree-compact">
                  {renderYaml(
                    d.data.config,
                    0,
                    collapsed,
                    toggleCollapse,
                    `col${i}`,
                    diffPaths
                  )}
                </pre>
              ) : (
                <div className="empty-state" style={{ padding: 20 }}>
                  No config available
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <DiffView
          flatConfigs={flatConfigs}
          allKeyPaths={allKeyPaths}
          diffPaths={diffPaths}
          labels={compareLabels}
          colors={compareColors}
          collapsed={collapsed}
          toggleCollapse={toggleCollapse}
        />
      )}
    </div>
  );
}

// Unified diff view
function DiffView({
  flatConfigs,
  allKeyPaths,
  diffPaths,
  labels,
  colors,
  collapsed,
  toggleCollapse,
}: {
  flatConfigs: Map<string, unknown>[];
  allKeyPaths: string[];
  diffPaths: Set<string>;
  labels: string[];
  colors: string[];
  collapsed: Set<string>;
  toggleCollapse: (path: string) => void;
}) {
  const [showOnlyDiffs, setShowOnlyDiffs] = useState(false);

  // Group key paths by their top-level parent for collapsing
  const groupedPaths = useMemo(() => {
    const groups = new Map<string, string[]>();
    for (const keyPath of allKeyPaths) {
      const topLevel = keyPath.split('.')[0];
      const list = groups.get(topLevel) ?? [];
      list.push(keyPath);
      groups.set(topLevel, list);
    }
    return groups;
  }, [allKeyPaths]);

  return (
    <div className="config-section">
      <div style={{ marginBottom: 8 }}>
        <label style={{ fontSize: 13, color: 'var(--text-muted)', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={showOnlyDiffs}
            onChange={(e) => setShowOnlyDiffs(e.target.checked)}
            style={{ marginRight: 6 }}
          />
          Show only differences
        </label>
      </div>
      <div className="config-diff-table">
        <div className="config-diff-header">
          <div className="config-diff-key-col">Key</div>
          {labels.map((label, i) => (
            <div
              key={i}
              className="config-diff-val-col"
              style={{ borderTopColor: colors[i] }}
            >
              <span
                className="compare-badge-dot"
                style={{ background: colors[i] }}
              />
              {label.split('/').pop()}
            </div>
          ))}
        </div>
        <div className="config-diff-body">
          {Array.from(groupedPaths.entries()).map(([group, paths]) => {
            const groupHasDiff = paths.some((p) => diffPaths.has(p));
            const isCollapsed = collapsed.has(`diff-${group}`);

            if (showOnlyDiffs && !groupHasDiff) return null;

            const filteredPaths = showOnlyDiffs
              ? paths.filter((p) => diffPaths.has(p))
              : paths;

            return (
              <div key={group}>
                <div
                  className={`config-diff-group ${groupHasDiff ? 'has-diff' : ''}`}
                  onClick={() => toggleCollapse(`diff-${group}`)}
                >
                  <span className="config-toggle">
                    {isCollapsed ? '+ ' : '- '}
                  </span>
                  {group}
                  {groupHasDiff && (
                    <span className="config-diff-count">
                      {paths.filter((p) => diffPaths.has(p)).length} diff
                    </span>
                  )}
                </div>
                {!isCollapsed &&
                  filteredPaths.map((keyPath) => {
                    const isDiff = diffPaths.has(keyPath);
                    const shortKey = keyPath.slice(group.length + 1) || keyPath;

                    return (
                      <div
                        key={keyPath}
                        className={`config-diff-row ${isDiff ? 'diff' : ''}`}
                      >
                        <div className="config-diff-key-col" title={keyPath}>
                          {shortKey}
                        </div>
                        {flatConfigs.map((flat, i) => {
                          const has = flat.has(keyPath);
                          const val = has ? flat.get(keyPath) : undefined;
                          return (
                            <div
                              key={i}
                              className={`config-diff-val-col ${isDiff ? 'highlight' : ''}`}
                              style={
                                isDiff
                                  ? {
                                      borderLeftColor: colors[i],
                                    }
                                  : undefined
                              }
                            >
                              {has ? (
                                <span className={valueClass(val)}>
                                  {valueToString(val)}
                                </span>
                              ) : (
                                <span className="config-null">(absent)</span>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    );
                  })}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function valueClass(val: unknown): string {
  if (val === null || val === undefined) return 'config-null';
  if (typeof val === 'number') return 'config-num';
  if (typeof val === 'boolean') return 'config-bool';
  return 'config-str';
}

export function ConfigViewer({
  basePath,
  compareBasePaths,
  compareLabels,
  compareColors,
}: ConfigViewerProps) {
  if (compareBasePaths && compareLabels && compareColors) {
    return (
      <CompareConfigView
        basePath={basePath}
        compareBasePaths={compareBasePaths}
        compareLabels={compareLabels}
        compareColors={compareColors}
      />
    );
  }
  return <SingleConfigView basePath={basePath} />;
}
