import { useState, useEffect, useRef, useMemo } from 'react';
import { useApi, postApi } from '../hooks/useApi';
import type { ProcessInfo, ProcessOutput } from '../types';

interface RecipeTreeNode {
  name: string;
  type: 'dir' | 'file';
  path?: string;
  children?: RecipeTreeNode[];
}

function buildCommand(args: string[]): string {
  return ['python', '-m', 'algos.brain_factory.main', ...args].join(' \\\n  ');
}

function CommandPreview({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);

  const copy = () => {
    navigator.clipboard.writeText(command.replace(/ \\\n  /g, ' '));
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="command-preview">
      <div className="command-preview-header">
        <span className="command-preview-label">Command</span>
        <button className="btn btn-sm" onClick={copy}>
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      <pre className="command-preview-code">{command}</pre>
    </div>
  );
}

function flattenTree(
  nodes: RecipeTreeNode[],
  prefix: string[] = []
): { path: string; name: string; breadcrumb: string[] }[] {
  const result: { path: string; name: string; breadcrumb: string[] }[] = [];
  for (const node of nodes) {
    if (node.type === 'file' && node.path) {
      result.push({ path: node.path, name: node.name, breadcrumb: prefix });
    }
    if (node.type === 'dir' && node.children) {
      result.push(...flattenTree(node.children, [...prefix, node.name]));
    }
  }
  return result;
}

function RecipeTreePicker({
  nodes,
  selected,
  onSelect,
}: {
  nodes: RecipeTreeNode[];
  selected: string;
  onSelect: (path: string) => void;
}) {
  const [search, setSearch] = useState('');
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  const allRecipes = useMemo(() => flattenTree(nodes), [nodes]);

  const filtered = useMemo(() => {
    if (!search) return allRecipes;
    const q = search.toLowerCase();
    return allRecipes.filter(
      (r) =>
        r.name.toLowerCase().includes(q) ||
        r.path.toLowerCase().includes(q) ||
        r.breadcrumb.some((b) => b.toLowerCase().includes(q))
    );
  }, [allRecipes, search]);

  // Group by top-level breadcrumb
  const grouped = useMemo(() => {
    const map = new Map<string, typeof filtered>();
    for (const r of filtered) {
      const key = r.breadcrumb[0] ?? '';
      const list = map.get(key) ?? [];
      list.push(r);
      map.set(key, list);
    }
    return map;
  }, [filtered]);

  const toggleGroup = (group: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(group)) next.delete(group);
      else next.add(group);
      return next;
    });
  };

  return (
    <div className="recipe-picker">
      <input
        type="text"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder="Search recipes..."
        className="recipe-picker-search"
      />
      <div className="recipe-picker-list">
        {filtered.length === 0 && (
          <div className="recipe-picker-empty">No matching recipes</div>
        )}
        {Array.from(grouped.entries()).map(([group, items]) => (
          <div key={group}>
            {group && (
              <div
                className="recipe-picker-group"
                onClick={() => toggleGroup(group)}
              >
                <span className="recipe-tree-arrow">
                  {collapsed.has(group) ? '\u25B8' : '\u25BE'}
                </span>
                {group}
                <span className="recipe-picker-count">{items.length}</span>
              </div>
            )}
            {!collapsed.has(group) &&
              items.map((r) => (
                <div
                  key={r.path}
                  className={`recipe-picker-item ${selected === r.path ? 'selected' : ''}`}
                  onClick={() => onSelect(r.path)}
                >
                  <span className="recipe-picker-name">{r.name}</span>
                  <span className="recipe-picker-path">{r.path}</span>
                </div>
              ))}
          </div>
        ))}
      </div>
    </div>
  );
}

export function ProcessesPanel() {
  const { data: processes, refetch: refetchProcesses } = useApi<ProcessInfo[]>(
    '/api/actions/processes',
    2000
  );
  const { data: recipeTree } = useApi<RecipeTreeNode[]>('/api/actions/recipes');

  const [recipe, setRecipe] = useState('');
  const [overrides, setOverrides] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const overridesList = useMemo(
    () =>
      overrides
        .split('\n')
        .map((l) => l.trim())
        .filter(Boolean),
    [overrides]
  );

  const command = useMemo(() => {
    if (!recipe) return '';
    const args = [`projects=${recipe}`, ...overridesList];
    return buildCommand(args);
  }, [recipe, overridesList]);

  const handleRun = async () => {
    if (!recipe) return;
    setSubmitting(true);
    try {
      await postApi('/api/actions/train', {
        recipe,
        overrides: overridesList,
      });
      refetchProcesses();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to start');
    } finally {
      setSubmitting(false);
    }
  };

  const running = processes?.filter((p) => p.status === 'running') ?? [];
  const exited = processes?.filter((p) => p.status === 'exited') ?? [];

  return (
    <div>
      {/* Launch new run */}
      <div className="action-section">
        <h3>Launch Run</h3>
        <div className="form-group">
          <label>Recipe</label>
          {recipeTree && recipeTree.length > 0 ? (
            <RecipeTreePicker
              nodes={recipeTree}
              selected={recipe}
              onSelect={setRecipe}
            />
          ) : (
            <div style={{ color: 'var(--text-dim)', fontSize: 13 }}>
              Loading recipes...
            </div>
          )}
          {recipe && (
            <div className="recipe-selected">
              {recipe}
            </div>
          )}
        </div>
        <div className="form-group">
          <label>Overrides (one per line)</label>
          <textarea
            value={overrides}
            onChange={(e) => setOverrides(e.target.value)}
            placeholder={`projects.config.max_steps=10000\nprojects.config.learning_rate=0.0001`}
          />
        </div>
        {recipe && <CommandPreview command={command} />}
        <button
          className="btn btn-primary"
          onClick={handleRun}
          disabled={submitting || !recipe}
          style={{ marginTop: 12 }}
        >
          {submitting ? 'Starting...' : 'Run'}
        </button>
      </div>

      {/* Running processes */}
      {running.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <h3 style={{ fontSize: 16, marginBottom: 12 }}>
            Running ({running.length})
          </h3>
          {running.map((p) => (
            <ProcessCard key={p.id} process={p} onKilled={refetchProcesses} />
          ))}
        </div>
      )}

      {/* Exited processes */}
      {exited.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <h3 style={{ fontSize: 16, marginBottom: 12, color: 'var(--text-muted)' }}>
            Completed ({exited.length})
          </h3>
          {exited.map((p) => (
            <ProcessCard key={p.id} process={p} onKilled={refetchProcesses} />
          ))}
        </div>
      )}

      {(!processes || processes.length === 0) && (
        <div
          className="empty-state"
          style={{ padding: '30px', marginTop: 16 }}
        >
          <p>No active processes. Select a recipe and launch a run.</p>
        </div>
      )}
    </div>
  );
}

function ProcessCard({
  process,
  onKilled,
}: {
  process: ProcessInfo;
  onKilled: () => void;
}) {
  const [expanded, setExpanded] = useState(process.status === 'running');
  const [output, setOutput] = useState<string[]>([]);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!expanded) return;
    const fetchOutput = async () => {
      try {
        const res = await fetch(
          `/api/actions/processes/${process.id}?since=0`
        );
        const data: ProcessOutput = await res.json();
        setOutput(data.output);
        if (logRef.current) {
          logRef.current.scrollTop = logRef.current.scrollHeight;
        }
      } catch {
        // ignore
      }
    };
    fetchOutput();
    const id = setInterval(fetchOutput, 2000);
    return () => clearInterval(id);
  }, [expanded, process.id]);

  const handleKill = async () => {
    await postApi(`/api/actions/processes/${process.id}/kill`, {});
    onKilled();
  };

  return (
    <div className="action-section" style={{ marginBottom: 12 }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <div>
          <strong className="mono">{process.id}</strong>{' '}
          <span className="mono" style={{ color: 'var(--text-muted)', marginLeft: 4 }}>
            {process.recipe}
          </span>{' '}
          <span
            className={`badge ${process.status === 'running' ? 'badge-green' : 'badge-dim'}`}
          >
            {process.status}
          </span>
          {process.exitCode !== null && (
            <span
              className="mono"
              style={{
                marginLeft: 8,
                fontSize: 12,
                color:
                  process.exitCode === 0 ? 'var(--green)' : 'var(--red)',
              }}
            >
              exit: {process.exitCode}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-sm" onClick={() => setExpanded(!expanded)}>
            {expanded ? 'Hide Log' : 'Show Log'}
          </button>
          {process.status === 'running' && (
            <button className="btn btn-sm btn-danger" onClick={handleKill}>
              Kill
            </button>
          )}
        </div>
      </div>
      {expanded && (
        <div className="process-log" ref={logRef} style={{ marginTop: 12 }}>
          {output.length > 0 ? output.join('') : '(waiting for output...)'}
        </div>
      )}
    </div>
  );
}
