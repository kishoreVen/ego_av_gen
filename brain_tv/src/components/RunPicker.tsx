import { useState, useRef, useEffect, useMemo } from 'react';
import { useApi } from '../hooks/useApi';
import type { RunInfo } from '../types';

interface RunPickerProps {
  excludeKeys: string[]; // "experiment/timestamp" strings to exclude
  onSelect: (experiment: string, timestamp: string) => void;
  onClose: () => void;
}

export function RunPicker({ excludeKeys, onSelect, onClose }: RunPickerProps) {
  const { data: runs } = useApi<RunInfo[]>('/api/runs');
  const [search, setSearch] = useState('');
  const ref = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [onClose]);

  const excludeSet = useMemo(() => new Set(excludeKeys), [excludeKeys]);

  const filtered = useMemo(() => {
    if (!runs) return [];
    return runs.filter((r) => {
      const key = `${r.experiment}/${r.timestamp}`;
      if (excludeSet.has(key)) return false;
      if (!search) return true;
      const q = search.toLowerCase();
      return (
        r.experiment.toLowerCase().includes(q) ||
        r.timestamp.toLowerCase().includes(q)
      );
    });
  }, [runs, search, excludeSet]);

  const grouped = useMemo(() => {
    const map = new Map<string, RunInfo[]>();
    for (const run of filtered) {
      const list = map.get(run.experiment) ?? [];
      list.push(run);
      map.set(run.experiment, list);
    }
    return map;
  }, [filtered]);

  return (
    <div className="run-picker" ref={ref}>
      <input
        ref={inputRef}
        type="text"
        className="run-picker-search"
        placeholder="Search runs..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />
      <div className="run-picker-list">
        {filtered.length === 0 ? (
          <div className="run-picker-empty">No matching runs</div>
        ) : (
          Array.from(grouped.entries()).map(([experiment, expRuns]) => (
            <div key={experiment}>
              <div className="run-picker-group">{experiment}</div>
              {expRuns.map((run) => (
                <div
                  key={`${run.experiment}/${run.timestamp}`}
                  className="run-picker-item"
                  onClick={() => onSelect(run.experiment, run.timestamp)}
                >
                  <span className="run-picker-name">{run.timestamp}</span>
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
                    {run.lastStep !== null && (
                      <span className="badge badge-dim">
                        step {run.lastStep.toLocaleString()}
                      </span>
                    )}
                  </span>
                </div>
              ))}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
