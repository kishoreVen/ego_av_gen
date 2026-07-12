import { useState } from 'react';
import { useApi } from '../hooks/useApi';
import type { CheckpointInfo } from '../types';

export function CheckpointList({ basePath }: { basePath: string }) {
  const { data: checkpoints, loading, error } = useApi<CheckpointInfo[]>(
    `${basePath}/checkpoints`
  );
  const [expandedConfig, setExpandedConfig] = useState<string | null>(null);
  const [configData, setConfigData] = useState<string | null>(null);

  if (loading) return <div className="loading">Loading checkpoints...</div>;
  if (error) return <div className="error-msg">{error}</div>;
  if (!checkpoints || checkpoints.length === 0) {
    return (
      <div className="empty-state" style={{ padding: '30px' }}>
        <h3>No checkpoints</h3>
        <p>No checkpoints saved for this run yet.</p>
      </div>
    );
  }

  const viewConfig = async (cpName: string) => {
    if (expandedConfig === cpName) {
      setExpandedConfig(null);
      return;
    }
    try {
      const res = await fetch(
        `${basePath}/files/checkpoints/${cpName}/recipe_config.yaml`
      );
      if (res.ok) {
        const text = await res.text();
        setConfigData(text);
      } else {
        setConfigData('(config file not found)');
      }
      setExpandedConfig(cpName);
    } catch {
      setConfigData('(failed to load config)');
      setExpandedConfig(cpName);
    }
  };

  return (
    <div>
      <h3 style={{ fontSize: 16, marginBottom: 12 }}>
        Checkpoints ({checkpoints.length})
      </h3>
      {checkpoints.map((cp) => (
        <div key={cp.name} className="checkpoint-card">
          <h4>{cp.name}</h4>
          <div className="checkpoint-files">
            {cp.files.map((f) => (
              <span key={f.name}>
                {f.name} ({f.size})
              </span>
            ))}
          </div>
          <div style={{ marginTop: 8 }}>
            <button
              className="btn btn-sm"
              onClick={() => viewConfig(cp.name)}
            >
              {expandedConfig === cp.name ? 'Hide Config' : 'View Config'}
            </button>
          </div>
          {expandedConfig === cp.name && configData && (
            <pre
              className="config-tree"
              style={{ marginTop: 8, maxHeight: 300, overflow: 'auto' }}
            >
              {configData}
            </pre>
          )}
        </div>
      ))}
    </div>
  );
}
