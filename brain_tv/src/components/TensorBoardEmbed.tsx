import { useState, useEffect, useCallback } from 'react';
import { postApi } from '../hooks/useApi';

interface TbResponse {
  port: number;
  status: string;
}

export function TensorBoardEmbed({
  experiment,
  timestamp,
  compareRuns,
}: {
  experiment: string;
  timestamp: string;
  compareRuns?: { experiment: string; timestamp: string }[];
}) {
  const [port, setPort] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const compareKey = compareRuns
    ?.map((r) => `${r.experiment}/${r.timestamp}`)
    .join(',') ?? '';

  const startTb = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await postApi<TbResponse>('/api/tensorboard/start', {
        experiment,
        timestamp,
        compareRuns,
      });
      setPort(res.port);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to start TensorBoard'
      );
    } finally {
      setLoading(false);
    }
  }, [experiment, timestamp, compareKey]);

  useEffect(() => {
    startTb();
    return () => {
      postApi('/api/tensorboard/stop', {}).catch(() => {});
    };
  }, [startTb]);

  if (error) {
    return (
      <div>
        <div className="error-msg" style={{ marginBottom: 12 }}>
          {error}
        </div>
        <p style={{ color: 'var(--text-muted)', fontSize: 14 }}>
          Make sure TensorBoard is installed:{' '}
          <code>pip install tensorboard</code>
        </p>
        <button className="btn btn-primary" onClick={startTb} style={{ marginTop: 8 }}>
          Retry
        </button>
      </div>
    );
  }

  if (loading || !port) {
    return (
      <div className="tb-container">
        <div className="tb-loading">Starting TensorBoard...</div>
      </div>
    );
  }

  const tbUrl = `http://${window.location.hostname}:${port}`;

  return (
    <div>
      <div style={{ marginBottom: 8, fontSize: 13, color: 'var(--text-muted)' }}>
        TensorBoard running at{' '}
        <a href={tbUrl} target="_blank" rel="noreferrer" style={{ color: 'var(--accent)' }}>
          {tbUrl}
        </a>
      </div>
      <div className="tb-container">
        <iframe src={tbUrl} title="TensorBoard" />
      </div>
    </div>
  );
}
