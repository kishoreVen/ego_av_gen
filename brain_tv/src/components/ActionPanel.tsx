import { useState, useEffect, useRef, useMemo } from 'react';
import { useApi, postApi } from '../hooks/useApi';
import type { CheckpointInfo, ProcessInfo, ProcessOutput } from '../types';

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

export function ActionPanel({
  experiment,
  timestamp,
  basePath,
}: {
  experiment: string;
  timestamp: string;
  basePath: string;
}) {
  const { data: checkpoints } = useApi<CheckpointInfo[]>(
    `${basePath}/checkpoints`
  );
  const { data: processes, refetch: refetchProcesses } = useApi<ProcessInfo[]>(
    '/api/actions/processes',
    3000
  );

  return (
    <div>
      <div className="split-layout">
        <TrainAction
          experiment={experiment}
          timestamp={timestamp}
          checkpoints={checkpoints ?? []}
          onStarted={refetchProcesses}
        />
        <InferenceAction
          checkpoints={checkpoints ?? []}
          onStarted={refetchProcesses}
        />
      </div>

      {processes && processes.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <h3 style={{ fontSize: 16, marginBottom: 12 }}>Active Processes</h3>
          {processes.map((p) => (
            <ProcessCard key={p.id} process={p} />
          ))}
        </div>
      )}
    </div>
  );
}

function TrainAction({
  experiment,
  timestamp,
  checkpoints,
  onStarted,
}: {
  experiment: string;
  timestamp: string;
  checkpoints: CheckpointInfo[];
  onStarted: () => void;
}) {
  const [recipe, setRecipe] = useState('dummy/training/dummy');
  const [resumeFrom, setResumeFrom] = useState('');
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
    const args = [`projects=${recipe}`];
    if (resumeFrom) {
      args.push(`projects.config.resume_checkpoint.path=${resumeFrom}`);
    }
    args.push(...overridesList);
    return buildCommand(args);
  }, [recipe, resumeFrom, overridesList]);

  const handleTrain = async () => {
    setSubmitting(true);
    try {
      await postApi('/api/actions/train', {
        recipe,
        checkpointPath: resumeFrom || undefined,
        overrides: overridesList,
      });
      onStarted();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to start training');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="action-section">
      <h3>Start Training</h3>
      <div className="form-group">
        <label>Recipe</label>
        <input
          value={recipe}
          onChange={(e) => setRecipe(e.target.value)}
          placeholder="training/flow_matching"
        />
      </div>
      <div className="form-group">
        <label>Resume from checkpoint (optional)</label>
        <select
          value={resumeFrom}
          onChange={(e) => setResumeFrom(e.target.value)}
        >
          <option value="">From scratch</option>
          {checkpoints.map((cp) => (
            <option key={cp.name} value={cp.path}>
              {cp.name}
            </option>
          ))}
        </select>
      </div>
      <div className="form-group">
        <label>Overrides (one per line)</label>
        <textarea
          value={overrides}
          onChange={(e) => setOverrides(e.target.value)}
          placeholder={`projects.config.max_steps=20000\nprojects.config.learning_rate=0.00005`}
        />
      </div>
      <CommandPreview command={command} />
      <button
        className="btn btn-primary"
        onClick={handleTrain}
        disabled={submitting}
        style={{ marginTop: 12 }}
      >
        {submitting ? 'Starting...' : 'Start Training'}
      </button>
    </div>
  );
}

function InferenceAction({
  checkpoints,
  onStarted,
}: {
  checkpoints: CheckpointInfo[];
  onStarted: () => void;
}) {
  const [recipe, setRecipe] = useState('dummy/inference/example');
  const [checkpoint, setCheckpoint] = useState('');
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
    const args = [`projects=${recipe}`];
    if (checkpoint) {
      args.push(`projects.config.model_checkpoint.path=${checkpoint}`);
    }
    args.push(...overridesList);
    return buildCommand(args);
  }, [recipe, checkpoint, overridesList]);

  const handleInference = async () => {
    if (!checkpoint) {
      alert('Select a checkpoint');
      return;
    }
    setSubmitting(true);
    try {
      await postApi('/api/actions/inference', {
        recipe,
        checkpointPath: checkpoint,
        overrides: overridesList,
      });
      onStarted();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to start inference');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="action-section">
      <h3>Run Inference</h3>
      <div className="form-group">
        <label>Recipe</label>
        <input
          value={recipe}
          onChange={(e) => setRecipe(e.target.value)}
          placeholder="inference/flow_matching"
        />
      </div>
      <div className="form-group">
        <label>Checkpoint</label>
        <select
          value={checkpoint}
          onChange={(e) => setCheckpoint(e.target.value)}
        >
          <option value="">Select checkpoint...</option>
          {checkpoints.map((cp) => (
            <option key={cp.name} value={cp.path}>
              {cp.name}
            </option>
          ))}
        </select>
      </div>
      <div className="form-group">
        <label>Overrides (one per line)</label>
        <textarea
          value={overrides}
          onChange={(e) => setOverrides(e.target.value)}
          placeholder="projects.config.num_sampling_steps=100"
        />
      </div>
      <CommandPreview command={command} />
      <button
        className="btn btn-primary"
        onClick={handleInference}
        disabled={submitting || !checkpoint}
        style={{ marginTop: 12 }}
      >
        {submitting ? 'Starting...' : 'Run Inference'}
      </button>
    </div>
  );
}

function ProcessCard({ process }: { process: ProcessInfo }) {
  const [expanded, setExpanded] = useState(false);
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
          <span
            className={`badge ${process.status === 'running' ? 'badge-green' : 'badge-dim'}`}
          >
            {process.status}
          </span>
          {process.exitCode !== null && (
            <span className="mono" style={{ marginLeft: 8, fontSize: 12 }}>
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
          {output.length > 0
            ? output.join('')
            : '(waiting for output...)'}
        </div>
      )}
    </div>
  );
}
