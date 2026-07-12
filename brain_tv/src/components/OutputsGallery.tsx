import { useState, useMemo } from 'react';
import { useApi } from '../hooks/useApi';
import type { FileEntry } from '../types';

interface OutputsGalleryProps {
  basePath: string;
  compareBasePaths?: string[];
  compareLabels?: string[];
  compareColors?: string[];
}

function getFileType(
  path: string
): 'image' | 'video' | 'json' | 'yaml' | 'text' | 'unknown' {
  const ext = path.split('.').pop()?.toLowerCase();
  if (!ext) return 'unknown';
  if (['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'].includes(ext))
    return 'image';
  if (['mp4', 'webm', 'mov', 'avi'].includes(ext)) return 'video';
  if (ext === 'json' || ext === 'jsonl') return 'json';
  if (ext === 'yaml' || ext === 'yml') return 'yaml';
  if (['txt', 'log', 'csv'].includes(ext)) return 'text';
  return 'unknown';
}

function FilePreview({
  entry,
  basePath,
}: {
  entry: FileEntry;
  basePath: string;
}) {
  const type = getFileType(entry.path);
  const fileUrl = `${basePath}/files/visualizations/${entry.path}`;

  switch (type) {
    case 'image':
      return (
        <div className="output-card">
          <img src={fileUrl} alt={entry.path} loading="lazy" />
          <div className="output-card-label">
            {entry.path.split('/').pop()}
          </div>
        </div>
      );
    case 'video':
      return (
        <div className="output-card">
          <video controls preload="metadata">
            <source src={fileUrl} />
          </video>
          <div className="output-card-label">
            {entry.path.split('/').pop()}
          </div>
        </div>
      );
    case 'json':
    case 'yaml':
    case 'text':
      return <TextPreview entry={entry} fileUrl={fileUrl} />;
    default:
      return (
        <div className="output-card">
          <div className="output-json" style={{ color: 'var(--text-dim)' }}>
            {entry.path.split('/').pop()}
            <br />
            <span style={{ fontSize: 11 }}>
              ({(entry.size / 1024).toFixed(1)} KB)
            </span>
          </div>
        </div>
      );
  }
}

function TextPreview({
  entry,
  fileUrl,
}: {
  entry: FileEntry;
  fileUrl: string;
}) {
  const [content, setContent] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  const load = async () => {
    if (loaded) return;
    try {
      const res = await fetch(fileUrl);
      const text = await res.text();
      setContent(text);
    } catch {
      setContent('(failed to load)');
    }
    setLoaded(true);
  };

  return (
    <div className="output-card" onClick={load} style={{ cursor: 'pointer' }}>
      <div className="output-json">
        {content ?? (
          <span style={{ color: 'var(--text-dim)' }}>
            Click to load {entry.path.split('/').pop()}
          </span>
        )}
        {content && (
          <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{content}</pre>
        )}
      </div>
      <div className="output-card-label">{entry.path.split('/').pop()}</div>
    </div>
  );
}

// Single-run outputs (unchanged behavior)
function SingleOutputsGallery({ basePath }: { basePath: string }) {
  const {
    data: files,
    loading,
    error,
  } = useApi<FileEntry[]>(`${basePath}/outputs`);

  const grouped = useMemo(() => {
    if (!files) return new Map<string, FileEntry[]>();
    const groups = new Map<string, FileEntry[]>();
    const fileEntries = files.filter((f) => !f.isDir);

    for (const entry of fileEntries) {
      const topDir = entry.path.split('/')[0] ?? 'root';
      const list = groups.get(topDir) ?? [];
      list.push(entry);
      groups.set(topDir, list);
    }
    return groups;
  }, [files]);

  if (loading) return <div className="loading">Loading outputs...</div>;
  if (error) return <div className="error-msg">{error}</div>;
  if (!files || files.length === 0) {
    return (
      <div className="empty-state">
        <h3>No outputs</h3>
        <p>
          No visualization outputs found. Configure{' '}
          <code>output_save_interval</code> in your training monitor or run
          inference to generate outputs.
        </p>
      </div>
    );
  }

  return (
    <div>
      {Array.from(grouped.entries()).map(([stepDir, entries]) => (
        <div key={stepDir} className="outputs-step">
          <h4>{stepDir}</h4>
          <div className="outputs-grid">
            {entries.map((entry) => (
              <FilePreview
                key={entry.path}
                entry={entry}
                basePath={basePath}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

// Multi-run comparison outputs
function CompareOutputsGallery({
  basePath,
  compareBasePaths,
  compareLabels,
  compareColors,
}: Required<Omit<OutputsGalleryProps, 'compareColors'>> & {
  compareColors: string[];
}) {
  const allPaths = [basePath, ...compareBasePaths];

  const primaryFiles = useApi<FileEntry[]>(`${allPaths[0]}/outputs`);
  const compare1Files = useApi<FileEntry[]>(
    allPaths[1] ? `${allPaths[1]}/outputs` : ''
  );
  const compare2Files = useApi<FileEntry[]>(
    allPaths[2] ? `${allPaths[2]}/outputs` : ''
  );

  const allFiles = [primaryFiles, compare1Files, compare2Files].slice(
    0,
    allPaths.length
  );

  // Collect all unique file paths across runs, grouped by step dir
  const { stepDirs, filesByStep } = useMemo(() => {
    const allEntries = allFiles.map(
      (f) => f.data?.filter((e) => !e.isDir) ?? []
    );

    // Collect all unique step dirs
    const dirs = new Set<string>();
    for (const entries of allEntries) {
      for (const entry of entries) {
        dirs.add(entry.path.split('/')[0] ?? 'root');
      }
    }
    const sortedDirs = Array.from(dirs).sort();

    // For each step dir, collect unique filenames and map to entries per run
    const byStep = new Map<
      string,
      { filename: string; entries: (FileEntry | null)[] }[]
    >();

    for (const dir of sortedDirs) {
      const filenames = new Set<string>();
      const entriesByRun = allEntries.map((entries) => {
        const map = new Map<string, FileEntry>();
        for (const e of entries) {
          const topDir = e.path.split('/')[0] ?? 'root';
          if (topDir === dir) {
            const filename = e.path.split('/').slice(1).join('/') || e.path;
            map.set(filename, e);
            filenames.add(filename);
          }
        }
        return map;
      });

      const rows = Array.from(filenames)
        .sort()
        .map((filename) => ({
          filename,
          entries: entriesByRun.map((m) => m.get(filename) ?? null),
        }));

      byStep.set(dir, rows);
    }

    return { stepDirs: sortedDirs, filesByStep: byStep };
  }, [allFiles.map((f) => f.data).join(',')]);

  const anyLoading = allFiles.some((f) => f.loading);
  if (anyLoading)
    return <div className="loading">Loading outputs for comparison...</div>;

  const anyData = allFiles.some((f) => f.data && f.data.length > 0);
  if (!anyData) {
    return (
      <div className="empty-state">
        <h3>No outputs</h3>
        <p>None of the selected runs have visualization outputs.</p>
      </div>
    );
  }

  return (
    <div>
      {/* Column headers */}
      <div
        className="compare-outputs-header"
        style={{
          gridTemplateColumns: `repeat(${allPaths.length}, 1fr)`,
        }}
      >
        {compareLabels.map((label, i) => (
          <div
            key={i}
            className="compare-column-header"
            style={{ borderBottomColor: compareColors[i] }}
          >
            <span
              className="compare-badge-dot"
              style={{ background: compareColors[i] }}
            />
            {label}
          </div>
        ))}
      </div>

      {stepDirs.map((dir) => {
        const rows = filesByStep.get(dir) ?? [];
        if (rows.length === 0) return null;

        return (
          <div key={dir} className="outputs-step">
            <h4>{dir}</h4>
            {rows.map(({ filename, entries }) => (
              <div
                key={filename}
                className="compare-outputs-row"
                style={{
                  gridTemplateColumns: `repeat(${allPaths.length}, 1fr)`,
                }}
              >
                {entries.map((entry, i) => (
                  <div key={i} className="compare-outputs-cell">
                    {entry ? (
                      <FilePreview entry={entry} basePath={allPaths[i]} />
                    ) : (
                      <div className="output-card">
                        <div
                          className="output-json"
                          style={{
                            color: 'var(--text-dim)',
                            textAlign: 'center',
                            padding: 20,
                          }}
                        >
                          (no output)
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
}

export function OutputsGallery({
  basePath,
  compareBasePaths,
  compareLabels,
  compareColors,
}: OutputsGalleryProps) {
  if (compareBasePaths && compareLabels && compareColors && compareBasePaths.length > 0) {
    return (
      <CompareOutputsGallery
        basePath={basePath}
        compareBasePaths={compareBasePaths}
        compareLabels={compareLabels}
        compareColors={compareColors}
      />
    );
  }
  return <SingleOutputsGallery basePath={basePath} />;
}
