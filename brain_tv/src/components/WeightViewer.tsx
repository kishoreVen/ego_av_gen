import { useState, useRef, useEffect, useCallback } from 'react';
import type { WeightTensorInfo, WeightHeatmapData } from '../types';

interface WeightViewerProps {
  basePath: string;
  checkpointName: string;
  moduleName: string;
}

/** Diverging blue-white-red colormap centered at 0. */
function valueToColor(
  value: number,
  absMax: number
): [number, number, number] {
  if (absMax === 0) return [255, 255, 255];
  const t = Math.max(-1, Math.min(1, value / absMax));

  if (t < 0) {
    const s = 1 + t;
    return [
      Math.round(30 + 225 * s),
      Math.round(60 + 195 * s),
      Math.round(200 + 55 * s),
    ];
  } else {
    return [255, Math.round(255 - 200 * t), Math.round(255 - 215 * t)];
  }
}

function HeatmapCanvas({ data }: { data: WeightHeatmapData }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { values, min, max } = data;
    const rows = values.length;
    const cols = values[0]?.length ?? 0;
    if (rows === 0 || cols === 0) return;

    const pixelSize = Math.max(
      1,
      Math.min(4, Math.floor(220 / Math.max(rows, cols)))
    );
    canvas.width = cols * pixelSize;
    canvas.height = rows * pixelSize;

    const absMax = Math.max(Math.abs(min), Math.abs(max));
    const imageData = ctx.createImageData(canvas.width, canvas.height);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const [red, green, blue] = valueToColor(values[r][c], absMax);
        for (let py = 0; py < pixelSize; py++) {
          for (let px = 0; px < pixelSize; px++) {
            const idx =
              ((r * pixelSize + py) * canvas.width + (c * pixelSize + px)) * 4;
            imageData.data[idx] = red;
            imageData.data[idx + 1] = green;
            imageData.data[idx + 2] = blue;
            imageData.data[idx + 3] = 255;
          }
        }
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }, [data]);

  return (
    <div>
      <canvas
        ref={canvasRef}
        style={{
          width: '100%',
          imageRendering: 'pixelated',
          border: '1px solid var(--border)',
          borderRadius: 4,
        }}
      />
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: 10,
          color: 'var(--text-muted)',
          marginTop: 2,
          fontFamily: 'var(--font-mono)',
        }}
      >
        <span style={{ color: '#1e3cc8' }}>{data.min.toFixed(3)}</span>
        <span>0</span>
        <span style={{ color: '#f85149' }}>{data.max.toFixed(3)}</span>
      </div>
    </div>
  );
}

export function WeightViewer({
  basePath,
  checkpointName,
  moduleName,
}: WeightViewerProps) {
  const [allTensors, setAllTensors] = useState<WeightTensorInfo[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [heatmaps, setHeatmaps] = useState<Map<string, WeightHeatmapData>>(
    new Map()
  );
  const [loadingTensor, setLoadingTensor] = useState<string | null>(null);

  // Fetch all tensor metadata, filter to this module
  useEffect(() => {
    setLoading(true);
    setHeatmaps(new Map());
    fetch(`${basePath}/checkpoints/${checkpointName}/weights`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: WeightTensorInfo[]) => {
        // Filter to tensors belonging to this module (direct params only)
        const prefix = moduleName ? moduleName + '.' : '';
        const filtered = data.filter((t) => {
          if (!moduleName) return false;
          if (!t.name.startsWith(prefix)) return false;
          // Only direct params (e.g. "conv_in.weight" not "conv_in.sub.weight")
          const suffix = t.name.slice(prefix.length);
          return !suffix.includes('.');
        });
        setAllTensors(filtered);
        setLoading(false);

        // Auto-fetch heatmap for first tensor (usually weight)
        if (filtered.length > 0) {
          fetchHeatmap(filtered[0].name);
        }
      })
      .catch(() => {
        setAllTensors(null);
        setLoading(false);
      });
  }, [basePath, checkpointName, moduleName]);

  const fetchHeatmap = useCallback(
    async (name: string) => {
      if (heatmaps.has(name)) return;
      setLoadingTensor(name);
      try {
        const res = await fetch(
          `${basePath}/checkpoints/${checkpointName}/weights/${name}`
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data: WeightHeatmapData = await res.json();
        setHeatmaps((prev) => new Map(prev).set(name, data));
      } catch {
        // ignore
      }
      setLoadingTensor(null);
    },
    [basePath, checkpointName, heatmaps]
  );

  if (loading) return <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>Loading weights...</div>;
  if (!allTensors || allTensors.length === 0) {
    return (
      <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>
        No weight tensors
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {allTensors.map((t) => {
        const shortName = t.name.split('.').pop() ?? t.name;
        const heatmap = heatmaps.get(t.name);
        return (
          <div
            key={t.name}
            onClick={() => fetchHeatmap(t.name)}
            style={{ cursor: 'pointer' }}
          >
            <div
              style={{
                fontSize: 11,
                fontFamily: 'var(--font-mono)',
                color: 'var(--accent)',
                marginBottom: 2,
              }}
            >
              {shortName}{' '}
              <span style={{ color: 'var(--text-muted)' }}>
                [{t.shape.join(', ')}]
              </span>
            </div>
            <div
              style={{
                fontSize: 10,
                fontFamily: 'var(--font-mono)',
                color: 'var(--text-dim)',
                marginBottom: 4,
              }}
            >
              min {t.min.toFixed(3)} · max {t.max.toFixed(3)} · std{' '}
              {t.std.toFixed(3)}
            </div>
            {loadingTensor === t.name && (
              <div style={{ fontSize: 10, color: 'var(--text-dim)' }}>
                Loading...
              </div>
            )}
            {heatmap && <HeatmapCanvas data={heatmap} />}
          </div>
        );
      })}
    </div>
  );
}
