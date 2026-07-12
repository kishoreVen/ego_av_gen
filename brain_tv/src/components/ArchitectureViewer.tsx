import { useState, useMemo, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useApi } from '../hooks/useApi';
import type { ArchitectureExport, ArchHudSettings, CheckpointInfo } from '../types';
import { computeLayout, computeNodeShapes, layerColor, type RenderNode } from './arch3d/layoutGraph';
import { ModuleNode } from './arch3d/ModuleNode';
import { FlowEdge } from './arch3d/FlowEdge';
import { WeightViewer } from './WeightViewer';

interface Props {
  basePath: string;
}

const DEFAULT_HUD: ArchHudSettings = {
  showName: true,
  showType: true,
  showParams: true,
  showInputShape: true,
  showOutputShape: true,
  showEdgeShapes: false,
  showParticles: true,
};

function formatParams(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function formatShape(shape: number[] | null): string {
  if (!shape) return '-';
  return shape.join('\u00d7');
}

function Scene({
  data,
  hud,
  selectedNode,
  onSelectNode,
}: {
  data: ArchitectureExport;
  hud: ArchHudSettings;
  selectedNode: string | null;
  onSelectNode: (name: string | null) => void;
}) {
  const { positions, nodes, edges } = useMemo(
    () => computeLayout(data.modules, data.flow),
    [data]
  );

  const nodeShapes = useMemo(
    () => computeNodeShapes(data.flow, nodes),
    [data.flow, nodes]
  );

  const handleSelect = useCallback(
    (name: string) => {
      onSelectNode(name === selectedNode ? null : name);
    },
    [selectedNode, onSelectNode]
  );

  return (
    <>
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 20, 10]} intensity={0.7} />
      <directionalLight position={[-8, 12, -8]} intensity={0.3} />
      <pointLight position={[0, -15, 0]} intensity={0.2} color="#58a6ff" />

      {nodes.map((rn) => {
        const pos = positions.get(rn.fullName);
        if (!pos) return null;
        return (
          <ModuleNode
            key={rn.fullName}
            module={rn.module}
            fullName={rn.fullName}
            category={rn.category}
            position={pos}
            tensorInfo={nodeShapes.get(rn.fullName) ?? null}
            hud={hud}
            selected={selectedNode === rn.fullName}
            onSelect={handleSelect}
          />
        );
      })}

      {edges.map((edge, i) => {
        const fromPos = positions.get(edge.from);
        const toPos = positions.get(edge.to);
        if (!fromPos || !toPos) return null;

        const srcNode = nodes.find((n) => n.fullName === edge.from);
        const color = srcNode ? layerColor(srcNode.category) : '#8b949e';

        return (
          <FlowEdge
            key={`${edge.from}->${edge.to}`}
            from={fromPos}
            to={toPos}
            color={color}
            shape={edge.shape}
            hud={hud}
            index={i}
          />
        );
      })}

      <gridHelper args={[60, 60, '#1c2128', '#1c2128']} position={[0, -20, 0]} />

      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.1}
        minDistance={3}
        maxDistance={80}
      />
    </>
  );
}

export function ArchitectureViewer({ basePath }: Props) {
  const { data, loading, error } = useApi<ArchitectureExport>(
    `${basePath}/architecture`
  );
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [hud, setHud] = useState<ArchHudSettings>(DEFAULT_HUD);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string | null>(null);

  const { data: checkpoints } = useApi<CheckpointInfo[]>(
    `${basePath}/checkpoints`
  );

  const layoutResult = useMemo(() => {
    if (!data) return null;
    return computeLayout(data.modules, data.flow);
  }, [data]);

  const nodeShapes = useMemo(() => {
    if (!data || !layoutResult) return null;
    return computeNodeShapes(data.flow, layoutResult.nodes);
  }, [data, layoutResult]);

  if (loading) return <div className="loading">Loading architecture...</div>;
  if (error)
    return (
      <div className="empty-state">
        <h3>No architecture graph</h3>
        <p>
          Run a training step to generate <code>architecture.json</code>, or the
          model has not been exported yet.
        </p>
      </div>
    );
  if (!data || !layoutResult) return null;

  const selectedRenderNode = selectedNode
    ? layoutResult.nodes.find((n) => n.fullName === selectedNode)
    : null;

  const selectedTensor = selectedNode && nodeShapes
    ? nodeShapes.get(selectedNode) ?? null
    : null;

  const toggleHud = (key: keyof ArchHudSettings) => {
    setHud((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <div className="arch-container">
      <Canvas
        camera={{ position: [0, 5, 40], fov: 50 }}
        style={{ background: '#0d1117' }}
        onPointerMissed={() => setSelectedNode(null)}
      >
        <Scene
          data={data}
          hud={hud}
          selectedNode={selectedNode}
          onSelectNode={setSelectedNode}
        />
      </Canvas>

      {/* HUD Settings */}
      <div className="arch-hud">
        <div className="arch-hud-title">Display</div>
        {(
          [
            ['showName', 'Name'],
            ['showType', 'Type'],
            ['showParams', 'Params'],
            ['showInputShape', 'Input shape'],
            ['showOutputShape', 'Output shape'],
            ['showEdgeShapes', 'Edge shapes'],
            ['showParticles', 'Flow particles'],
          ] as const
        ).map(([key, label]) => (
          <label key={key} className="arch-hud-toggle">
            <input
              type="checkbox"
              checked={hud[key]}
              onChange={() => toggleHud(key)}
            />
            <span>{label}</span>
          </label>
        ))}
      </div>

      {/* Info sidebar */}
      <div className="arch-sidebar">
        <div className="arch-sidebar-header">
          <span className="arch-model-name">{data.model_name}</span>
          <span className="arch-model-params">
            {formatParams(data.modules.params)} params · {layoutResult.nodes.length} modules
          </span>
        </div>

        {checkpoints && checkpoints.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              Checkpoint
            </span>
            <select
              value={selectedCheckpoint ?? ''}
              onChange={(e) =>
                setSelectedCheckpoint(e.target.value || null)
              }
              style={{
                background: 'var(--bg)',
                color: 'var(--text)',
                border: '1px solid var(--border)',
                borderRadius: 4,
                padding: '4px 6px',
                fontSize: 12,
                fontFamily: 'var(--font-mono)',
              }}
            >
              <option value="">None</option>
              {checkpoints.map((cp) => (
                <option key={cp.name} value={cp.name}>
                  {cp.name}
                </option>
              ))}
            </select>
          </div>
        )}

        {selectedRenderNode ? (
          <SelectedDetail
            node={selectedRenderNode}
            tensorInfo={selectedTensor}
            basePath={basePath}
            checkpointName={selectedCheckpoint}
          />
        ) : (
          <div className="arch-hint">Click a node to inspect</div>
        )}

        {/* Legend */}
        <div className="arch-legend">
          {(
            [
              ['Linear', 'linear'],
              ['Conv', 'conv'],
              ['ResBlock', 'resblock'],
              ['Attention', 'attention'],
              ['Norm', 'norm'],
              ['Activation', 'activation'],
              ['Downsample', 'downsample'],
              ['Upsample', 'upsample'],
              ['Embedding', 'embedding'],
            ] as const
          ).map(([label, cat]) => (
            <div key={cat} className="arch-legend-item">
              <span
                className="arch-legend-dot"
                style={{ background: layerColor(cat) }}
              />
              <span>{label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function SelectedDetail({
  node,
  tensorInfo,
  basePath,
  checkpointName,
}: {
  node: RenderNode;
  tensorInfo: import('./arch3d/layoutGraph').NodeTensorInfo | null;
  basePath: string;
  checkpointName: string | null;
}) {
  const mod = node.module;

  return (
    <div className="arch-detail">
      <div className="arch-detail-row">
        <span className="arch-detail-label">Name</span>
        <span className="arch-detail-value">{node.fullName}</span>
      </div>
      <div className="arch-detail-row">
        <span className="arch-detail-label">Type</span>
        <span
          className="arch-detail-type"
          style={{ color: layerColor(node.category) }}
        >
          {mod.type}
        </span>
      </div>
      <div className="arch-detail-row">
        <span className="arch-detail-label">Parameters</span>
        <span className="arch-detail-value">{formatParams(mod.params)}</span>
      </div>
      {mod.shape_desc && (
        <div className="arch-detail-row">
          <span className="arch-detail-label">Shape</span>
          <span className="arch-detail-value arch-detail-mono">
            {mod.shape_desc}
          </span>
        </div>
      )}
      {tensorInfo && (
        <>
          <div className="arch-detail-row">
            <span className="arch-detail-label">Input</span>
            <span className="arch-detail-value arch-detail-mono" style={{ color: '#3fb950' }}>
              {formatShape(tensorInfo.inputShape)}
            </span>
          </div>
          <div className="arch-detail-row">
            <span className="arch-detail-label">Output</span>
            <span className="arch-detail-value arch-detail-mono" style={{ color: '#58a6ff' }}>
              {formatShape(tensorInfo.outputShape)}
            </span>
          </div>
        </>
      )}
      {mod.children.length > 0 && (
        <div className="arch-detail-children">
          <span className="arch-detail-label">Children ({mod.children.length})</span>
          <div className="arch-detail-child-list">
            {mod.children.map((c) => (
              <div key={c.name} className="arch-detail-child">
                <span style={{ color: '#e6edf3' }}>{c.name}</span>
                <span style={{ color: '#8b949e' }}>
                  {c.type} · {formatParams(c.params)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
      {checkpointName && mod.params > 0 && (
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: 8 }}>
          <span
            style={{
              fontSize: 11,
              color: 'var(--text-muted)',
              marginBottom: 6,
              display: 'block',
            }}
          >
            Weights ({checkpointName})
          </span>
          <WeightViewer
            basePath={basePath}
            checkpointName={checkpointName}
            moduleName={node.fullName}
          />
        </div>
      )}
    </div>
  );
}
