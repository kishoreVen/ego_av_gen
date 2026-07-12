export interface RunInfo {
  experiment: string;
  timestamp: string;
  hasMetrics: boolean;
  hasCheckpoints: boolean;
  hasVisualizations: boolean;
  hasConfig: boolean;
  lastStep: number | null;
  lastLoss: number | null;
  lastCheckpoint: string | null;
}

export interface MetricEntry {
  step: number;
  [key: string]: number;
}

export interface CheckpointInfo {
  name: string;
  step: number | null;
  path: string;
  files: { name: string; size: string }[];
}

export interface FileEntry {
  path: string;
  size: number;
  isDir: boolean;
}

export interface ProcessInfo {
  id: string;
  type: 'train' | 'inference';
  recipe: string;
  startedAt: string;
  status: 'running' | 'exited';
  exitCode: number | null;
}

export interface ProcessOutput {
  id: string;
  status: string;
  exitCode: number | null;
  output: string[];
  totalLines: number;
}

// Architecture graph types
export interface ArchModule {
  name: string;
  type: string;
  params: number;
  own_params: number;
  shape_desc: string | null;
  children: ArchModule[];
}

export interface ArchFlow {
  from: string;
  to: string;
  shape: number[] | null;
}

export interface ArchitectureExport {
  model_name: string;
  modules: ArchModule;
  flow: ArchFlow[];
}

// Weight visualization types
export interface WeightTensorInfo {
  name: string;
  shape: number[];
  dtype: string;
  num_elements: number;
  min: number;
  max: number;
  mean: number;
  std: number;
}

export interface WeightHeatmapData {
  name: string;
  original_shape: number[];
  heatmap_shape: number[];
  min: number;
  max: number;
  mean: number;
  std: number;
  values: number[][];
}

export interface ArchHudSettings {
  showName: boolean;
  showType: boolean;
  showParams: boolean;
  showInputShape: boolean;
  showOutputShape: boolean;
  showEdgeShapes: boolean;
  showParticles: boolean;
}
