import type { ArchModule, ArchFlow } from '../../types';

export interface NodePosition {
  x: number;
  y: number;
  z: number;
  width: number;
  height: number;
  depth: number;
}

export type NodePositionMap = Map<string, NodePosition>;

// ── Layer type classification ──────────────────────────────────────────

export type LayerCategory =
  | 'linear'
  | 'conv'
  | 'attention'
  | 'norm'
  | 'activation'
  | 'dropout'
  | 'embedding'
  | 'downsample'
  | 'upsample'
  | 'pool'
  | 'container'
  | 'resblock'
  | 'identity'
  | 'unknown';

const ACTIVATION_TYPES = new Set([
  'ReLU', 'SiLU', 'GELU', 'Sigmoid', 'Tanh', 'LeakyReLU',
  'PReLU', 'ELU', 'Softmax', 'LogSoftmax', 'Mish', 'Swish',
  'SinusoidalTimeEmbedding',
]);

const NORM_TYPES = new Set([
  'LayerNorm', 'GroupNorm', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d',
  'InstanceNorm1d', 'InstanceNorm2d', 'RMSNorm',
]);

export function classifyModule(mod: ArchModule): LayerCategory {
  const t = mod.type;
  const n = mod.name.toLowerCase();

  // Compound blocks
  if (t === 'ResBlock' || n.includes('resblock') || n.includes('res_block')) return 'resblock';
  if (t === 'Downsample' || n.includes('downsample')) return 'downsample';
  if (t === 'Upsample' || n.includes('upsample')) return 'upsample';

  // Leaf types
  if (t === 'Linear') return 'linear';
  if (t.startsWith('Conv') && !t.includes('Transpose')) return 'conv';
  if (t.includes('ConvTranspose')) return 'upsample';
  if (t.includes('Attention') || t.includes('MultiheadAttention')) return 'attention';
  if (NORM_TYPES.has(t)) return 'norm';
  if (ACTIVATION_TYPES.has(t)) return 'activation';
  if (t === 'Dropout' || t.startsWith('Dropout')) return 'dropout';
  if (t === 'Embedding') return 'embedding';
  if (t.includes('Pool')) return 'pool';
  if (t === 'Identity') return 'identity';
  if (t === 'Sequential' || t === 'ModuleList' || t === 'ModuleDict') return 'container';

  return 'unknown';
}

// ── Architecture pattern detection ─────────────────────────────────────

type ArchPattern = 'unet' | 'generic';

function detectPattern(root: ArchModule): ArchPattern {
  const childNames = root.children.map((c) => c.name.toLowerCase());
  const hasEncoder = childNames.some((n) => n.includes('encoder') || n.includes('down'));
  const hasDecoder = childNames.some((n) => n.includes('decoder') || n.includes('up'));
  if (hasEncoder && hasDecoder) return 'unet';
  return 'generic';
}

// ── Collect renderable nodes ───────────────────────────────────────────
// Instead of showing every leaf, we show "meaningful" modules:
// - Top-level children of the root model are always shown
// - For ModuleLists (encoder_blocks, decoder_blocks), show each item in the list
// - ResBlocks, Downsamples, Upsamples are shown as single nodes (not expanded)
// - Leaf modules at top level are shown directly

export interface RenderNode {
  fullName: string;
  module: ArchModule;
  category: LayerCategory;
  /** Which top-level group this belongs to */
  group: string;
}

export function collectRenderNodes(root: ArchModule): RenderNode[] {
  const nodes: RenderNode[] = [];

  for (const child of root.children) {
    const category = classifyModule(child);

    if (
      child.type === 'ModuleList' &&
      child.children.length > 0
    ) {
      // For ModuleLists, render each item as a single node
      for (const item of child.children) {
        nodes.push({
          fullName: `${child.name}.${item.name}`,
          module: item,
          category: classifyModule(item),
          group: child.name,
        });
      }
    } else if (child.type === 'Sequential' && child.children.length > 0) {
      // For Sequential, show each element
      for (const item of child.children) {
        nodes.push({
          fullName: `${child.name}.${item.name}`,
          module: item,
          category: classifyModule(item),
          group: child.name,
        });
      }
    } else {
      // Leaf or compound block — show as single node
      nodes.push({
        fullName: child.name,
        module: child,
        category,
        group: child.name,
      });
    }
  }

  return nodes;
}

// ── Node sizing ────────────────────────────────────────────────────────

function nodeScale(params: number): number {
  if (params <= 0) return 0.4;
  return 0.4 + Math.log10(params + 1) * 0.12;
}

// ── Layout: UNet ───────────────────────────────────────────────────────

const SPACING = 5.5;

function layoutUNet(root: ArchModule): { positions: NodePositionMap; nodes: RenderNode[] } {
  const positions: NodePositionMap = new Map();
  const allNodes = collectRenderNodes(root);

  // Categorize groups
  const groups: Record<string, RenderNode[]> = {};
  for (const node of allNodes) {
    if (!groups[node.group]) groups[node.group] = [];
    groups[node.group].push(node);
  }

  const childNames = root.children.map((c) => c.name);

  // Classify each top-level child
  const conditioning: string[] = []; // time_embed, info_proj
  const encoderInput: string[] = [];  // conv_in
  const encoderBlocks: string[] = []; // encoder_blocks
  const downsamples: string[] = [];   // downsamples
  const midBlocks: string[] = [];     // mid_block1, mid_block2
  const decoderBlocks: string[] = []; // decoder_blocks
  const upsamples: string[] = [];     // upsamples
  const skipConvs: string[] = [];     // skip_convs
  const outputBlocks: string[] = [];  // conv_out

  for (const name of childNames) {
    const lower = name.toLowerCase();
    if (lower.includes('time') || lower.includes('info') || lower.includes('embed'))
      conditioning.push(name);
    else if (lower === 'conv_in')
      encoderInput.push(name);
    else if (lower.includes('encoder') || lower === 'encoder_blocks')
      encoderBlocks.push(name);
    else if (lower.includes('downsample'))
      downsamples.push(name);
    else if (lower.includes('mid') || lower.includes('bottleneck'))
      midBlocks.push(name);
    else if (lower.includes('decoder') || lower === 'decoder_blocks')
      decoderBlocks.push(name);
    else if (lower.includes('upsample'))
      upsamples.push(name);
    else if (lower.includes('skip'))
      skipConvs.push(name);
    else if (lower === 'conv_out')
      outputBlocks.push(name);
    else
      conditioning.push(name);
  }

  // Get encoder nodes and determine how many levels there are
  const encNodes = encoderBlocks.flatMap((g) => groups[g] || []);
  const downNodes = downsamples.flatMap((g) => groups[g] || []);
  const midNodes = midBlocks.flatMap((g) => groups[g] || []);
  const decNodes = decoderBlocks.flatMap((g) => groups[g] || []);
  const upNodes = upsamples.flatMap((g) => groups[g] || []);
  const skipNodes = skipConvs.flatMap((g) => groups[g] || []);
  const condNodes = conditioning.flatMap((g) => groups[g] || []);
  const inNodes = encoderInput.flatMap((g) => groups[g] || []);
  const outNodes = outputBlocks.flatMap((g) => groups[g] || []);

  // Figure out how many "levels" the encoder has (each level = N res blocks + 1 downsample)
  const numDown = downNodes.length;
  const numResPerLevel = numDown > 0 ? Math.round(encNodes.length / (numDown + 1)) : encNodes.length;

  // Layout the U-shape
  // Encoder goes down on the LEFT side (negative X), each level drops in Y
  // Decoder goes back up on the RIGHT side (positive X)
  // Mid blocks at the BOTTOM center

  // Conditioning (time_embed items, info_proj) — floating above, far left
  for (let i = 0; i < condNodes.length; i++) {
    const n = condNodes[i];
    const s = nodeScale(n.module.params);
    positions.set(n.fullName, {
      x: -8 - i * SPACING * 0.8,
      y: 3,
      z: 0,
      width: s, height: s * 0.6, depth: s * 0.6,
    });
  }

  // Input conv — top left
  for (const n of inNodes) {
    const s = nodeScale(n.module.params);
    positions.set(n.fullName, {
      x: -SPACING * 1.5,
      y: 0,
      z: 0,
      width: s, height: s, depth: s,
    });
  }

  // Encoder: walk through blocks, grouping by level
  let encIdx = 0;
  for (let level = 0; level <= numDown; level++) {
    const numBlocks = (level === numDown) ? encNodes.length - encIdx : numResPerLevel;
    for (let b = 0; b < numBlocks && encIdx < encNodes.length; b++) {
      const n = encNodes[encIdx];
      const s = nodeScale(n.module.params);
      positions.set(n.fullName, {
        x: -SPACING * 1.5 + b * SPACING * 0.6,
        y: -(level + 1) * SPACING,
        z: 0,
        width: s, height: s, depth: s,
      });
      encIdx++;
    }

    // Place downsample after this level's blocks
    if (level < numDown && level < downNodes.length) {
      const dn = downNodes[level];
      const s = nodeScale(dn.module.params);
      positions.set(dn.fullName, {
        x: -SPACING * 0.5,
        y: -(level + 1) * SPACING - SPACING * 0.5,
        z: 0,
        width: s * 0.8, height: s * 1.2, depth: s * 0.8,
      });
    }
  }

  const bottomY = -(numDown + 1) * SPACING - SPACING;

  // Mid blocks — at the bottom center
  for (let i = 0; i < midNodes.length; i++) {
    const n = midNodes[i];
    const s = nodeScale(n.module.params);
    positions.set(n.fullName, {
      x: (i - (midNodes.length - 1) / 2) * SPACING,
      y: bottomY,
      z: 0,
      width: s * 1.2, height: s * 1.2, depth: s * 1.2,
    });
  }

  // Decoder: interleave skip_convs and decoder_blocks, rising back up
  // skip_convs come first, then the decoder block, then upsample
  const numUp = upNodes.length;
  const numDecPerLevel = numUp > 0 ? Math.round(decNodes.length / (numUp + 1)) : decNodes.length;

  let decIdx = 0;
  let skipIdx = 0;
  for (let level = 0; level <= numUp; level++) {
    const levelY = bottomY + (level + 1) * SPACING;
    const numBlocks = (level === numUp) ? decNodes.length - decIdx : numDecPerLevel;

    // Skip convs for this level
    const skipsThisLevel = numBlocks; // typically same number
    for (let b = 0; b < skipsThisLevel && skipIdx < skipNodes.length; b++) {
      const sn = skipNodes[skipIdx];
      const s = nodeScale(sn.module.params);
      positions.set(sn.fullName, {
        x: SPACING * 0.4 + b * SPACING * 0.5,
        y: levelY + SPACING * 0.3,
        z: SPACING * 0.5,
        width: s * 0.7, height: s * 0.5, depth: s * 0.5,
      });
      skipIdx++;
    }

    // Decoder blocks
    for (let b = 0; b < numBlocks && decIdx < decNodes.length; b++) {
      const n = decNodes[decIdx];
      const s = nodeScale(n.module.params);
      positions.set(n.fullName, {
        x: SPACING * 1.5 + b * SPACING * 0.6,
        y: levelY,
        z: 0,
        width: s, height: s, depth: s,
      });
      decIdx++;
    }

    // Upsample after decoder blocks for this level
    if (level < numUp && level < upNodes.length) {
      const un = upNodes[level];
      const s = nodeScale(un.module.params);
      positions.set(un.fullName, {
        x: SPACING * 0.5,
        y: levelY + SPACING * 0.5,
        z: 0,
        width: s * 0.8, height: s * 1.2, depth: s * 0.8,
      });
    }
  }

  // Output conv — top right
  for (let i = 0; i < outNodes.length; i++) {
    const n = outNodes[i];
    const s = nodeScale(n.module.params);
    positions.set(n.fullName, {
      x: SPACING * 1.5 + i * SPACING * 0.6,
      y: 0,
      z: 0,
      width: s, height: s, depth: s,
    });
  }

  return { positions, nodes: allNodes };
}

// ── Layout: Generic / fallback ─────────────────────────────────────────

function layoutGeneric(root: ArchModule): { positions: NodePositionMap; nodes: RenderNode[] } {
  const positions: NodePositionMap = new Map();
  const allNodes = collectRenderNodes(root);

  // Simple flow layout along Z axis
  allNodes.forEach((node, i) => {
    const s = nodeScale(node.module.params);
    positions.set(node.fullName, {
      x: 0,
      y: 0,
      z: i * SPACING,
      width: s, height: s, depth: s,
    });
  });

  return { positions, nodes: allNodes };
}

// ── Build flow edges that match renderable nodes ───────────────────────
// The raw flow connects leaf modules. We need to map those to the
// collapsed renderable nodes.

export function buildRenderableFlow(
  rawFlow: ArchFlow[],
  renderNodes: RenderNode[]
): { from: string; to: string; shape: number[] | null }[] {
  // Build a map: leaf full name → renderable node name
  const leafToRender = new Map<string, string>();
  for (const rn of renderNodes) {
    // This render node represents module at rn.fullName and all its descendants
    leafToRender.set(rn.fullName, rn.fullName);
  }

  // For a given leaf name, find which render node "owns" it
  function findOwner(leafName: string): string | null {
    // Check if this leaf name starts with any render node's fullName
    // Sort render nodes by name length descending so we match the most specific one
    for (const rn of renderNodes) {
      if (leafName === rn.fullName || leafName.startsWith(rn.fullName + '.')) {
        return rn.fullName;
      }
    }
    return null;
  }

  const seen = new Set<string>();
  const edges: { from: string; to: string; shape: number[] | null }[] = [];

  for (const raw of rawFlow) {
    const fromOwner = findOwner(raw.from);
    const toOwner = findOwner(raw.to);
    if (!fromOwner || !toOwner || fromOwner === toOwner) continue;

    const key = `${fromOwner}->${toOwner}`;
    if (seen.has(key)) continue;
    seen.add(key);

    edges.push({ from: fromOwner, to: toOwner, shape: raw.shape });
  }

  return edges;
}

// ── Public API ──────────────────────────────────────────────────────────

export function computeLayout(
  root: ArchModule,
  flow: ArchFlow[]
): {
  positions: NodePositionMap;
  nodes: RenderNode[];
  edges: { from: string; to: string; shape: number[] | null }[];
  pattern: ArchPattern;
} {
  const pattern = detectPattern(root);

  let result: { positions: NodePositionMap; nodes: RenderNode[] };
  switch (pattern) {
    case 'unet':
      result = layoutUNet(root);
      break;
    default:
      result = layoutGeneric(root);
  }

  const edges = buildRenderableFlow(flow, result.nodes);

  return {
    positions: result.positions,
    nodes: result.nodes,
    edges,
    pattern,
  };
}

// ── Per-node tensor shapes ──────────────────────────────────────────────

export interface NodeTensorInfo {
  inputShape: number[] | null;
  outputShape: number[] | null;
}

export function computeNodeShapes(
  rawFlow: ArchFlow[],
  renderNodes: RenderNode[]
): Map<string, NodeTensorInfo> {
  const info = new Map<string, NodeTensorInfo>();

  // Initialize all nodes
  for (const rn of renderNodes) {
    info.set(rn.fullName, { inputShape: null, outputShape: null });
  }

  // For a given leaf name, find the render node that owns it
  function findOwner(leafName: string): string | null {
    for (const rn of renderNodes) {
      if (leafName === rn.fullName || leafName.startsWith(rn.fullName + '.')) {
        return rn.fullName;
      }
    }
    return null;
  }

  // Walk raw flow: the first edge entering a render node gives its input shape,
  // the last edge leaving a render node gives its output shape
  for (const raw of rawFlow) {
    const fromOwner = findOwner(raw.from);
    const toOwner = findOwner(raw.to);

    // Edge leaving fromOwner → shape is the output of fromOwner
    if (fromOwner && raw.shape) {
      const entry = info.get(fromOwner)!;
      // Keep updating — last one wins (output of the whole block)
      entry.outputShape = raw.shape;
    }

    // Edge entering toOwner from a different block → shape is input to toOwner
    if (toOwner && fromOwner !== toOwner && raw.shape) {
      const entry = info.get(toOwner)!;
      if (!entry.inputShape) {
        entry.inputShape = raw.shape;
      }
    }
  }

  return info;
}

// ── Color mapping ──────────────────────────────────────────────────────

export function layerColor(category: LayerCategory): string {
  switch (category) {
    case 'linear':      return '#3fb950';
    case 'conv':        return '#58a6ff';
    case 'attention':   return '#d946ef';
    case 'norm':        return '#d29922';
    case 'activation':  return '#f97316';
    case 'dropout':     return '#8b949e';
    case 'embedding':   return '#22d3ee';
    case 'downsample':  return '#f85149';
    case 'pool':        return '#f85149';
    case 'upsample':    return '#a5d6ff';
    case 'resblock':    return '#7c5cbf';
    case 'container':   return '#30363d';
    case 'identity':    return '#484f58';
    default:            return '#8b949e';
  }
}
