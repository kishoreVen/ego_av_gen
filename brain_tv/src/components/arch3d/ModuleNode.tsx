import { useRef, useState, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';
import type { ArchModule, ArchHudSettings } from '../../types';
import { layerColor, type LayerCategory, type NodePosition, type NodeTensorInfo } from './layoutGraph';

interface ModuleNodeProps {
  module: ArchModule;
  fullName: string;
  category: LayerCategory;
  position: NodePosition;
  tensorInfo: NodeTensorInfo | null;
  hud: ArchHudSettings;
  selected: boolean;
  onSelect: (name: string) => void;
}

function formatParams(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function formatShape(shape: number[]): string {
  return shape.join('\u00d7');
}

export function ModuleNode({
  module,
  fullName,
  category,
  position,
  tensorInfo,
  hud,
  selected,
  onSelect,
}: ModuleNodeProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  const color = layerColor(category);

  useFrame(() => {
    if (!meshRef.current) return;
    const mat = meshRef.current.material as THREE.MeshStandardMaterial;
    const targetIntensity = hovered || selected ? 0.5 : 0.05;
    mat.emissiveIntensity += (targetIntensity - mat.emissiveIntensity) * 0.12;
  });

  const geometry = useMemo(() => {
    const w = position.width;
    const h = position.height;
    const d = position.depth;

    switch (category) {
      case 'activation':
        return <octahedronGeometry args={[w * 0.5, 0]} />;
      case 'attention':
        return <sphereGeometry args={[w * 0.5, 16, 16]} />;
      case 'norm':
        return <cylinderGeometry args={[w * 0.5, w * 0.5, h * 0.35, 20]} />;
      case 'embedding':
        return <cylinderGeometry args={[w * 0.4, w * 0.4, h, 12]} />;
      case 'downsample':
      case 'pool':
        return <cylinderGeometry args={[w * 0.2, w * 0.5, h, 6]} />;
      case 'upsample':
        return <cylinderGeometry args={[w * 0.5, w * 0.2, h, 6]} />;
      case 'linear':
        return <boxGeometry args={[w * 2, h * 0.4, d * 0.5]} />;
      default:
        return <boxGeometry args={[w, h, d]} />;
    }
  }, [category, position.width, position.height, position.depth]);

  // Build label lines based on HUD settings
  const topLabel = useMemo(() => {
    const parts: string[] = [];
    if (hud.showName) parts.push(module.name || module.type);
    if (hud.showType && module.name) parts.push(module.type);
    if (hud.showParams && module.params > 0) parts.push(formatParams(module.params));
    return parts.join(' \u00b7 ');
  }, [hud.showName, hud.showType, hud.showParams, module]);

  const inputLabel = useMemo(() => {
    if (!hud.showInputShape || !tensorInfo?.inputShape) return null;
    return `\u2192 ${formatShape(tensorInfo.inputShape)}`;
  }, [hud.showInputShape, tensorInfo]);

  const outputLabel = useMemo(() => {
    if (!hud.showOutputShape || !tensorInfo?.outputShape) return null;
    return `${formatShape(tensorInfo.outputShape)} \u2192`;
  }, [hud.showOutputShape, tensorInfo]);

  return (
    <group position={[position.x, position.y, position.z]}>
      <mesh
        ref={meshRef}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
          document.body.style.cursor = 'pointer';
        }}
        onPointerOut={() => {
          setHovered(false);
          document.body.style.cursor = 'auto';
        }}
        onClick={(e) => {
          e.stopPropagation();
          onSelect(fullName);
        }}
      >
        {geometry}
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.05}
          transparent
          opacity={category === 'dropout' ? 0.3 : 0.88}
          wireframe={category === 'dropout'}
          roughness={0.35}
          metalness={0.3}
        />
      </mesh>

      {selected && (
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <ringGeometry args={[position.width * 0.8, position.width * 0.9, 32]} />
          <meshBasicMaterial color="#58a6ff" transparent opacity={0.5} side={THREE.DoubleSide} />
        </mesh>
      )}

      {/* Name / type / params label — topmost */}
      {topLabel && (
        <Text
          position={[0, position.height * 0.5 + (inputLabel ? 0.7 : 0.3), 0]}
          fontSize={0.25}
          color="#e6edf3"
          anchorX="center"
          anchorY="bottom"
          outlineWidth={0.025}
          outlineColor="#0d1117"
          maxWidth={5}
        >
          {topLabel}
        </Text>
      )}

      {/* Input shape — between name and node */}
      {inputLabel && (
        <Text
          position={[0, position.height * 0.5 + 0.3, 0]}
          fontSize={0.17}
          color="#3fb950"
          anchorX="center"
          anchorY="bottom"
          outlineWidth={0.02}
          outlineColor="#0d1117"
          maxWidth={5}
        >
          {inputLabel}
        </Text>
      )}

      {/* Output shape — below the node */}
      {outputLabel && (
        <Text
          position={[0, -position.height * 0.5 - 0.2, 0]}
          fontSize={0.17}
          color="#58a6ff"
          anchorX="center"
          anchorY="top"
          outlineWidth={0.02}
          outlineColor="#0d1117"
          maxWidth={5}
        >
          {outputLabel}
        </Text>
      )}
    </group>
  );
}
