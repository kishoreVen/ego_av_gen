import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Line, Text } from '@react-three/drei';
import * as THREE from 'three';
import type { ArchHudSettings } from '../../types';
import type { NodePosition } from './layoutGraph';

interface FlowEdgeProps {
  from: NodePosition;
  to: NodePosition;
  color: string;
  shape: number[] | null;
  hud: ArchHudSettings;
  index: number;
}

const CURVE_POINTS = 50;
const PARTICLE_COUNT = 2;

function formatShape(s: number[]): string {
  return s.join('\u00d7');
}

export function FlowEdge({ from, to, color, shape, hud, index }: FlowEdgeProps) {
  const particleRefs = useRef<(THREE.Mesh | null)[]>([]);

  const { curve, linePoints, midPoint } = useMemo(() => {
    const start = new THREE.Vector3(from.x, from.y, from.z);
    const end = new THREE.Vector3(to.x, to.y, to.z);

    const mid = new THREE.Vector3(
      (from.x + to.x) / 2,
      Math.max(from.y, to.y) + 0.6,
      (from.z + to.z) / 2
    );

    const c = new THREE.QuadraticBezierCurve3(start, mid, end);
    const pts = c.getPoints(CURVE_POINTS);
    const lp = pts.map((p): [number, number, number] => [p.x, p.y, p.z]);
    const mp = c.getPoint(0.5);
    return { curve: c, linePoints: lp, midPoint: mp };
  }, [from.x, from.y, from.z, to.x, to.y, to.z]);

  useFrame(({ clock }) => {
    if (!hud.showParticles) return;
    const t = clock.getElapsedTime();
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const mesh = particleRefs.current[i];
      if (!mesh) continue;
      const phase = (i / PARTICLE_COUNT + index * 0.37) % 1;
      const progress = (t * 0.3 + phase) % 1;
      const pos = curve.getPoint(progress);
      mesh.position.copy(pos);
    }
  });

  return (
    <group>
      <Line
        points={linePoints}
        color={color}
        lineWidth={1}
        transparent
        opacity={0.25}
      />

      {/* Shape label at midpoint of edge */}
      {hud.showEdgeShapes && shape && (
        <Text
          position={[midPoint.x, midPoint.y + 0.15, midPoint.z]}
          fontSize={0.14}
          color="#8b949e"
          anchorX="center"
          anchorY="bottom"
          outlineWidth={0.015}
          outlineColor="#0d1117"
        >
          {formatShape(shape)}
        </Text>
      )}

      {/* Animated particles */}
      {hud.showParticles &&
        Array.from({ length: PARTICLE_COUNT }).map((_, i) => (
          <mesh
            key={i}
            ref={(el) => {
              particleRefs.current[i] = el;
            }}
          >
            <sphereGeometry args={[0.06, 8, 8]} />
            <meshBasicMaterial color={color} transparent opacity={0.9} />
          </mesh>
        ))}
    </group>
  );
}
