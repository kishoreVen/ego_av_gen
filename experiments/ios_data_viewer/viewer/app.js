import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const container = document.getElementById("canvas-container");
const sessionSelect = document.getElementById("session-select");
const pointCountEl = document.getElementById("point-count");
const poseCountEl = document.getElementById("pose-count");
const modeEl = document.getElementById("mode");
const showTrajectoryEl = document.getElementById("show-trajectory");
const showFrustumsEl = document.getElementById("show-frustums");
const pointSizeEl = document.getElementById("point-size");
const videoEl = document.getElementById("video-player");
const depthImageEl = document.getElementById("depth-image");
const depthStatusEl = document.getElementById("depth-status");
const topbarStatsEl = document.getElementById("topbar-stats");

const TOPBAR_HEIGHT = 48;
function viewportHeight() {
  return window.innerHeight - TOPBAR_HEIGHT;
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0b0d);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / viewportHeight(), 0.01, 1000);
camera.position.set(1, 1, 1);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, viewportHeight());
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AxesHelper(0.3));
scene.add(new THREE.GridHelper(4, 40, 0x333333, 0x1c1c1c));

let pointCloud = null;
let trajectoryLine = null;
let frustumGroup = null;
let depthEntries = []; // [{timestamp, image}], sorted by timestamp, for inspector sync

function clearSceneObjects() {
  for (const obj of [pointCloud, trajectoryLine, frustumGroup]) {
    if (obj) scene.remove(obj);
  }
  pointCloud = trajectoryLine = frustumGroup = null;
}

function buildPointCloud(positions, colors) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3, true));
  const material = new THREE.PointsMaterial({
    size: parseFloat(pointSizeEl.value),
    vertexColors: true,
  });
  return new THREE.Points(geometry, material);
}

function buildTrajectoryLine(trajectory) {
  const points = trajectory.map((p) => new THREE.Vector3(p.position[0], p.position[1], p.position[2]));
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({ color: 0xff5050 });
  return new THREE.Line(geometry, material);
}

function poseMatrix(pose) {
  const r = pose.rotation;
  const p = pose.position;
  const m = new THREE.Matrix4();
  // r is row-major 3x3; this mirrors ARFrame.camera.transform (camera-to-world).
  m.set(
    r[0], r[1], r[2], p[0],
    r[3], r[4], r[5], p[1],
    r[6], r[7], r[8], p[2],
    0, 0, 0, 1
  );
  return m;
}

function buildFrustums(trajectory, stride = 8, size = 0.05) {
  const group = new THREE.Group();
  const halfW = size * 0.6;
  const halfH = size * 0.45;
  const depth = size;
  // ARKit camera-local space: +Z toward viewer, so "forward" is -Z.
  const localApex = new THREE.Vector3(0, 0, 0);
  const localCorners = [
    new THREE.Vector3(-halfW, halfH, -depth),
    new THREE.Vector3(halfW, halfH, -depth),
    new THREE.Vector3(halfW, -halfH, -depth),
    new THREE.Vector3(-halfW, -halfH, -depth),
  ];

  const material = new THREE.LineBasicMaterial({ color: 0x8ecbff });

  for (let i = 0; i < trajectory.length; i += stride) {
    const m = poseMatrix(trajectory[i]);
    const apex = localApex.clone().applyMatrix4(m);
    const corners = localCorners.map((c) => c.clone().applyMatrix4(m));

    const verts = [];
    for (const c of corners) {
      verts.push(apex.x, apex.y, apex.z, c.x, c.y, c.z);
    }
    for (let j = 0; j < 4; j++) {
      const a = corners[j];
      const b = corners[(j + 1) % 4];
      verts.push(a.x, a.y, a.z, b.x, b.y, b.z);
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
    group.add(new THREE.LineSegments(geometry, material));
  }
  return group;
}

function fitCameraToPoints(positions) {
  if (positions.length === 0) return;
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  for (let i = 0; i < positions.length; i += 3) {
    v.set(positions[i], positions[i + 1], positions[i + 2]);
    box.expandByPoint(v);
  }
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length() || 1;
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(size, size * 0.6, size));
  camera.near = size / 1000;
  camera.far = size * 50;
  camera.updateProjectionMatrix();
  controls.update();
}

async function loadSession(name) {
  clearSceneObjects();

  const [meta, trajectory, positionsBuf, colorsBuf] = await Promise.all([
    fetch(`data/${name}/meta.json`).then((r) => r.json()),
    fetch(`data/${name}/trajectory.json`).then((r) => r.json()),
    fetch(`data/${name}/positions.f32`).then((r) => r.arrayBuffer()),
    fetch(`data/${name}/colors.u8`).then((r) => r.arrayBuffer()),
  ]);

  const positions = new Float32Array(positionsBuf);
  const colors = new Uint8Array(colorsBuf);

  pointCountEl.textContent = meta.point_count.toLocaleString();
  poseCountEl.textContent = meta.pose_count.toLocaleString();
  modeEl.textContent = meta.mode;
  topbarStatsEl.textContent = `${meta.pose_count} poses · ${meta.depth_frame_count ?? 0} depth frames`;

  pointCloud = buildPointCloud(positions, colors);
  scene.add(pointCloud);

  trajectoryLine = buildTrajectoryLine(trajectory);
  trajectoryLine.visible = showTrajectoryEl.checked;
  scene.add(trajectoryLine);

  frustumGroup = buildFrustums(trajectory);
  frustumGroup.visible = showFrustumsEl.checked;
  scene.add(frustumGroup);

  fitCameraToPoints(positions);

  // --- video + depth inspector ---
  depthEntries = trajectory
    .filter((p) => p.depth_image)
    .map((p) => ({ timestamp: p.timestamp, image: p.depth_image, frame_index: p.frame_index }))
    .sort((a, b) => a.timestamp - b.timestamp);

  if (meta.has_video) {
    videoEl.src = `data/${name}/video.mp4`;
    videoEl.load();
  } else {
    videoEl.removeAttribute("src");
  }
  updateDepthForTime(0);
}

function nearestDepthEntry(t) {
  if (depthEntries.length === 0) return null;
  let lo = 0;
  let hi = depthEntries.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (depthEntries[mid].timestamp < t) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(depthEntries[lo - 1].timestamp - t) < Math.abs(depthEntries[lo].timestamp - t)) {
    lo -= 1;
  }
  return depthEntries[lo];
}

function updateDepthForTime(t) {
  const entry = nearestDepthEntry(t);
  const currentName = sessionSelect.value;
  if (!entry) {
    depthStatusEl.textContent = "no depth data for this session";
    depthImageEl.removeAttribute("src");
    return;
  }
  depthImageEl.src = `data/${currentName}/${entry.image}`;
  depthStatusEl.textContent = `frame ${entry.frame_index} · t = ${entry.timestamp.toFixed(2)}s (video t = ${t.toFixed(2)}s)`;
}

videoEl.addEventListener("timeupdate", () => updateDepthForTime(videoEl.currentTime));
videoEl.addEventListener("seeking", () => updateDepthForTime(videoEl.currentTime));

async function init() {
  const index = await fetch("data/index.json").then((r) => (r.ok ? r.json() : []));
  if (index.length === 0) {
    pointCountEl.textContent = "no sessions exported";
    return;
  }
  sessionSelect.innerHTML = index
    .map((entry) => `<option value="${entry.name}">${entry.name} (${entry.point_count.toLocaleString()} pts)</option>`)
    .join("");
  sessionSelect.addEventListener("change", () => loadSession(sessionSelect.value));
  await loadSession(index[index.length - 1].name);
}

showTrajectoryEl.addEventListener("change", () => {
  if (trajectoryLine) trajectoryLine.visible = showTrajectoryEl.checked;
});
showFrustumsEl.addEventListener("change", () => {
  if (frustumGroup) frustumGroup.visible = showFrustumsEl.checked;
});
pointSizeEl.addEventListener("input", () => {
  if (pointCloud) pointCloud.material.size = parseFloat(pointSizeEl.value);
});

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / viewportHeight();
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, viewportHeight());
});

for (const btn of document.querySelectorAll(".tab-btn")) {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const target = btn.dataset.view;
    document.getElementById("view-3d").classList.toggle("active", target === "3d");
    document.getElementById("view-inspector").classList.toggle("active", target === "inspector");
  });
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

init();
animate();
