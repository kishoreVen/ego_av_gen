import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// ── State ──────────────────────────────────────────────────────────────────

let currentResult   = null;
let snapshotResult  = null;
let snapshotPatches = null;
let embeddingMode   = 'siglip';
let simMode         = 'spatial';
let renderImg       = null;   // Image of last render (for patch thumbnails)
let snapImg         = null;   // Image of snapshot render

// ── Camera constants (orthographic, top-down) ─────────────────────────────
// Frustum exactly matches table dimensions so objects near the edge are near
// the image edge — makes patch arithmetic trivial.
const FRUSTUM_W = 3.8;   // world units (table is 3.2, add small margin)
const FRUSTUM_H = 2.8;   // world units (table is 2.2, add small margin)
const CAM_HEIGHT = 10;

// ── Three.js Setup ────────────────────────────────────────────────────────

const canvas    = document.getElementById('three-canvas');
const threeWrap = document.getElementById('three-wrap');

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

// Orthographic camera — no perspective distortion, simple patch math
const camera = new THREE.OrthographicCamera(
  -FRUSTUM_W / 2, FRUSTUM_W / 2,
   FRUSTUM_H / 2, -FRUSTUM_H / 2,
  0.1, 50
);
camera.position.set(0, CAM_HEIGHT, 0);
camera.up.set(0, 0, -1);   // world -Z = screen up
camera.lookAt(0, 0, 0);

function onResize() {
  const w = threeWrap.clientWidth;
  const h = threeWrap.clientHeight;
  renderer.setSize(w, h, false);
  // keep frustum fixed; just update canvas aspect (letterbox/pillarbox effect)
}
new ResizeObserver(onResize).observe(threeWrap);
onResize();

// ── Camera slider controls ────────────────────────────────────────────────

const camXSlider = document.getElementById('cam-x');
const camZSlider = document.getElementById('cam-z');
const camXVal    = document.getElementById('cam-x-val');
const camZVal    = document.getElementById('cam-z-val');

function updateCameraFromSliders() {
  const dx = parseFloat(camXSlider.value);
  const dz = parseFloat(camZSlider.value);
  camXVal.textContent = dx.toFixed(2);
  camZVal.textContent = dz.toFixed(2);
  camera.position.set(dx, CAM_HEIGHT, dz);
  camera.lookAt(dx, 0, dz);
}
camXSlider.addEventListener('input', updateCameraFromSliders);
camZSlider.addEventListener('input', updateCameraFromSliders);

// ── Lighting ──────────────────────────────────────────────────────────────

scene.add(new THREE.AmbientLight(0xffffff, 0.5));

const sun = new THREE.DirectionalLight(0xfff4e0, 2.0);
sun.position.set(4, 8, 4);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.near = 0.5; sun.shadow.camera.far = 30;
sun.shadow.camera.left = -6; sun.shadow.camera.right = 6;
sun.shadow.camera.top  =  6; sun.shadow.camera.bottom = -6;
scene.add(sun);
const fill = new THREE.DirectionalLight(0xc0d8ff, 0.5);
fill.position.set(-3, 3, -3);
scene.add(fill);

// ── Table ─────────────────────────────────────────────────────────────────

const tableMat = new THREE.MeshStandardMaterial({ color: 0x6b3a1f, roughness: 0.8, metalness: 0.05 });
const tableTop = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.1, 2.2), tableMat);
tableTop.position.set(0, 0.5, 0);
tableTop.receiveShadow = true;
scene.add(tableTop);

const legMat  = new THREE.MeshStandardMaterial({ color: 0x4a2810, roughness: 0.9 });
const legGeom = new THREE.BoxGeometry(0.12, 0.5, 0.12);
[[-1.5, 0.25, -1.0],[1.5, 0.25, -1.0],[-1.5, 0.25, 1.0],[1.5, 0.25, 1.0]].forEach(([x,y,z]) => {
  const leg = new THREE.Mesh(legGeom, legMat);
  leg.position.set(x, y, z);
  leg.castShadow = true;
  scene.add(leg);
});

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(20, 20),
  new THREE.MeshStandardMaterial({ color: 0x12121a, roughness: 0.95 })
);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);

// ── Object registry ───────────────────────────────────────────────────────
// Each entry: { name, worldX, worldZ, color }
const objectRegistry = [];

// Palette for consistent per-object colouring across the UI
const PALETTE = [
  '#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6',
  '#1abc9c','#e67e22','#e91e63','#00bcd4','#cddc39',
  '#ff5722','#607d8b','#795548','#009688','#673ab7','#ffc107',
];

// ── Object Helpers ────────────────────────────────────────────────────────

function normaliseGLTF(gltf, targetSize = 0.25) {
  const group = gltf.scene;
  group.traverse(n => { if (n.isMesh) { n.castShadow = true; n.receiveShadow = true; } });
  const box = new THREE.Box3().setFromObject(group);
  const size = box.getSize(new THREE.Vector3());
  const scale = targetSize / Math.max(size.x, size.y, size.z);
  group.scale.setScalar(scale);
  const box2 = new THREE.Box3().setFromObject(group);
  const centre = box2.getCenter(new THREE.Vector3());
  group.position.sub(centre);
  group.position.y -= box2.min.y;
  return group;
}

function placeOnTable(obj, x, z) {
  obj.position.set(x, 0.55, z);
  scene.add(obj);
}

function makeFallback(shape, color, size = 0.22) {
  let geom;
  switch (shape) {
    case 'sphere':   geom = new THREE.SphereGeometry(size * 0.5, 32, 32); break;
    case 'cylinder': geom = new THREE.CylinderGeometry(size * 0.35, size * 0.35, size, 32); break;
    case 'cone':     geom = new THREE.ConeGeometry(size * 0.4, size, 32); break;
    default:         geom = new THREE.BoxGeometry(size, size, size);
  }
  const mesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: 0.1 }));
  mesh.castShadow = true;
  return mesh;
}

// ── Random non-overlapping table placement ────────────────────────────────

function randomTablePositions(n, minDist = 0.34) {
  const X = 1.1, Z = 0.75;
  const pts = [];
  for (let i = 0; i < n; i++) {
    let p, tries = 0;
    do {
      p = { x: (Math.random() * 2 - 1) * X, z: (Math.random() * 2 - 1) * Z };
      tries++;
    } while (tries < 600 && pts.some(q => Math.hypot(q.x - p.x, q.z - p.z) < minDist));
    pts.push(p);
  }
  return pts;
}

// ── Load assets ───────────────────────────────────────────────────────────

const loader = new GLTFLoader();

const ASSETS = [
  { file: 'Duck.glb',              shape: 'sphere',   color: 0xf5c518 },
  { file: 'Avocado.glb',           shape: 'sphere',   color: 0x4caf50 },
  { file: 'ToyCar.glb',            shape: 'cone',     color: 0xe74c3c },
  { file: 'Lantern.glb',           shape: 'cylinder', color: 0xe67e22 },
  { file: 'WaterBottle.glb',       shape: 'cylinder', color: 0x2980b9 },
  { file: 'DamagedHelmet.glb',     shape: 'box',      color: 0x7f8c8d },
  { file: 'BarramundiFish.glb',    shape: 'cone',     color: 0x16a085 },
  { file: 'BoomBox.glb',           shape: 'box',      color: 0x8e44ad },
  { file: 'AntiqueCamera.glb',     shape: 'box',      color: 0xc0392b },
  { file: 'Suzanne.glb',           shape: 'sphere',   color: 0xf39c12 },
  { file: 'FlightHelmet.glb',      shape: 'sphere',   color: 0x1abc9c },
  { file: 'Fox.glb',               shape: 'cone',     color: 0xe67e22 },
  { file: 'DragonAttenuation.glb', shape: 'sphere',   color: 0x9b59b6 },
  { file: 'StainedGlassLamp.glb',  shape: 'cylinder', color: 0x3498db },
  { file: 'IridescenceLamp.glb',   shape: 'cylinder', color: 0x1abc9c },
  { file: 'CesiumMilkTruck.glb',   shape: 'box',      color: 0xd35400 },
];

const tablePositions = randomTablePositions(ASSETS.length);

ASSETS.forEach(({ file, shape, color }, idx) => {
  const { x, z } = tablePositions[idx];
  const name = file.replace('.glb', '');
  const uiColor = PALETTE[idx % PALETTE.length];

  loader.load(
    `/assets/${file}`,
    (gltf) => {
      const obj = normaliseGLTF(gltf, 0.25);
      placeOnTable(obj, x, z);
      objectRegistry.push({ name, worldX: x, worldZ: z, color: uiColor });
      window._objRegistry = objectRegistry;
    },
    undefined,
    () => {
      const obj = makeFallback(shape, color);
      placeOnTable(obj, x, z);
      objectRegistry.push({ name, worldX: x, worldZ: z, color: uiColor });
      window._objRegistry = objectRegistry;
    }
  );
});

// ── Animation Loop ────────────────────────────────────────────────────────

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
animate();

// ── Patch projection ──────────────────────────────────────────────────────
// With ortho camera at (camX, H, camZ) looking straight down, up=(0,0,-1):
//   screen-u (left→right) = (worldX - camX) / FRUSTUM_W + 0.5
//   screen-v (top→bottom) = (worldZ - camZ) / FRUSTUM_H + 0.5

function worldToUV(worldX, worldZ) {
  const camX = parseFloat(camXSlider.value);
  const camZ = parseFloat(camZSlider.value);
  return {
    u: (worldX - camX) / FRUSTUM_W + 0.5,
    v: (worldZ - camZ) / FRUSTUM_H + 0.5,
  };
}

function getObjectPatches(gridSize) {
  return objectRegistry.map(({ name, worldX, worldZ, color }) => {
    const { u, v } = worldToUV(worldX, worldZ);
    const col = Math.floor(u * gridSize);
    const row = Math.floor(v * gridSize);
    const inBounds = u >= 0 && u < 1 && v >= 0 && v < 1;
    const patchIdx = inBounds ? row * gridSize + col : -1;
    return { name, u, v, col, row, patchIdx, inBounds, color };
  });
}

// ── UI References ─────────────────────────────────────────────────────────

const computeBtn     = document.getElementById('compute-btn');
const snapshotBtn    = document.getElementById('snapshot-btn');
const clearSnapBtn   = document.getElementById('clear-snap-btn');
const snapIndicator  = document.getElementById('snap-indicator');
const statusEl       = document.getElementById('status');
const loadingOverlay = document.getElementById('loading-overlay');

const renderCanvas     = document.getElementById('render-canvas');
const snapCanvas       = document.getElementById('snap-canvas');
const patchColorCanvas = document.getElementById('patch-color-canvas');
const pcaCanvas        = document.getElementById('pca-canvas');
const tsneCanvas       = document.getElementById('tsne-canvas');
const simCanvas        = document.getElementById('sim-canvas');
const simProjCanvas    = document.getElementById('sim-proj-canvas');
const gridBadge        = document.getElementById('grid-badge');
const embBadge         = document.getElementById('emb-badge');
const simBadge         = document.getElementById('sim-badge');
const objLegend        = document.getElementById('obj-legend');

const toggleSiglip    = document.getElementById('toggle-siglip');
const toggleProj      = document.getElementById('toggle-proj');
const simSpatialBtn   = document.getElementById('sim-spatial-btn');
const simSemanticBtn  = document.getElementById('sim-semantic-btn');

// ── Server health check ───────────────────────────────────────────────────

async function waitForServer() {
  loadingOverlay.classList.add('show');
  while (true) {
    try { if ((await fetch('/health')).ok) break; } catch (_) {}
    await new Promise(r => setTimeout(r, 2000));
  }
  loadingOverlay.classList.remove('show');
  computeBtn.disabled = false;
  setStatus('Ready — adjust X slider then Compute');
}
waitForServer();

// ── Embedding mode toggle ─────────────────────────────────────────────────

toggleSiglip.addEventListener('click', () => {
  embeddingMode = 'siglip';
  toggleSiglip.classList.add('active'); toggleProj.classList.remove('active');
  if (currentResult) redrawScatter(currentResult, snapshotResult);
});
toggleProj.addEventListener('click', () => {
  embeddingMode = 'proj';
  toggleProj.classList.add('active'); toggleSiglip.classList.remove('active');
  if (currentResult) redrawScatter(currentResult, snapshotResult);
});

simSpatialBtn.addEventListener('click', () => {
  simMode = 'spatial';
  simSpatialBtn.classList.add('active'); simSemanticBtn.classList.remove('active');
  if (currentResult && snapshotResult) drawSimilarity(currentResult, snapshotResult);
});
simSemanticBtn.addEventListener('click', () => {
  simMode = 'semantic';
  simSemanticBtn.classList.add('active'); simSpatialBtn.classList.remove('active');
  if (currentResult && snapshotResult) drawSimilarity(currentResult, snapshotResult);
});

// ── Compute embeddings ────────────────────────────────────────────────────

computeBtn.addEventListener('click', computeEmbeddings);

async function computeEmbeddings() {
  computeBtn.disabled = true;
  setStatus('Capturing scene...');
  renderer.render(scene, camera);
  const dataUrl = renderer.domElement.toDataURL('image/png');
  const base64  = dataUrl.split(',')[1];

  setStatus('Running PaliGemma...');
  try {
    const resp = await fetch('/embed', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64 }),
    });
    if (!resp.ok) throw new Error(await resp.text());
    currentResult = await resp.json();

    // Attach current object→patch map to the result for later comparison
    currentResult._objPatches = getObjectPatches(currentResult.grid_size);

    // Cache render image for patch thumbnail scatter plots
    renderImg = new Image();
    renderImg.src = dataUrl;

    drawRenderTarget(dataUrl, currentResult);
    drawPatchColorMap(currentResult);
    redrawScatter(currentResult, snapshotResult);
    updateObjectLegend(currentResult._objPatches, currentResult, snapshotResult);
    if (snapshotResult) drawSimilarity(currentResult, snapshotResult);

    snapshotBtn.disabled = false;
    gridBadge.textContent = `${currentResult.grid_size}×${currentResult.grid_size} patches`;
    embBadge.textContent  = `SigLIP:${currentResult.patch_dim}d  Proj:${currentResult.proj_dim}d`;
    setStatus(`Done — ${currentResult.num_patches} patches`);
  } catch (e) {
    setStatus(`Error: ${e.message}`);
  }
  computeBtn.disabled = false;
}

// ── Snapshot ──────────────────────────────────────────────────────────────

snapshotBtn.addEventListener('click', () => {
  if (!currentResult) return;
  snapshotResult  = currentResult;
  snapshotPatches = Object.fromEntries(
    (currentResult._objPatches || []).map(o => [o.name, o.patchIdx])
  );

  const snapDataUrl = renderer.domElement.toDataURL('image/png');
  snapImg = new Image();
  snapImg.src = snapDataUrl;

  const img = new Image();
  img.onload = () => {
    const ctx = snapCanvas.getContext('2d');
    ctx.clearRect(0, 0, 256, 256);
    ctx.drawImage(img, 0, 0, 256, 256);
  };
  img.src = snapDataUrl;
  snapCanvas.style.display = 'block';

  snapIndicator.style.display = 'inline';
  clearSnapBtn.style.display  = 'inline';
  simBadge.textContent = 'snapshot active';
  setStatus('Snapshot saved — slide X then Compute');
});

clearSnapBtn.addEventListener('click', () => {
  snapshotResult = null; snapshotPatches = null;
  snapCanvas.style.display = 'none';
  snapCanvas.getContext('2d').clearRect(0, 0, 256, 256);
  snapIndicator.style.display = 'none';
  clearSnapBtn.style.display  = 'none';
  simBadge.textContent = 'no snapshot';
  clearCanvas(simCanvas); clearCanvas(simProjCanvas);
  if (currentResult) {
    redrawScatter(currentResult, null);
    updateObjectLegend(currentResult._objPatches, currentResult, null);
  }
});

// ── Draw: Render Target + patch grid + object dots ───────────────────────

function drawRenderTarget(dataUrl, result) {
  const ctx  = renderCanvas.getContext('2d');
  const g    = result.grid_size;
  const cell = 256 / g;

  const img = new Image();
  img.onload = () => {
    ctx.clearRect(0, 0, 256, 256);
    ctx.drawImage(img, 0, 0, 256, 256);

    // Patch grid
    ctx.strokeStyle = 'rgba(255, 220, 0, 0.35)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= g; i++) {
      const p = i * cell;
      ctx.beginPath(); ctx.moveTo(p, 0);   ctx.lineTo(p, 256); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p);   ctx.lineTo(256, p); ctx.stroke();
    }

    // Object dots
    (result._objPatches || []).forEach(({ u, v, inBounds, color, name }) => {
      if (!inBounds) return;
      const px = u * 256, py = v * 256;
      ctx.fillStyle   = color;
      ctx.strokeStyle = '#000';
      ctx.lineWidth   = 1;
      ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fill(); ctx.stroke();

      // Short label
      ctx.fillStyle = '#fff';
      ctx.font = '7px monospace';
      ctx.fillText(name.slice(0, 5), px + 5, py - 3);
    });
  };
  img.src = dataUrl;
}

// ── Draw: Patch PCA-RGB coloring + object outlines ───────────────────────

function drawPatchColorMap(result) {
  const ctx  = patchColorCanvas.getContext('2d');
  const g    = result.grid_size;
  const cell = 256 / g;
  ctx.clearRect(0, 0, 256, 256);

  result.patch_pca_rgb.forEach(([r, gr, b], i) => {
    const row = Math.floor(i / g), col = i % g;
    ctx.fillStyle = `rgb(${Math.round(r*255)},${Math.round(gr*255)},${Math.round(b*255)})`;
    ctx.fillRect(col * cell, row * cell, cell, cell);
  });

  // Grid lines
  ctx.strokeStyle = 'rgba(0,0,0,0.25)'; ctx.lineWidth = 0.5;
  for (let i = 0; i <= g; i++) {
    const p = i * cell;
    ctx.beginPath(); ctx.moveTo(p, 0);   ctx.lineTo(p, 256); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, p);   ctx.lineTo(256, p); ctx.stroke();
  }

  // Highlight object patches
  (result._objPatches || []).forEach(({ col, row, inBounds, color }) => {
    if (!inBounds) return;
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.strokeRect(col * cell + 1, row * cell + 1, cell - 2, cell - 2);
  });
}

// ── Draw: Scatter with patch image thumbnails ─────────────────────────────

function normCoords(pts, W, H, margin = 18) {
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const mnX = Math.min(...xs), mxX = Math.max(...xs);
  const mnY = Math.min(...ys), mxY = Math.max(...ys);
  const rX = mxX - mnX || 1, rY = mxY - mnY || 1;
  return pts.map(p => [
    margin + ((p[0] - mnX) / rX) * (W - 2 * margin),
    margin + ((p[1] - mnY) / rY) * (H - 2 * margin),
  ]);
}

function drawScatterThumbnails(cvs, coords, gridSize, snapCoords, snapSrcImg, objPatches, srcImg) {
  const ctx = cvs.getContext('2d');
  const W = cvs.width, H = cvs.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#111115'; ctx.fillRect(0, 0, W, H);

  const THUMB = 18;   // thumbnail side length in px
  const objPatchMap = new Map((objPatches || []).filter(o => o.inBounds).map(o => [o.patchIdx, o]));

  // Draw snapshot thumbnails first (faded, behind current)
  if (snapCoords && snapSrcImg?.complete) {
    const sn = normCoords(snapCoords, W, H);
    ctx.globalAlpha = 0.28;
    sn.forEach(([x, y], i) => {
      const row = Math.floor(i / gridSize), col = i % gridSize;
      const sx = (col / gridSize) * snapSrcImg.width;
      const sy = (row / gridSize) * snapSrcImg.height;
      ctx.drawImage(snapSrcImg, sx, sy, snapSrcImg.width / gridSize, snapSrcImg.height / gridSize,
        x - THUMB / 2, y - THUMB / 2, THUMB, THUMB);
    });
    ctx.globalAlpha = 1;
  }

  // Draw current thumbnails
  const cn = normCoords(coords, W, H);
  cn.forEach(([x, y], i) => {
    const row = Math.floor(i / gridSize), col = i % gridSize;

    if (srcImg?.complete) {
      const sx = (col / gridSize) * srcImg.width;
      const sy = (row / gridSize) * srcImg.height;
      ctx.drawImage(srcImg, sx, sy, srcImg.width / gridSize, srcImg.height / gridSize,
        x - THUMB / 2, y - THUMB / 2, THUMB, THUMB);
    } else {
      ctx.fillStyle = '#333';
      ctx.fillRect(x - THUMB / 2, y - THUMB / 2, THUMB, THUMB);
    }

    // Colored border for object-containing patches
    const obj = objPatchMap.get(i);
    if (obj) {
      ctx.strokeStyle = obj.color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x - THUMB / 2 - 1, y - THUMB / 2 - 1, THUMB + 2, THUMB + 2);
      // Short label above
      ctx.fillStyle = obj.color;
      ctx.font = 'bold 8px monospace';
      ctx.fillText(obj.name.slice(0, 5), x - THUMB / 2, y - THUMB / 2 - 2);
    }
  });
}

function redrawScatter(result, snap) {
  const isSig = embeddingMode === 'siglip';
  const currCoords = isSig ? result.pca_coords : result.pca_proj_coords;
  const snapCoords = snap ? (isSig ? snap.pca_coords : snap.pca_proj_coords) : null;

  drawScatterThumbnails(pcaCanvas,  currCoords,           result.grid_size,
    snapCoords,          snapImg, result._objPatches, renderImg);
  drawScatterThumbnails(tsneCanvas, result.tsne_coords,   result.grid_size,
    snap?.tsne_coords,   snapImg, result._objPatches, renderImg);
}

// ── Draw: Cosine Similarity heatmap + object patch highlights ────────────

function cosSim(a, b) { return a.reduce((s, v, i) => s + v * b[i], 0); }

// For each current patch, find max cosine sim to ANY snapshot patch (semantic mode)
function semanticSims(currEmbs, snapEmbs) {
  return currEmbs.map(ce => Math.max(...snapEmbs.map(se => cosSim(ce, se))));
}

function drawSimilarity(current, snapshot) {
  if (!snapshot) return;
  const g    = current.grid_size;
  const SIZE = 512;                 // internal canvas size (displayed at 256 via CSS)
  const cell = SIZE / g;

  function heatmap(cvs, currEmbs, snapEmbs) {
    const ctx = cvs.getContext('2d');
    ctx.clearRect(0, 0, SIZE, SIZE);

    const sims = simMode === 'semantic'
      ? semanticSims(currEmbs, snapEmbs)
      : currEmbs.map((ce, i) => cosSim(ce, snapEmbs[i] || ce));

    sims.forEach((sim, i) => {
      const row = Math.floor(i / g), col = i % g;
      const t = (sim + 1) / 2;
      ctx.fillStyle = `rgb(${Math.round(t*255)},${Math.round(Math.min(t,1-t)*2*200)},${Math.round((1-t)*255)})`;
      ctx.fillRect(col * cell, row * cell, cell, cell);

      // Sim value — readable at 512px (cell = 32px)
      ctx.fillStyle = 'rgba(0,0,0,0.75)';
      ctx.font = `bold ${Math.round(cell * 0.38)}px monospace`;
      ctx.textAlign = 'center';
      ctx.fillText(sim.toFixed(2), col * cell + cell / 2, row * cell + cell * 0.65);
    });
    ctx.textAlign = 'left';

    // Grid lines
    ctx.strokeStyle = 'rgba(0,0,0,0.3)'; ctx.lineWidth = 0.5;
    for (let i = 0; i <= g; i++) {
      const p = i * cell;
      ctx.beginPath(); ctx.moveTo(p, 0);    ctx.lineTo(p, SIZE); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p);    ctx.lineTo(SIZE, p); ctx.stroke();
    }

    // Solid border = object's CURRENT patch
    (current._objPatches || []).forEach(({ col, row, inBounds, color }) => {
      if (!inBounds) return;
      ctx.strokeStyle = color; ctx.lineWidth = 3;
      ctx.strokeRect(col * cell + 2, row * cell + 2, cell - 4, cell - 4);
    });

    // Dashed border = where object WAS in snapshot
    if (snapshotPatches) {
      (current._objPatches || []).forEach(({ name, color }) => {
        const snapIdx = snapshotPatches[name];
        if (snapIdx == null || snapIdx < 0) return;
        const sRow = Math.floor(snapIdx / g), sCol = snapIdx % g;
        ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash([4, 3]);
        ctx.strokeRect(sCol * cell + 2, sRow * cell + 2, cell - 4, cell - 4);
        ctx.setLineDash([]);
      });
    }

    // Legend
    const grad = ctx.createLinearGradient(4, 0, SIZE - 4, 0);
    grad.addColorStop(0,   'rgb(0,0,255)');
    grad.addColorStop(0.5, 'rgb(200,200,200)');
    grad.addColorStop(1,   'rgb(255,0,0)');
    ctx.fillStyle = grad; ctx.fillRect(4, SIZE - 10, SIZE - 8, 8);
  }

  heatmap(simCanvas,     current.patch_embeddings_norm, snapshot.patch_embeddings_norm);
  heatmap(simProjCanvas, current.proj_embeddings_norm,  snapshot.proj_embeddings_norm);
  simBadge.textContent = `${simMode} · updated`;
}

// ── Object→Patch legend ───────────────────────────────────────────────────

function updateObjectLegend(objPatches, current, snap) {
  if (!objPatches) return;
  const g = current.grid_size;
  objLegend.innerHTML = '';

  objPatches.forEach(({ name, patchIdx, col, row, inBounds, color }) => {
    const chip = document.createElement('span');
    chip.className = 'obj-chip';
    chip.style.borderColor = color;
    chip.style.color = color;

    let label = inBounds ? `${name.slice(0, 6)} (${col},${row})` : `${name.slice(0, 6)} [off]`;

    if (snap && snapshotPatches) {
      const sIdx = snapshotPatches[name];
      if (sIdx != null && sIdx >= 0 && patchIdx >= 0) {
        const sRow = Math.floor(sIdx / g), sCol = sIdx % g;
        const sim = cosSim(
          current.patch_embeddings_norm[patchIdx],
          snap.patch_embeddings_norm[sIdx]
        );
        label += ` → (${sCol},${sRow}) sim=${sim.toFixed(2)}`;
      }
    }
    chip.textContent = label;
    objLegend.appendChild(chip);
  });
}

// ── Zoom modal ────────────────────────────────────────────────────────────

const zoomModal = document.getElementById('zoom-modal');
const zoomImg   = document.getElementById('zoom-img');
const zoomTitle = document.getElementById('zoom-title');

function openZoom(getDataUrl, title) {
  zoomImg.src = getDataUrl();
  zoomTitle.textContent = title;
  zoomModal.classList.add('show');
}

zoomModal.addEventListener('click', () => zoomModal.classList.remove('show'));
document.addEventListener('keydown', e => { if (e.key === 'Escape') zoomModal.classList.remove('show'); });

// Composite render canvas + ghost snapshot overlay into one image
function renderTargetDataUrl() {
  const tmp = document.createElement('canvas');
  tmp.width = renderCanvas.width; tmp.height = renderCanvas.height;
  const ctx = tmp.getContext('2d');
  ctx.drawImage(renderCanvas, 0, 0);
  if (snapCanvas.style.display !== 'none') {
    ctx.globalAlpha = 0.45;
    ctx.drawImage(snapCanvas, 0, 0);
    ctx.globalAlpha = 1;
  }
  return tmp.toDataURL();
}

// Register all zoomable canvases
[
  [renderCanvas,     () => renderTargetDataUrl(),         'Render Target + Patches'],
  [patchColorCanvas, () => patchColorCanvas.toDataURL(),  'Patch PCA-RGB Coloring'],
  [pcaCanvas,        () => pcaCanvas.toDataURL(),         'PCA 2D — Patch Embeddings'],
  [tsneCanvas,       () => tsneCanvas.toDataURL(),        't-SNE 2D — Patch Embeddings'],
  [simCanvas,        () => simCanvas.toDataURL(),         'Cosine Similarity — SigLIP Space'],
  [simProjCanvas,    () => simProjCanvas.toDataURL(),     'Cosine Similarity — Projected (Gemma)'],
].forEach(([cvs, getUrl, title]) => {
  cvs.classList.add('zoomable');
  cvs.title = 'Click to enlarge';
  cvs.addEventListener('click', () => openZoom(getUrl, title));
});

// ── Utilities ─────────────────────────────────────────────────────────────

function clearCanvas(cvs) { cvs.getContext('2d').clearRect(0, 0, cvs.width, cvs.height); }
function setStatus(msg)    { statusEl.textContent = msg; }
