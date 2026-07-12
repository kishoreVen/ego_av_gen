# Brain Factory Watch - Implementation Plan

## Tech Stack
- **Frontend**: React 19 + TypeScript + Vite (matches story_creator)
- **Backend**: Express API server (same process via Vite plugin or separate)
- **Styling**: Vanilla CSS with CSS modules (keep it simple, dark theme)
- **Charts**: Lightweight chart lib (recharts) for loss curves
- **No external DB** — reads directly from `brain_factory_out/` filesystem

## Architecture

Single Vite dev server with an Express API backend (via `vite-plugin-api` or a parallel Express process). The Express server reads `../../brain_factory_out/` and serves run metadata, metrics, configs, and static output files.

### Backend API (`server/`)

| Endpoint | Purpose |
|----------|---------|
| `GET /api/runs` | Scan `brain_factory_out/` → list all `{experiment, timestamp, hasMetrics, hasCheckpoints}` |
| `GET /api/runs/:experiment/:timestamp/config` | Read `.hydra/config.yaml` → return parsed YAML |
| `GET /api/runs/:experiment/:timestamp/metrics` | Read `metrics.jsonl` → return JSON array |
| `GET /api/runs/:experiment/:timestamp/checkpoints` | List `checkpoints/` subdirs with sizes |
| `GET /api/runs/:experiment/:timestamp/outputs` | List `visualizations/` tree (images, videos, JSON) |
| `GET /api/runs/:experiment/:timestamp/files/*` | Serve static files (images, videos) from run dir |
| `POST /api/tensorboard/start` | Spawn `tensorboard --logdir=<tb_dir> --port=<dynamic>`, return port |
| `POST /api/tensorboard/stop` | Kill tensorboard subprocess |
| `POST /api/actions/train` | Spawn `python -m brain_factory.main projects=<project>/... [overrides]` |
| `POST /api/actions/inference` | Spawn inference with checkpoint path override |

### Frontend Pages

**1. Runs List (`/`)**
- Table grouped by experiment name
- Columns: experiment, timestamp, status (has metrics? still writing?), steps, last loss
- Click row → navigate to run detail
- Auto-refresh every 5s

**2. Run Detail (`/run/:experiment/:timestamp`)** — Tabbed layout:

**Tab A: Config**
- Pretty-print the resolved Hydra config (collapsible YAML tree)
- Also show overrides.yaml as a separate section

**Tab B: TensorBoard**
- On tab open, POST `/api/tensorboard/start` with this run's tb dir
- Embed TensorBoard in an iframe at the returned port
- On tab leave / unmount, POST `/api/tensorboard/stop`

**Tab C: Outputs (Visualizations)**
- Scan `visualizations/` directory
- For each step folder, show a card/gallery
- Auto-detect file types: images (PNG/JPG) → gallery, MP4 → video player, JSON → pretty-print
- This is the "flexible matching" — any output container just drops files, the viewer renders by extension

**Tab D: Checkpoints & Metrics**
- Loss curve chart (from metrics.jsonl) — multi-line for loss/total, loss/flow_loss, lr, etc.
- Interactive: hover for values, zoom
- Below chart: list of checkpoints with step number, file sizes
- Click checkpoint → shows the recipe_config.yaml snapshot

**Tab E: Actions**
- "Restart Training" button → form pre-filled with this run's config
  - Toggle: "from scratch" vs "resume from checkpoint" (dropdown of available checkpoints)
  - Editable overrides text area
  - Submit → POST `/api/actions/train`
- "Run Inference" button → form with checkpoint selector
  - Submit → POST `/api/actions/inference`
- Show spawned process output in a live terminal-like log viewer

## File Structure

```
web/brain_factory_watch/
├── package.json
├── tsconfig.json
├── tsconfig.node.json
├── vite.config.ts
├── index.html
├── server/
│   ├── index.ts          # Express app entry
│   ├── routes/
│   │   ├── runs.ts       # /api/runs endpoints
│   │   ├── tensorboard.ts # TB process management
│   │   └── actions.ts    # Train/inference spawning
│   └── utils.ts          # FS scanning helpers
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── App.css
│   ├── pages/
│   │   ├── RunsList.tsx
│   │   └── RunDetail.tsx
│   ├── components/
│   │   ├── ConfigViewer.tsx
│   │   ├── TensorBoardEmbed.tsx
│   │   ├── OutputsGallery.tsx
│   │   ├── MetricsChart.tsx
│   │   ├── CheckpointList.tsx
│   │   ├── ActionPanel.tsx
│   │   └── ProcessLog.tsx
│   ├── hooks/
│   │   └── useApi.ts
│   └── types.ts
```

## Implementation Order

1. Scaffold project (package.json, vite config, tsconfig, index.html)
2. Express backend — runs listing + config + metrics + checkpoints + file serving
3. Frontend shell — routing, runs list page
4. Run detail page — config tab, metrics/checkpoints tab
5. TensorBoard embedding (subprocess management)
6. Outputs gallery (file-type-aware rendering)
7. Actions panel (train/inference spawning + live log)
