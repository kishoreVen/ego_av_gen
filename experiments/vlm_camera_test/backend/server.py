"""
FastAPI server for VLM camera test.
Loads PaliGemma-3B and exposes a /embed endpoint that takes a rendered
scene image and returns SigLIP patch embeddings + PCA/t-SNE coordinates.

Requires:
  huggingface-cli login
  Accept license at https://huggingface.co/google/paligemma-3b-pt-224
"""

import base64
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_ID = "google/paligemma-3b-pt-224"
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
ASSETS_DIR   = Path(__file__).parent.parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

model = None
processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    log.info("Loading PaliGemma-3B...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    log.info("Model ready.")
    yield
    del model, processor
    torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


class EmbedRequest(BaseModel):
    image: str  # base64-encoded PNG


def cosine_norm(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norms + 1e-8)


@app.post("/embed")
async def embed(req: EmbedRequest):
    img_bytes = base64.b64decode(req.image)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to("cuda")

    with torch.no_grad():
        vision_out = model.model.vision_tower(pixel_values=inputs["pixel_values"])
        # all_hidden: (1, num_patches+1, siglip_dim)  — index 0 is CLS
        all_hidden = vision_out.last_hidden_state

        projected_all = model.model.multi_modal_projector(all_hidden)

    # Strip CLS token; patches are 1: onwards
    patch_embs = all_hidden[0, 1:].cpu().float().numpy()      # (N, 1152)
    proj_embs  = projected_all[0, 1:].cpu().float().numpy()   # (N, 2048)

    N = patch_embs.shape[0]
    grid_size = int(round(N ** 0.5))

    # PCA (2D) on patch embeddings (SigLIP space)
    pca2 = PCA(n_components=2)
    pca_coords = pca2.fit_transform(patch_embs).tolist()

    # PCA (2D) on projected tokens (Gemma space)
    pca2_proj = PCA(n_components=2)
    pca_proj_coords = pca2_proj.fit_transform(proj_embs).tolist()

    # t-SNE via PCA-50 pre-reduction for speed
    n_components_pre = min(50, N - 1, patch_embs.shape[1])
    pca_pre = PCA(n_components=n_components_pre)
    reduced = pca_pre.fit_transform(patch_embs)
    tsne = TSNE(n_components=2, perplexity=min(30, N // 2), random_state=42, max_iter=1000)
    tsne_coords = tsne.fit_transform(reduced).tolist()

    # Normalised patch embeddings for client-side cosine similarity
    patch_embs_norm = cosine_norm(patch_embs).tolist()
    proj_embs_norm  = cosine_norm(proj_embs).tolist()

    # First-PCA component reshaped to grid (useful for patch coloring)
    pca3 = PCA(n_components=3)
    pca3_coords = pca3.fit_transform(patch_embs)
    # Normalise each component to [0,1] for RGB coloring
    for c in range(3):
        mn, mx = pca3_coords[:, c].min(), pca3_coords[:, c].max()
        pca3_coords[:, c] = (pca3_coords[:, c] - mn) / (mx - mn + 1e-8)
    patch_pca_rgb = pca3_coords.tolist()

    return JSONResponse({
        "patch_embeddings_norm": patch_embs_norm,
        "proj_embeddings_norm":  proj_embs_norm,
        "pca_coords":      pca_coords,
        "pca_proj_coords": pca_proj_coords,
        "tsne_coords":     tsne_coords,
        "patch_pca_rgb":   patch_pca_rgb,
        "grid_size":       grid_size,
        "num_patches":     N,
        "patch_dim":       int(patch_embs.shape[1]),
        "proj_dim":        int(proj_embs.shape[1]),
    })


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "gpu": torch.cuda.get_device_name(0)}


# Serve frontend static files and downloaded 3D assets
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)),   name="assets")


@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
