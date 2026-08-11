#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Download 3D assets if not already present
echo "==> Checking 3D assets..."
python download_assets.py

# Start the FastAPI server (serves both backend API and frontend)
echo ""
echo "==> Starting server at http://localhost:8888"
echo "    First run will download PaliGemma-3B (~6 GB) and may take a few minutes."
echo "    Requires:  huggingface-cli login  +  accepted license at"
echo "    https://huggingface.co/google/paligemma-3b-pt-224"
echo ""
uvicorn backend.server:app --host 0.0.0.0 --port 8888
