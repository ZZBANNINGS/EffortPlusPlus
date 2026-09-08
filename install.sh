#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m pip install \
  torch==1.12.0+cu113 \
  torchvision==0.13.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113

"${PYTHON_BIN}" -m pip install -r "${PROJECT_ROOT}/requirements.txt"

CLIP_DIR="${PROJECT_ROOT}/DeepfakeBench/training/models--openai--clip-vit-large-patch14"
CLIP_DIR="${CLIP_DIR}" "${PYTHON_BIN}" -c '
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="openai/clip-vit-large-patch14",
    local_dir=os.environ["CLIP_DIR"],
    allow_patterns=("config.json", "model.safetensors", "preprocessor_config.json"),
)
'

echo "Installation complete."
