#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-LGAI-EXAONE/EXAONE-4.0-1.2B}"
TARGET_DIR="${TARGET_DIR:-models/base}"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
target_dir = os.environ["TARGET_DIR"]

snapshot_download(
    repo_id=model_id,
    local_dir=target_dir,
    local_dir_use_symlinks=False,
)

print(f"Downloaded {model_id} -> {target_dir}")
PY
