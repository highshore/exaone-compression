#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-LGAI-EXAONE/EXAONE-4.0-1.2B}"
TARGET_DIR="${TARGET_DIR:-open/base_model}"
RETRIES="${RETRIES:-3}"
STRICT_BASELINE="${STRICT_BASELINE:-1}"
export MODEL_ID TARGET_DIR STRICT_BASELINE

if [[ "${HF_HUB_ENABLE_HF_TRANSFER:-0}" == "1" ]]; then
  if ! python -c "import hf_transfer" >/dev/null 2>&1; then
    echo "WARN: HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer is not installed."
    echo "WARN: install it with: uv pip install hf_transfer"
  fi
fi

run_download() {
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
}

attempt=1
while true; do
  if run_download; then
    break
  fi

  if [[ "$attempt" -ge "$RETRIES" ]]; then
    echo "ERROR: download failed after ${RETRIES} attempts."
    exit 1
  fi

  backoff=$((attempt * 5))
  echo "WARN: download attempt ${attempt} failed. Retrying in ${backoff}s..."
  sleep "$backoff"
  attempt=$((attempt + 1))
done

python - <<'PY'
import json
import os
import pathlib
import sys

model_id = os.environ["MODEL_ID"]
target_dir = pathlib.Path(os.environ["TARGET_DIR"]).resolve()
strict_baseline = os.environ.get("STRICT_BASELINE", "1") == "1"

config_path = target_dir / "config.json"
weights_path = target_dir / "model.safetensors"

if not config_path.is_file():
    raise SystemExit(f"Missing config: {config_path}")
if not weights_path.is_file():
    raise SystemExit(f"Missing weights: {weights_path}")

config = json.loads(config_path.read_text(encoding="utf-8"))
quant_cfg = config.get("quantization_config")
size_gb = weights_path.stat().st_size / (1024 ** 3)

print(f"Model ID: {model_id}")
print(f"Architecture: {config.get('architectures')}")
print(f"torch_dtype: {config.get('torch_dtype')}")
print(f"quantization_config: {quant_cfg}")
print(f"model.safetensors size: {size_gb:.2f} GiB")

# The official EXAONE-4.0-1.2B baseline is expected to be non-quantized.
if strict_baseline and model_id == "LGAI-EXAONE/EXAONE-4.0-1.2B" and quant_cfg is not None:
    raise SystemExit("Baseline model unexpectedly has quantization_config set.")

print("OK: model download validation passed")
PY
