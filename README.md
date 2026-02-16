# EXAONE Compression Toolkit

This repo is a reproducible pipeline for EXAONE-4.0-1.2B compression work.
The evaluator ignores code and only reads `submit.zip` -> `model/`, so this repo
exists to make your RunPod execution deterministic and repeatable.

## Quick Start

```bash
git clone <your-repo-url>
cd exaone-compression

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

uv sync --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
```

Run checks and pipeline steps:

```bash
uv run python scripts/00_env_check.py
uv run bash scripts/01_download_awq_baseline.sh
uv run python scripts/02_verify_vllm.py --model-dir models/base --report-file outputs/verify_vllm.json
uv run python scripts/02_verify_transformers.py --model-dir models/base
uv run python scripts/06_package_submit.py --model-dir models/base --output submit.zip
```

## Layout

```
exaone-compression/
  pyproject.toml
  uv.lock
  scripts/
    00_env_check.py
    01_download_awq_baseline.sh
    02_verify_vllm.py
    02_verify_transformers.py
    03_lora_train.py
    04_merge_lora.py
    05_awq_quantize.py
    06_package_submit.py
  configs/
    lora.yaml
    awq.json
  README.md
  .gitignore
  Makefile
```

## Notes

- The evaluation server has no internet access.
- Pin to the exact dependency versions to avoid mismatch.
- `submit.zip` must contain a top-level `model/` directory only.
- If vLLM crashes on first load, rerun verification in eager mode:
  `uv run python scripts/02_verify_vllm.py --model-dir models/base --enforce-eager --dtype float16`.

## Environment Variables

Use a persistent cache location on RunPod to avoid repeated downloads:

```bash
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
```
