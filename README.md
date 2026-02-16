# EXAONE Compression Toolkit

Reproducible pipeline for EXAONE-4.0-1.2B compression targeting the hackathon
evaluation format (`submit.zip` containing only `model/`).

This README documents only the workflow that was validated end-to-end.

## Proven Outcome

- Baseline model: `models/base`
- Compressed model: `models/compressed-l29` (uniform layer-drop, 30 -> 29 layers)
- Final artifact: `submit_compressed_l29.zip`
- vLLM and Transformers load/generate both succeeded for the compressed model.

## Environment

- Python: `3.11.x`
- CUDA: `12.8`
- vLLM: `0.14.1`
- Torch stack pinned to evaluation target:
  - `torch==2.9.0+cu128`
  - `torchaudio==2.9.0+cu128`
  - `torchvision==0.24.0+cu128`
  - `triton==3.5.0`

## From Scratch (RunPod)

```bash
git clone <your-repo-url>
cd exaone-compression

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
```

Run the full pipeline:

```bash
make sync
make check
make baseline
make compress
make verify
make verify-compressed
make verify-tfm
make verify-tfm-compressed
make package-compressed
```

Primary outputs:

- Compressed model: `models/compressed-l29`
- vLLM report: `outputs/verify_vllm_compressed.json`
- Submission zip: `submit_compressed_l29.zip`

## Submission Format Check

```bash
python - <<'PY'
import zipfile
z = zipfile.ZipFile("submit_compressed_l29.zip")
roots = sorted({n.split("/")[0] for n in z.namelist() if n.strip()})
print("top_level_entries:", roots)
print("valid:", roots == ["model"])
PY
```

Expected output:

- `top_level_entries: ['model']`
- `valid: True`

## Notes

- `scripts/05_awq_quantize.py` is the compression script (name retained for
  compatibility with older pipeline naming).
- `vllm==0.14.1` metadata declares `torch==2.9.1`, but this project enforces
  evaluation-compatible `torch==2.9.0+cu128` through `pyproject.toml` overrides.
  Use runtime smoke tests (`make verify*`) as the source of truth.
- If baseline `make verify` is unstable on your host, run:
  - `make verify-safe`
