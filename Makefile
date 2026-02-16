sync:
	uv sync --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match

check:
	uv run python scripts/00_env_check.py

baseline:
	uv run bash scripts/01_download_awq_baseline.sh

compress:
	uv run python scripts/05_awq_quantize.py --model-dir models/base --output-dir models/compressed-l29 --target-layers 29 --selection uniform --dtype float16

verify:
	uv run python scripts/02_verify_vllm.py --model-dir models/base --report-file outputs/verify_vllm.json

verify-safe:
	uv run python scripts/02_verify_vllm.py --model-dir models/base --enforce-eager --dtype float16 --report-file outputs/verify_vllm_safe.json

verify-tfm:
	uv run python scripts/02_verify_transformers.py --model-dir models/base

verify-tfm-compressed:
	uv run python scripts/02_verify_transformers.py --model-dir models/compressed-l29

verify-compressed:
	uv run python scripts/02_verify_vllm.py --model-dir models/compressed-l29 --enforce-eager --dtype float16 --report-file outputs/verify_vllm_compressed.json

package:
	uv run python scripts/06_package_submit.py --model-dir models/base --output submit.zip

package-compressed:
	uv run python scripts/06_package_submit.py --model-dir models/compressed-l29 --output submit_compressed_l29.zip
