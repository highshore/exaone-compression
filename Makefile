sync:
	uv sync --extra-index-url https://download.pytorch.org/whl/cu128

check:
	uv run python scripts/00_env_check.py

baseline:
	uv run bash scripts/01_download_awq_baseline.sh

verify:
	uv run python scripts/02_verify_vllm.py --model-dir models/base

package:
	uv run python scripts/06_package_submit.py --model-dir models/base --output submit.zip
