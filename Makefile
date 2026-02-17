BASE_MODEL_DIR ?= open/base_model
LORA_ADAPTER_DIR ?= outputs/lora/compressed-l29
RECOVERED_MODEL_DIR ?= models/compressed-l29-lora-recovered
CANDIDATE_MODELS ?= models/compressed-l29 models/compressed-l29-distilled $(RECOVERED_MODEL_DIR)

sync:
	uv sync --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match

check:
	uv run python scripts/00_env_check.py

baseline:
	@if [ -f "$(BASE_MODEL_DIR)/config.json" ] && [ -f "$(BASE_MODEL_DIR)/model.safetensors" ]; then \
		echo "Using existing baseline model at $(BASE_MODEL_DIR)"; \
	else \
		echo "Baseline not found at $(BASE_MODEL_DIR). Downloading from HF..."; \
		TARGET_DIR="$(BASE_MODEL_DIR)" uv run bash scripts/01_download_awq_baseline.sh; \
	fi

compress:
	uv run python scripts/05_awq_quantize.py --model-dir $(BASE_MODEL_DIR) --output-dir models/compressed-l29 --target-layers 29 --selection uniform --dtype float16

distill:
	uv run python scripts/07_distill_student.py --teacher-model $(BASE_MODEL_DIR) --student-model models/compressed-l29 --output-dir models/compressed-l29-distilled --num-samples 20 --max-new-tokens 64 --epochs 1 --batch-size 2 --learning-rate 5e-6 --report-file outputs/distill_report.json --dataset-file outputs/distill_dataset.jsonl

distill-lora:
	uv run python scripts/03_lora_train.py --teacher-model $(BASE_MODEL_DIR) --student-model models/compressed-l29 --output-dir $(LORA_ADAPTER_DIR) --dataset-file outputs/distill_lora_dataset.jsonl --report-file outputs/lora_train_report.json --num-samples 800 --max-new-tokens 96 --epochs 1 --batch-size 2 --learning-rate 2e-4

merge-lora:
	uv run python scripts/04_merge_lora.py --base-model models/compressed-l29 --lora-dir $(LORA_ADAPTER_DIR) --output-dir $(RECOVERED_MODEL_DIR) --report-file outputs/lora_merge_report.json

recover-lora: distill-lora merge-lora

evaluate:
	uv run python scripts/08_eval_compare.py --baseline-model $(BASE_MODEL_DIR) --candidate-models models/compressed-l29 models/compressed-l29-distilled --report-file outputs/eval_compare.json

eval-quality:
	uv run python scripts/09_eval_quality.py --baseline-model $(BASE_MODEL_DIR) --candidate-models $(CANDIDATE_MODELS) --datasets mmlu,boolq --limit-per-dataset 64 --report-file outputs/eval_quality.json

eval-token-speed:
	uv run python scripts/10_eval_token_latency.py --baseline-model $(BASE_MODEL_DIR) --candidate-models $(CANDIDATE_MODELS) --rounds 2 --report-file outputs/eval_token_latency.json

verify:
	uv run python scripts/02_verify_vllm.py --model-dir $(BASE_MODEL_DIR) --report-file outputs/verify_vllm.json

verify-safe:
	uv run python scripts/02_verify_vllm.py --model-dir $(BASE_MODEL_DIR) --enforce-eager --dtype float16 --report-file outputs/verify_vllm_safe.json

verify-tfm:
	uv run python scripts/02_verify_transformers.py --model-dir $(BASE_MODEL_DIR)

verify-tfm-compressed:
	uv run python scripts/02_verify_transformers.py --model-dir models/compressed-l29

verify-tfm-distilled:
	uv run python scripts/02_verify_transformers.py --model-dir models/compressed-l29-distilled

verify-compressed:
	uv run python scripts/02_verify_vllm.py --model-dir models/compressed-l29 --enforce-eager --dtype float16 --report-file outputs/verify_vllm_compressed.json

verify-distilled:
	uv run python scripts/02_verify_vllm.py --model-dir models/compressed-l29-distilled --enforce-eager --dtype float16 --report-file outputs/verify_vllm_distilled.json

package:
	uv run python scripts/06_package_submit.py --model-dir $(BASE_MODEL_DIR) --output submit.zip

package-compressed:
	uv run python scripts/06_package_submit.py --model-dir models/compressed-l29 --output submit_compressed_l29.zip

package-distilled:
	uv run python scripts/06_package_submit.py --model-dir models/compressed-l29-distilled --output submit_compressed_l29_distilled.zip
