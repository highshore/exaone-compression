import argparse
import json
import logging
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    _PEFT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - runtime dependency guard
    PeftModel = None
    _PEFT_IMPORT_ERROR = exc


LOG = logging.getLogger("merge_lora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge trained LoRA adapters into a base model and export a standalone HF model."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "auto"], default="float16")
    parser.add_argument("--max-shard-size", default="2GB")
    parser.add_argument("--report-file", default="outputs/lora_merge_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    start = time.time()

    if _PEFT_IMPORT_ERROR is not None:
        LOG.error("Failed to import peft: %s", _PEFT_IMPORT_ERROR)
        raise SystemExit(
            "peft is required for LoRA merge. Install with: uv pip install 'peft>=0.17,<0.18'"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        dtype = None
        if args.dtype == "float16":
            dtype = torch.float16
        elif args.dtype == "bfloat16":
            dtype = torch.bfloat16

        LOG.info("Loading base model: %s", args.base_model)
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            local_files_only=True,
            device_map="cpu",
            torch_dtype=dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            local_files_only=True,
        )

        LOG.info("Loading LoRA adapters: %s", args.lora_dir)
        peft_model = PeftModel.from_pretrained(base, args.lora_dir, is_trainable=False)
        merged = peft_model.merge_and_unload()

        if dtype is not None:
            merged = merged.to(dtype=dtype)
            merged.config.torch_dtype = str(args.dtype)

        LOG.info("Saving merged model: %s", output_dir)
        merged.save_pretrained(
            str(output_dir),
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        tokenizer.save_pretrained(str(output_dir))

        report = {
            "method": "lora_merge",
            "base_model": args.base_model,
            "lora_dir": args.lora_dir,
            "output_dir": str(output_dir),
            "dtype": args.dtype,
            "max_shard_size": args.max_shard_size,
            "elapsed_sec": round(time.time() - start, 3),
            "ok": True,
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOG.info("Wrote report: %s", report_path)
    except Exception as exc:
        report = {
            "method": "lora_merge",
            "base_model": args.base_model,
            "lora_dir": args.lora_dir,
            "output_dir": str(output_dir),
            "ok": False,
            "error": str(exc),
            "elapsed_sec": round(time.time() - start, 3),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOG.exception("LoRA merge failed")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
