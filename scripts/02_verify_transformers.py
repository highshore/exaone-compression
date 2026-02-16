import argparse
import importlib.metadata
import json
import logging
import traceback
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LOG = logging.getLogger("verify_transformers")


def parse_semver(version_text: str) -> tuple[int, int, int]:
    core = version_text.split("+", 1)[0]
    parts = core.split(".")
    nums = []
    for part in parts[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        nums.append(int(digits) if digits else 0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)


def autoawq_version() -> Optional[str]:
    try:
        return importlib.metadata.version("autoawq")
    except importlib.metadata.PackageNotFoundError:
        return None


def preflight_awq(model_dir: Path) -> None:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"Missing config.json: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    quant_cfg = config.get("quantization_config")
    if not quant_cfg:
        LOG.info("No quantization_config in model config; AWQ preflight skipped.")
        return

    quant_method = str(quant_cfg.get("quant_method", "")).lower()
    LOG.info("Detected quantization_config.quant_method=%s", quant_method or "<missing>")
    if quant_method != "awq":
        return

    version = autoawq_version()
    if version is None:
        raise SystemExit(
            "AWQ model detected but autoawq is not installed. "
            "Install with: uv pip install 'autoawq>=0.1.8'"
        )

    if parse_semver(version) < parse_semver("0.1.8"):
        raise SystemExit(
            f"AWQ model detected but autoawq=={version} is too old. "
            "Upgrade with: uv pip install -U 'autoawq>=0.1.8'"
        )

    LOG.info("autoawq version check passed: %s", version)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device-map", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    model_dir = Path(args.model_dir).resolve()
    preflight_awq(model_dir)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=True,
            device_map=args.device_map,
            torch_dtype=torch.float16,
        )
        inputs = tokenizer(args.prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(text)
        print("OK: Transformers load + generate succeeded")
    except Exception as exc:
        LOG.error("Transformers verification failed: %s", exc)
        LOG.error(traceback.format_exc())
        raise SystemExit("ERROR: transformers verification failed") from exc


if __name__ == "__main__":
    main()
