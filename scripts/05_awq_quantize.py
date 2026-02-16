import argparse
import json
import logging
import time
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LOG = logging.getLogger("compress_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compress EXAONE model by dropping decoder layers and exporting "
            "a Hugging Face-compatible checkpoint."
        )
    )
    parser.add_argument("--model-dir", required=True, help="Input HF model directory")
    parser.add_argument("--output-dir", required=True, help="Output HF model directory")
    parser.add_argument(
        "--target-layers",
        type=int,
        default=29,
        help="Number of decoder layers to keep (original EXAONE-4.0-1.2B has 30)",
    )
    parser.add_argument(
        "--selection",
        choices=["uniform", "first", "last"],
        default="uniform",
        help="Layer selection strategy",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16"],
        default="float16",
        help="Export dtype for model weights",
    )
    parser.add_argument(
        "--max-shard-size",
        default="2GB",
        help="Shard size for save_pretrained (e.g., 2GB)",
    )
    return parser.parse_args()


def choose_layers(total: int, target: int, strategy: str) -> list[int]:
    if target <= 0 or target > total:
        raise ValueError(f"target-layers must be in [1, {total}], got {target}")
    if target == total:
        return list(range(total))

    if strategy == "first":
        return list(range(target))
    if strategy == "last":
        return list(range(total - target, total))

    # Uniformly sample layers from shallow->deep while preserving order.
    picks = []
    for i in range(target):
        pos = round(i * (total - 1) / (target - 1))
        picks.append(pos)
    dedup = sorted(set(picks))
    if len(dedup) != target:
        # Fallback for pathological rounding collisions.
        dedup = list(range(total))[:target]
    return dedup


def cast_model_dtype(model: torch.nn.Module, dtype_name: str) -> torch.nn.Module:
    if dtype_name == "auto":
        return model
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    return model.to(dtype=dtype)


def file_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def summarize_keep_indices(indices: Iterable[int]) -> str:
    indices = list(indices)
    if not indices:
        return "[]"
    if len(indices) <= 12:
        return str(indices)
    return f"{indices[:6]} ... {indices[-6:]}"


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    start = time.time()
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.is_dir():
        raise SystemExit(f"model-dir not found: {model_dir}")

    LOG.info("Loading model from %s", model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise SystemExit("Unexpected model structure: expected model.model.layers")

    layers = model.model.layers
    total_layers = len(layers)
    keep = choose_layers(total_layers, args.target_layers, args.selection)
    LOG.info(
        "Layer compression: total=%d target=%d strategy=%s keep=%s",
        total_layers,
        args.target_layers,
        args.selection,
        summarize_keep_indices(keep),
    )

    model.model.layers = torch.nn.ModuleList([layers[i] for i in keep])
    model.config.num_hidden_layers = len(keep)
    if hasattr(model.config, "layer_types") and isinstance(model.config.layer_types, list):
        model.config.layer_types = [model.config.layer_types[i] for i in keep]
    if args.dtype != "auto":
        model.config.torch_dtype = args.dtype
    for new_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_idx

    model = cast_model_dtype(model, args.dtype)

    LOG.info("Saving compressed model to %s", output_dir)
    model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(str(output_dir))

    report = {
        "method": "layer_drop",
        "source_model_dir": str(model_dir),
        "output_model_dir": str(output_dir),
        "selection": args.selection,
        "dtype": args.dtype,
        "total_layers": total_layers,
        "kept_layers": len(keep),
        "kept_indices": keep,
        "output_size_bytes": file_size_bytes(output_dir),
        "elapsed_sec": round(time.time() - start, 3),
    }
    report_path = output_dir / "compression_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("Wrote report: %s", report_path)
    LOG.info("Compression complete in %.2fs", report["elapsed_sec"])


if __name__ == "__main__":
    main()
