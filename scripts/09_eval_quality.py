import argparse
import gc
import json
import logging
import re
import time
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams


LOG = logging.getLogger("eval_quality")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark-style quality evaluation for baseline/candidate models. "
            "Computes accuracy-like metrics and PerfNorm against baseline."
        )
    )
    parser.add_argument("--baseline-model", default="open/base_model")
    parser.add_argument("--candidate-models", nargs="+", required=True)
    parser.add_argument(
        "--datasets",
        default="mmlu,boolq",
        help="Comma-separated dataset list. Supported: mmlu,boolq",
    )
    parser.add_argument("--limit-per-dataset", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--report-file", default="outputs/eval_quality.json")
    return parser.parse_args()


def load_mmlu(limit: int) -> list[dict]:
    errors = []
    for repo, subset in [("cais/mmlu", "all"), ("hendrycks_test", "all")]:
        try:
            ds = load_dataset(repo, subset, split="test")
            rows = ds.select(range(min(limit, len(ds))))
            out = []
            for r in rows:
                question = r["question"]
                choices = r["choices"]
                answer = int(r["answer"])
                labels = ["A", "B", "C", "D"]
                options = "\n".join(f"{labels[i]}. {choices[i]}" for i in range(min(4, len(choices))))
                prompt = (
                    "Answer the following multiple-choice question.\n"
                    "Return only one letter: A, B, C, or D.\n\n"
                    f"Question: {question}\n"
                    f"{options}\n"
                    "Answer:"
                )
                out.append(
                    {
                        "dataset": "mmlu",
                        "prompt": prompt,
                        "label": labels[answer] if 0 <= answer < 4 else "A",
                    }
                )
            return out
        except Exception as exc:  # pragma: no cover - network/data availability dependent
            errors.append(f"{repo}:{exc}")
    raise RuntimeError(f"Failed to load MMLU. Errors: {' | '.join(errors)}")


def load_boolq(limit: int) -> list[dict]:
    ds = load_dataset("google/boolq", split="validation")
    rows = ds.select(range(min(limit, len(ds))))
    out = []
    for r in rows:
        prompt = (
            "Read the passage and answer the question.\n"
            "Return only one word: yes or no.\n\n"
            f"Passage: {r['passage']}\n"
            f"Question: {r['question']}\n"
            "Answer:"
        )
        out.append(
            {
                "dataset": "boolq",
                "prompt": prompt,
                "label": "yes" if bool(r["answer"]) else "no",
            }
        )
    return out


def parse_mmlu_pred(text: str) -> str:
    m = re.search(r"\b([ABCD])\b", text.upper())
    if m:
        return m.group(1)
    text = text.strip().upper()
    return text[:1] if text else ""


def parse_boolq_pred(text: str) -> str:
    t = text.strip().lower()
    if "yes" in t:
        return "yes"
    if "no" in t:
        return "no"
    if t.startswith("y"):
        return "yes"
    if t.startswith("n"):
        return "no"
    return ""


def evaluate_model(
    model_dir: str,
    samples: list[dict],
    sampling_params: SamplingParams,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> dict:
    llm = None
    start = time.time()
    try:
        llm = LLM(
            model=model_dir,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="float16",
            enforce_eager=True,
        )
        messages = [[{"role": "user", "content": s["prompt"]}] for s in samples]
        outs = llm.chat(messages, sampling_params)
        texts = [o.outputs[0].text for o in outs]

        per_dataset = {}
        for s, text in zip(samples, texts, strict=True):
            name = s["dataset"]
            if name not in per_dataset:
                per_dataset[name] = {"correct": 0, "total": 0}
            if name == "mmlu":
                pred = parse_mmlu_pred(text)
            elif name == "boolq":
                pred = parse_boolq_pred(text)
            else:
                pred = ""
            per_dataset[name]["correct"] += int(pred == s["label"])
            per_dataset[name]["total"] += 1

        total_correct = sum(v["correct"] for v in per_dataset.values())
        total_count = sum(v["total"] for v in per_dataset.values())
        by_ds = {}
        for k, v in per_dataset.items():
            by_ds[k] = {
                "accuracy": round(v["correct"] / max(v["total"], 1), 4),
                "correct": v["correct"],
                "total": v["total"],
            }

        return {
            "model_dir": model_dir,
            "overall_accuracy": round(total_correct / max(total_count, 1), 4),
            "total_correct": total_correct,
            "total_count": total_count,
            "by_dataset": by_ds,
            "elapsed_sec": round(time.time() - start, 3),
        }
    finally:
        del llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def build_samples(dataset_names: list[str], limit: int) -> list[dict]:
    all_samples: list[dict] = []
    for name in dataset_names:
        if name == "mmlu":
            all_samples.extend(load_mmlu(limit))
        elif name == "boolq":
            all_samples.extend(load_boolq(limit))
        else:
            raise ValueError(f"Unsupported dataset: {name}")
    return all_samples


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    dataset_names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    LOG.info("Loading evaluation datasets: %s", dataset_names)
    samples = build_samples(dataset_names, args.limit_per_dataset)
    LOG.info("Loaded %d samples total", len(samples))

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    baseline = evaluate_model(
        model_dir=args.baseline_model,
        samples=samples,
        sampling_params=sampling_params,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    LOG.info("Baseline accuracy=%.4f", baseline["overall_accuracy"])

    candidates = []
    for model_dir in args.candidate_models:
        LOG.info("Evaluating candidate: %s", model_dir)
        cand = evaluate_model(
            model_dir=model_dir,
            samples=samples,
            sampling_params=sampling_params,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        perf_norm = cand["overall_accuracy"] / max(baseline["overall_accuracy"], 1e-9)
        cand["perf_norm_vs_baseline"] = round(perf_norm, 4)
        candidates.append(cand)

    report = {
        "datasets": dataset_names,
        "limit_per_dataset": args.limit_per_dataset,
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
        "baseline": baseline,
        "candidates": candidates,
    }
    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Wrote report: %s", report_path)

    print(f"Baseline accuracy: {baseline['overall_accuracy']:.4f}")
    for c in candidates:
        print(
            f"- {c['model_dir']}: accuracy={c['overall_accuracy']:.4f} "
            f"PerfNorm={c['perf_norm_vs_baseline']:.4f}"
        )


if __name__ == "__main__":
    main()
