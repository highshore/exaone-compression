import argparse
import gc
import json
import logging
import time
from difflib import SequenceMatcher
from pathlib import Path

from vllm import LLM, SamplingParams


LOG = logging.getLogger("eval_compare")


def default_prompts() -> list[str]:
    return [
        "Explain model compression in one paragraph.",
        "List 3 practical tips to reduce LLM latency.",
        "What is knowledge distillation?",
        "How is structured pruning different from unstructured pruning?",
        "quantization 과 pruning 의 차이를 한국어로 짧게 설명해줘.",
        "모델 경량화 검증에서 꼭 봐야 할 지표를 3개 말해줘.",
        "Explica brevemente la cuantización de modelos.",
        "Dame dos ideas para desplegar LLMs en dispositivos edge.",
        "hello",
        "Write one short sentence about transformers.",
    ]


def generate_outputs(
    model_dir: str,
    prompts: list[str],
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
        # EXAONE is chat-tuned; raw generate() often immediately emits EOS on plain prompts.
        # Use chat format so all variants can produce non-empty, comparable outputs.
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        outs = llm.chat(messages, sampling_params)
        texts = [o.outputs[0].text for o in outs]
        non_empty = sum(1 for t in texts if t.strip())
        return {
            "outputs": texts,
            "non_empty_count": non_empty,
            "non_empty_rate": round(non_empty / len(texts), 4),
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


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare candidate models against a baseline model using vLLM generation."
    )
    parser.add_argument("--baseline-model", default="open/base_model")
    parser.add_argument(
        "--candidate-models",
        nargs="+",
        required=True,
        help="One or more candidate model directories.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--min-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--report-file", default="outputs/eval_compare.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    prompts = default_prompts()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        presence_penalty=args.presence_penalty,
    )

    LOG.info("Evaluating with %d prompts", len(prompts))
    baseline = generate_outputs(
        model_dir=args.baseline_model,
        prompts=prompts,
        sampling_params=sampling_params,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    candidates = []
    for model_dir in args.candidate_models:
        LOG.info("Evaluating candidate: %s", model_dir)
        cand = generate_outputs(
            model_dir=model_dir,
            prompts=prompts,
            sampling_params=sampling_params,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        sims = [
            similarity(b, c) for b, c in zip(baseline["outputs"], cand["outputs"], strict=True)
        ]
        exact = [
            1 if b == c else 0
            for b, c in zip(baseline["outputs"], cand["outputs"], strict=True)
        ]
        speedup = baseline["elapsed_sec"] / cand["elapsed_sec"]
        candidates.append(
            {
                "model_dir": model_dir,
                "elapsed_sec": cand["elapsed_sec"],
                "speedup_vs_baseline": round(speedup, 4),
                "avg_similarity_to_baseline": round(sum(sims) / len(sims), 4),
                "exact_match_rate": round(sum(exact) / len(exact), 4),
                "non_empty_count": cand["non_empty_count"],
                "non_empty_rate": cand["non_empty_rate"],
                "outputs": cand["outputs"],
                "per_prompt_similarity": [round(s, 4) for s in sims],
            }
        )

    report = {
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "presence_penalty": args.presence_penalty,
            "max_tokens": args.max_tokens,
            "min_tokens": args.min_tokens,
        },
        "prompts": prompts,
        "baseline": {
            "model_dir": args.baseline_model,
            "elapsed_sec": baseline["elapsed_sec"],
            "non_empty_count": baseline["non_empty_count"],
            "non_empty_rate": baseline["non_empty_rate"],
            "outputs": baseline["outputs"],
        },
        "candidates": candidates,
    }
    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Wrote report: %s", out)

    print("Baseline elapsed_sec:", baseline["elapsed_sec"])
    for c in candidates:
        print(
            f"- {c['model_dir']}: elapsed={c['elapsed_sec']}s "
            f"speedup={c['speedup_vs_baseline']} "
            f"avg_similarity={c['avg_similarity_to_baseline']} "
            f"exact_match_rate={c['exact_match_rate']} "
            f"non_empty_rate={c['non_empty_rate']}"
        )


if __name__ == "__main__":
    main()
