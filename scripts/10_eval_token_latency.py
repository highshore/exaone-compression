import argparse
import gc
import json
import logging
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


LOG = logging.getLogger("eval_token_latency")


def default_prompts() -> list[str]:
    return [
        "Explain model compression in one paragraph.",
        "List 3 practical tips to reduce LLM latency.",
        "What is knowledge distillation?",
        "How is structured pruning different from unstructured pruning?",
        "quantization 과 pruning 의 차이를 한국어로 짧게 설명해줘.",
        "모델 경량화 검증에서 꼭 봐야 할 지표를 3개 말해줘.",
        "Explica brevemente la cuantizacion de modelos.",
        "Dame dos ideas para desplegar LLMs en dispositivos edge.",
        "hello",
        "Write one short sentence about transformers.",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure token-level generation latency for baseline/candidates "
            "and compute SpeedNorm aligned with competition formula intent."
        )
    )
    parser.add_argument("--baseline-model", default="open/base_model")
    parser.add_argument("--candidate-models", nargs="+", required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup-prompts", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--min-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--report-file", default="outputs/eval_token_latency.json")
    return parser.parse_args()


def measure_model_latency(
    model_dir: str,
    prompts: list[str],
    rounds: int,
    warmup_prompts: int,
    sampling_params: SamplingParams,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> dict:
    llm = None
    tokenizer = None
    load_start = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        llm = LLM(
            model=model_dir,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="float16",
            enforce_eager=True,
        )
        load_elapsed = time.time() - load_start

        warmup = prompts[: max(0, min(warmup_prompts, len(prompts)))]
        for prompt in warmup:
            llm.chat([[{"role": "user", "content": prompt}]], sampling_params)

        total_elapsed = 0.0
        total_gen_tokens = 0
        request_count = 0
        per_request = []

        for _ in range(max(rounds, 1)):
            for prompt in prompts:
                t0 = time.perf_counter()
                out = llm.chat([[{"role": "user", "content": prompt}]], sampling_params)[0]
                elapsed = time.perf_counter() - t0
                text = out.outputs[0].text
                token_ids = getattr(out.outputs[0], "token_ids", None)
                if token_ids is None:
                    token_ids = tokenizer.encode(text, add_special_tokens=False)
                gen_tokens = max(len(token_ids), 1)

                total_elapsed += elapsed
                total_gen_tokens += gen_tokens
                request_count += 1
                per_request.append(
                    {
                        "prompt": prompt,
                        "elapsed_sec": round(elapsed, 4),
                        "gen_tokens": int(gen_tokens),
                        "sec_per_token": round(elapsed / gen_tokens, 6),
                    }
                )

        sec_per_token = total_elapsed / max(total_gen_tokens, 1)
        toks_per_sec = total_gen_tokens / max(total_elapsed, 1e-9)
        return {
            "model_dir": model_dir,
            "load_elapsed_sec": round(load_elapsed, 3),
            "gen_elapsed_sec": round(total_elapsed, 3),
            "gen_tokens": int(total_gen_tokens),
            "requests": request_count,
            "sec_per_token": round(sec_per_token, 8),
            "tokens_per_sec": round(toks_per_sec, 4),
            "per_request": per_request,
        }
    finally:
        del llm
        del tokenizer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def main() -> None:
    args = parse_args()
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

    LOG.info("Measuring baseline token latency: %s", args.baseline_model)
    baseline = measure_model_latency(
        model_dir=args.baseline_model,
        prompts=prompts,
        rounds=args.rounds,
        warmup_prompts=args.warmup_prompts,
        sampling_params=sampling_params,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    base_spt = baseline["sec_per_token"]
    LOG.info("Baseline sec/token=%.8f", base_spt)

    candidates = []
    for model_dir in args.candidate_models:
        LOG.info("Measuring candidate token latency: %s", model_dir)
        cand = measure_model_latency(
            model_dir=model_dir,
            prompts=prompts,
            rounds=args.rounds,
            warmup_prompts=args.warmup_prompts,
            sampling_params=sampling_params,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        speed_norm = 1.0 - (cand["sec_per_token"] / max(base_spt, 1e-12))
        cand["speed_norm_vs_baseline"] = round(speed_norm, 6)
        candidates.append(cand)

    report = {
        "prompts": prompts,
        "rounds": args.rounds,
        "warmup_prompts": args.warmup_prompts,
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "presence_penalty": args.presence_penalty,
            "max_tokens": args.max_tokens,
            "min_tokens": args.min_tokens,
        },
        "baseline": baseline,
        "candidates": candidates,
    }
    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Wrote report: %s", out)

    print(
        f"Baseline sec_per_token={baseline['sec_per_token']:.8f} "
        f"tokens_per_sec={baseline['tokens_per_sec']:.4f}"
    )
    for c in candidates:
        print(
            f"- {c['model_dir']}: sec_per_token={c['sec_per_token']:.8f} "
            f"tokens_per_sec={c['tokens_per_sec']:.4f} "
            f"SpeedNorm={c['speed_norm_vs_baseline']:.6f}"
        )


if __name__ == "__main__":
    main()
