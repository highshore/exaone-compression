import argparse
import gc
import json
import logging
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from vllm import LLM, SamplingParams


LOG = logging.getLogger("verify_vllm")


@dataclass
class AttemptConfig:
    dtype: str
    enforce_eager: bool
    max_model_len: int
    gpu_memory_utilization: float


def classify_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "out of memory" in text or "cuda oom" in text:
        return "cuda_oom"
    if "enginecore" in text and "died unexpectedly" in text:
        return "engine_died"
    if "triton" in text or "inductor" in text or "compile" in text:
        return "compile_or_kernel"
    return "unknown"


def run_once(model_dir: str, prompt: str, max_tokens: int, attempt_cfg: AttemptConfig) -> dict:
    llm: Optional[LLM] = None
    start = time.time()
    try:
        llm = LLM(
            model=model_dir,
            tensor_parallel_size=1,
            gpu_memory_utilization=attempt_cfg.gpu_memory_utilization,
            max_model_len=attempt_cfg.max_model_len,
            dtype=attempt_cfg.dtype,
            enforce_eager=attempt_cfg.enforce_eager,
        )
        # EXAONE is chat-tuned; chat mode avoids immediate-EOS empty responses.
        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
            min_tokens=8,
            presence_penalty=0.0,
        )
        outputs = llm.chat([[{"role": "user", "content": prompt}]], params)
        text = outputs[0].outputs[0].text
        if not text.strip():
            raise RuntimeError("Generated empty output.")
        elapsed = time.time() - start
        return {"ok": True, "text": text, "elapsed_sec": round(elapsed, 3)}
    finally:
        # vLLM worker cleanup is not always immediate; force best-effort teardown.
        del llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def build_attempts(args: argparse.Namespace) -> list[AttemptConfig]:
    attempts = [
        AttemptConfig(
            dtype=args.dtype,
            enforce_eager=args.enforce_eager,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    ]

    fallback = AttemptConfig(
        dtype="float16",
        enforce_eager=True,
        max_model_len=min(args.max_model_len, 4096),
        gpu_memory_utilization=min(args.gpu_memory_utilization, 0.80),
    )
    if asdict(fallback) != asdict(attempts[0]):
        attempts.append(fallback)

    return attempts[: max(1, args.retries)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16"])
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--report-file", default="")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    report: dict = {
        "model_dir": args.model_dir,
        "prompt": args.prompt,
        "attempts": [],
        "ok": False,
    }

    attempts = build_attempts(args)
    for idx, attempt_cfg in enumerate(attempts, start=1):
        LOG.info("Attempt %d/%d with config=%s", idx, len(attempts), asdict(attempt_cfg))
        try:
            result = run_once(
                model_dir=args.model_dir,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                attempt_cfg=attempt_cfg,
            )
            report["attempts"].append(
                {"config": asdict(attempt_cfg), "result": result, "ok": True}
            )
            report["ok"] = True
            print(result["text"])
            print(
                "OK: vLLM load + generate succeeded "
                f"(elapsed={result['elapsed_sec']}s, attempt={idx})"
            )
            break
        except Exception as exc:
            triage = classify_error(exc)
            err = {
                "ok": False,
                "config": asdict(attempt_cfg),
                "error_type": triage,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            report["attempts"].append(err)
            LOG.exception("Attempt %d failed (type=%s)", idx, triage)

    if args.report_file:
        path = Path(args.report_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOG.info("Wrote report: %s", path)

    if not report["ok"]:
        raise SystemExit("ERROR: vLLM verification failed after retries")


if __name__ == "__main__":
    main()
