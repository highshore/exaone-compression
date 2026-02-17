import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LOG = logging.getLogger("distill_student")


def build_prompt_pool() -> list[str]:
    return [
        "Explain model compression in 3 bullet points.",
        "What is the difference between pruning and quantization?",
        "Give one practical tip to reduce LLM latency.",
        "Summarize why edge deployment matters for LLMs.",
        "Write a short answer about knowledge distillation.",
        "How does layer pruning affect inference speed?",
        "Describe PTQ vs QAT in simple language.",
        "What are trade-offs of 4-bit quantization?",
        "Give a concise explanation of LoRA.",
        "What is the role of attention heads in a transformer?",
        "vLLM 환경에서 모델 경량화를 검증할 때 핵심 체크포인트 3가지를 말해줘.",
        "한국어로 pruning과 quantization 차이를 간단히 설명해줘.",
        "온디바이스 LLM 배포 시 메모리 최적화 전략을 요약해줘.",
        "LLM 압축에서 성능 저하를 줄이는 방법을 알려줘.",
        "추론 속도와 정확도의 균형을 맞추는 방법을 설명해줘.",
        "Explica la cuantización de modelos en términos simples.",
        "¿Cuál es la ventaja de podar capas en un LLM?",
        "Resume técnicas para acelerar inferencia en GPU.",
        "Compara distillation y pruning brevemente.",
        "Da 3 recomendaciones para validar un modelo comprimido.",
    ]


def expand_prompts(base_prompts: list[str], num_samples: int) -> list[str]:
    out = []
    i = 0
    while len(out) < num_samples:
        p = base_prompts[i % len(base_prompts)]
        out.append(f"{p} (sample {i + 1})")
        i += 1
    return out


@dataclass
class DistillSample:
    prompt: str
    response: str
    input_ids: list[int]
    labels: list[int]


def generate_teacher_data(
    teacher_model_dir: str,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
) -> list[DistillSample]:
    LOG.info("Loading teacher model: %s", teacher_model_dir)
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_dir,
        trust_remote_code=True,
        local_files_only=True,
        device_map="cuda",
        torch_dtype=torch.float16,
    )
    teacher.eval()

    samples: list[DistillSample] = []
    with torch.inference_mode():
        for idx, prompt in enumerate(prompts, start=1):
            messages = [{"role": "user", "content": prompt}]
            enc = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(teacher.device)
            prompt_len = enc.shape[1]
            out = teacher.generate(
                enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            full_ids = out[0].detach().cpu().tolist()
            gen_ids = out[0][prompt_len:].detach().cpu().tolist()
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)

            labels = full_ids.copy()
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100

            samples.append(
                DistillSample(
                    prompt=prompt,
                    response=response,
                    input_ids=full_ids,
                    labels=labels,
                )
            )
            if idx % 10 == 0 or idx == len(prompts):
                LOG.info("Teacher generation progress: %d/%d", idx, len(prompts))

    del teacher
    torch.cuda.empty_cache()
    return samples


def collate_batch(
    batch: list[DistillSample],
    pad_token_id: int,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_tensors = []
    label_tensors = []
    attn_tensors = []
    for s in batch:
        ids = s.input_ids[:max_length]
        labels = s.labels[:max_length]
        pad_len = max_length - len(ids)
        if pad_len > 0:
            ids = ids + [pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
        attn = [1 if tok != pad_token_id else 0 for tok in ids]
        input_tensors.append(ids)
        label_tensors.append(labels)
        attn_tensors.append(attn)

    return (
        torch.tensor(input_tensors, dtype=torch.long, device=device),
        torch.tensor(label_tensors, dtype=torch.long, device=device),
        torch.tensor(attn_tensors, dtype=torch.long, device=device),
    )


def train_student_lm_head(
    student_model_dir: str,
    output_dir: str,
    tokenizer,
    samples: list[DistillSample],
    learning_rate: float,
    epochs: int,
    batch_size: int,
    max_length: int,
) -> dict:
    LOG.info("Loading student model: %s", student_model_dir)
    student = AutoModelForCausalLM.from_pretrained(
        student_model_dir,
        trust_remote_code=True,
        local_files_only=True,
        device_map="cuda",
        torch_dtype=torch.float16,
    )
    student.train()

    # Untie lm_head for stable FP32 optimization on the output projection.
    in_emb = student.get_input_embeddings().weight
    out_emb = student.get_output_embeddings().weight
    if in_emb.data_ptr() == out_emb.data_ptr():
        LOG.info("Detected tied embeddings. Untying lm_head for calibration.")
    student.config.tie_word_embeddings = False
    student.lm_head.weight = torch.nn.Parameter(student.lm_head.weight.detach().float().clone())
    if getattr(student.lm_head, "bias", None) is not None:
        student.lm_head.bias = torch.nn.Parameter(student.lm_head.bias.detach().float().clone())

    # Freeze all params except lm_head for stable/fast recovery training.
    for p in student.parameters():
        p.requires_grad = False
    for p in student.lm_head.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    LOG.info(
        "Trainable params: %d / %d (%.4f%%)",
        trainable,
        total,
        100.0 * trainable / max(total, 1),
    )

    optimizer = torch.optim.AdamW(student.lm_head.parameters(), lr=learning_rate)
    losses: list[float] = []
    device = next(student.parameters()).device
    steps = 0

    for epoch in range(epochs):
        random.shuffle(samples)
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            input_ids, labels, attention_mask = collate_batch(
                batch=batch,
                pad_token_id=tokenizer.pad_token_id,
                max_length=max_length,
                device=device,
            )

            with torch.no_grad():
                hidden = student.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )[0]
            logits = student.lm_head(hidden.float())
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.float().view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            if not torch.isfinite(loss):
                LOG.warning("Skipping non-finite loss at step=%d", steps + 1)
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.lm_head.parameters(), max_norm=1.0)
            optimizer.step()

            steps += 1
            losses.append(float(loss.item()))
            if steps % 10 == 0:
                LOG.info("epoch=%d step=%d loss=%.4f", epoch + 1, steps, losses[-1])

    # Restore compact dtype before export.
    student.lm_head.weight.data = student.lm_head.weight.data.half()
    if getattr(student.lm_head, "bias", None) is not None:
        student.lm_head.bias.data = student.lm_head.bias.data.half()

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(str(out), safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(str(out))

    del student
    torch.cuda.empty_cache()

    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": trainable / max(total, 1),
        "steps": steps,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_avg": sum(losses) / max(len(losses), 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Knowledge-distillation-style recovery for a pruned EXAONE model. "
            "Builds teacher outputs and calibrates the student lm_head."
        )
    )
    parser.add_argument("--teacher-model", default="open/base_model")
    parser.add_argument("--student-model", default="models/compressed-l29")
    parser.add_argument("--output-dir", default="models/compressed-l29-distilled")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument(
        "--report-file",
        default="outputs/distill_report.json",
    )
    parser.add_argument(
        "--dataset-file",
        default="outputs/distill_dataset.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = expand_prompts(build_prompt_pool(), args.num_samples)
    samples = generate_teacher_data(
        teacher_model_dir=args.teacher_model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
    )

    ds_path = Path(args.dataset_file)
    ds_path.parent.mkdir(parents=True, exist_ok=True)
    with ds_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(
                json.dumps(
                    {"prompt": s.prompt, "response": s.response},
                    ensure_ascii=False,
                )
                + "\n"
            )
    LOG.info("Wrote distill dataset: %s", ds_path)

    train_stats = train_student_lm_head(
        student_model_dir=args.student_model,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        samples=samples,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    report = {
        "method": "structured_pruning_plus_distillation_calibration",
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "output_model": args.output_dir,
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_stats": train_stats,
        "elapsed_sec": round(time.time() - start, 3),
    }

    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("Wrote report: %s", report_path)
    LOG.info("Distillation complete in %.2fs", report["elapsed_sec"])


if __name__ == "__main__":
    main()
