import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model
    _PEFT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - runtime dependency guard
    LoraConfig = None
    get_peft_model = None
    _PEFT_IMPORT_ERROR = exc


LOG = logging.getLogger("lora_train")


def default_prompt_pool() -> list[str]:
    return [
        "Explain model compression in one paragraph.",
        "List 3 practical tips to reduce LLM latency.",
        "What is knowledge distillation?",
        "How is structured pruning different from unstructured pruning?",
        "quantization 과 pruning 의 차이를 한국어로 짧게 설명해줘.",
        "모델 경량화 검증에서 꼭 봐야 할 지표를 3개 말해줘.",
        "Explica brevemente la cuantizacion de modelos.",
        "Dame dos ideas para desplegar LLMs en dispositivos edge.",
        "Write one short sentence about transformers.",
        "How do you balance quality and speed in LLM compression?",
    ]


def expand_prompts(base_prompts: list[str], num_samples: int) -> list[str]:
    prompts = []
    i = 0
    while len(prompts) < num_samples:
        p = base_prompts[i % len(base_prompts)]
        prompts.append(f"{p} (sample {i + 1})")
        i += 1
    return prompts


@dataclass
class TrainExample:
    prompt: str
    response: str
    input_ids: list[int]
    labels: list[int]


def read_dataset(dataset_file: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with dataset_file.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("prompt", "").strip()
            response = row.get("response", "").strip()
            if not prompt or not response:
                LOG.warning("Skipping invalid dataset row at line %d", lineno)
                continue
            pairs.append((prompt, response))
    return pairs


def generate_teacher_dataset(
    teacher_model_dir: str,
    dataset_file: Path,
    num_samples: int,
    max_new_tokens: int,
) -> list[tuple[str, str]]:
    tokenizer = AutoTokenizer.from_pretrained(
        teacher_model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOG.info("Loading teacher model: %s", teacher_model_dir)
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_dir,
        trust_remote_code=True,
        local_files_only=True,
        device_map="cuda",
        torch_dtype=torch.float16,
    )
    teacher.eval()

    prompts = expand_prompts(default_prompt_pool(), num_samples)
    pairs: list[tuple[str, str]] = []
    dataset_file.parent.mkdir(parents=True, exist_ok=True)

    with dataset_file.open("w", encoding="utf-8") as f:
        with torch.inference_mode():
            for idx, prompt in enumerate(prompts, start=1):
                messages = [{"role": "user", "content": prompt}]
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(teacher.device)
                prompt_len = input_ids.shape[1]
                out = teacher.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                gen_ids = out[0][prompt_len:].detach().cpu().tolist()
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                if not text:
                    text = "I am unable to provide a response."
                pairs.append((prompt, text))
                f.write(
                    json.dumps(
                        {"prompt": prompt, "response": text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if idx % 20 == 0 or idx == len(prompts):
                    LOG.info("Teacher dataset generation: %d/%d", idx, len(prompts))

    del teacher
    torch.cuda.empty_cache()
    LOG.info("Wrote generated distillation dataset: %s", dataset_file)
    return pairs


def build_examples(
    pairs: list[tuple[str, str]],
    tokenizer,
    max_length: int,
) -> list[TrainExample]:
    examples: list[TrainExample] = []
    eos_id = tokenizer.eos_token_id
    for prompt, response in pairs:
        messages = [{"role": "user", "content": prompt}]
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        resp_ids = tokenizer(
            response,
            add_special_tokens=False,
        )["input_ids"]
        full_ids = prompt_ids + resp_ids + ([eos_id] if eos_id is not None else [])
        labels = ([-100] * len(prompt_ids)) + resp_ids + ([eos_id] if eos_id is not None else [])

        full_ids = full_ids[:max_length]
        labels = labels[:max_length]
        if all(v == -100 for v in labels):
            continue
        examples.append(
            TrainExample(
                prompt=prompt,
                response=response,
                input_ids=full_ids,
                labels=labels,
            )
        )
    return examples


def collate_batch(
    batch: list[TrainExample],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(x.input_ids) for x in batch)
    input_tensors = []
    label_tensors = []
    attn_tensors = []
    for item in batch:
        pad = max_len - len(item.input_ids)
        ids = item.input_ids + ([pad_token_id] * pad)
        labels = item.labels + ([-100] * pad)
        attn = [1] * len(item.input_ids) + [0] * pad
        input_tensors.append(ids)
        label_tensors.append(labels)
        attn_tensors.append(attn)

    return (
        torch.tensor(input_tensors, dtype=torch.long, device=device),
        torch.tensor(label_tensors, dtype=torch.long, device=device),
        torch.tensor(attn_tensors, dtype=torch.long, device=device),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LoRA-based distillation recovery for compressed EXAONE models. "
            "Trains LoRA adapters on teacher-generated data, then save adapter checkpoints."
        )
    )
    parser.add_argument("--teacher-model", default="open/base_model")
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-file", default="outputs/distill_lora_dataset.jsonl")
    parser.add_argument("--report-file", default="outputs/lora_train_report.json")
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names for LoRA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    start = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if _PEFT_IMPORT_ERROR is not None:
        LOG.error("Failed to import peft: %s", _PEFT_IMPORT_ERROR)
        raise SystemExit(
            "peft is required for LoRA training. Install with: uv pip install 'peft>=0.17,<0.18'"
        )

    dataset_path = Path(args.dataset_file)
    report_path = Path(args.report_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if dataset_path.is_file():
            pairs = read_dataset(dataset_path)
            LOG.info("Loaded distill dataset: %s (%d rows)", dataset_path, len(pairs))
        else:
            pairs = generate_teacher_dataset(
                teacher_model_dir=args.teacher_model,
                dataset_file=dataset_path,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
            )

        if not pairs:
            raise RuntimeError("No valid training pairs available.")

        tokenizer = AutoTokenizer.from_pretrained(
            args.student_model,
            trust_remote_code=True,
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        examples = build_examples(
            pairs=pairs,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
        if not examples:
            raise RuntimeError("No trainable examples after tokenization.")

        LOG.info("Loading student model: %s", args.student_model)
        model = AutoModelForCausalLM.from_pretrained(
            args.student_model,
            trust_remote_code=True,
            local_files_only=True,
            device_map="cuda",
            torch_dtype=torch.float16,
        )
        model.config.use_cache = False

        target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
        peft_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            bias="none",
        )
        model = get_peft_model(model, peft_cfg)
        model.train()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        LOG.info(
            "Trainable params: %d / %d (%.4f%%)",
            trainable,
            total,
            100.0 * trainable / max(total, 1),
        )

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
        )

        losses: list[float] = []
        device = next(model.parameters()).device
        steps = 0

        for epoch in range(args.epochs):
            random.shuffle(examples)
            for i in range(0, len(examples), args.batch_size):
                batch = examples[i : i + args.batch_size]
                input_ids, labels, attention_mask = collate_batch(
                    batch=batch,
                    pad_token_id=tokenizer.pad_token_id,
                    device=device,
                )
                try:
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False,
                    )
                    loss = out.loss
                    if not torch.isfinite(loss):
                        LOG.warning("Skipping non-finite loss at step=%d", steps + 1)
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_norm=1.0,
                    )
                    optimizer.step()
                    steps += 1
                    losses.append(float(loss.item()))
                except torch.cuda.OutOfMemoryError:
                    LOG.warning("CUDA OOM at step=%d. Skipping batch.", steps + 1)
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue

                if steps % 20 == 0:
                    LOG.info("epoch=%d step=%d loss=%.4f", epoch + 1, steps, losses[-1])

        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        report = {
            "method": "lora_distillation_recovery",
            "teacher_model": args.teacher_model,
            "student_model": args.student_model,
            "output_adapter_dir": str(out_dir),
            "dataset_file": str(dataset_path),
            "dataset_size": len(pairs),
            "train_examples": len(examples),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "lora": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": target_modules,
            },
            "trainable_params": trainable,
            "total_params": total,
            "trainable_ratio": trainable / max(total, 1),
            "steps": steps,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "loss_avg": (sum(losses) / len(losses)) if losses else None,
            "elapsed_sec": round(time.time() - start, 3),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOG.info("Wrote report: %s", report_path)
        LOG.info("LoRA training completed in %.2fs", report["elapsed_sec"])
    except Exception as exc:
        fail_report = {
            "ok": False,
            "error": str(exc),
            "elapsed_sec": round(time.time() - start, 3),
        }
        report_path.write_text(json.dumps(fail_report, indent=2), encoding="utf-8")
        LOG.exception("LoRA training failed")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
