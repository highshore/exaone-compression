# EXAONE Compression Toolkit

Official baseline source for this repo is `open/base_model` (from the organizer-provided `open.zip`).
All baseline-dependent commands are aligned to that path.

## 한국어 (KOR)

### 1) 문제와 원인
기존 평가 스크립트(`scripts/08_eval_compare.py`)는 `llm.generate()`에 plain prompt 문자열을 직접 넣고 있었습니다.
EXAONE 계열(특히 압축/양자화 변형 포함)은 이 설정에서 EOS로 즉시 종료되는 경우가 많아, 다수 프롬프트가 빈 문자열(`''`)이 되는 문제가 있었습니다.

### 2) 적용한 수정 (Refactor)
빈 출력 문제를 해결하기 위해 아래를 코드에 반영했습니다.

1. `scripts/08_eval_compare.py`
- 생성 경로를 `generate()` -> `chat()`로 변경
- 프롬프트를 chat 메시지 형태(`[{"role":"user","content":...}]`)로 전달
- 기본 샘플링을 `temperature=0.0`, `min_tokens=8`, `max_tokens=64`로 고정
- `non_empty_count`, `non_empty_rate` 지표를 리포트에 추가

2. `scripts/02_verify_vllm.py`
- 검증도 동일하게 `chat()` 경로 사용
- 빈 문자열이면 즉시 실패하도록 강제 (`Generated empty output.`)

### 3) 수정 후 재검증 결과 (8개 모델)
리포트 파일:
- `/workspace/exaone-compression-clean/outputs/eval_compare_all_8cases_chatfix.json`

핵심 결과:
- **baseline + 7 candidates 전부 non_empty_rate = 1.0**
- 즉, 10개 고정 프롬프트 기준 **모든 모델이 빈 출력 없이 생성 성공**

| Model | Elapsed (s) | Speedup | Avg Similarity | Exact Match | Non-Empty Rate |
|---|---:|---:|---:|---:|---:|
| `open/base_model` | 27.146 | 1.0000 | 1.0000 | 1.0 | 1.0 |
| `models/base-distilled` | 15.505 | 1.7508 | 0.9691 | 0.9 | 1.0 |
| `models/compressed-l29` | 15.234 | 1.7819 | 0.4965 | 0.0 | 1.0 |
| `models/base-llmc-awq` | 17.166 | 1.5814 | 0.5420 | 0.0 | 1.0 |
| `models/compressed-l29-distilled` | 11.379 | 2.3856 | 0.4777 | 0.0 | 1.0 |
| `models/compressed-l29-llmc-awq` | 10.564 | 2.5697 | 0.5155 | 0.1 | 1.0 |
| `models/base-distilled-llmc-awq` | 11.221 | 2.4192 | 0.5359 | 0.1 | 1.0 |
| `models/compressed-l29-distilled-llmc-awq` | 10.688 | 2.5399 | 0.5148 | 0.1 | 1.0 |

참고:
- 기존 empty-output 기반 점수 왜곡은 해결되었습니다.
- 지금의 similarity/exact는 "정상 문장끼리 비교"가 되므로 이전보다 의미 있는 프록시입니다.

### 4) 생성 예시 (수정 후)
Prompt: `Explain model compression in one paragraph.`
- `open/base_model`: `Model compression is the technique used to reduce the size ...`
- `models/base-llmc-awq`: `Model compression is the process of reducing the size ...`
- `models/compressed-l29-llmc-awq`: `Model compression is a technique used in deep learning ...`

Prompt: `hello`
- `open/base_model`: `Hello! 😊 How can I assist you today?`
- `models/compressed-l29`: `Hello! How can I assist you today?`
- `models/base-llmc-awq`: `Hello! 😊 How can I help you today?`

### 5) 실행 방법
```bash
# 8개 모델 비교 (chat-fix 기본값 사용)
uv run python scripts/08_eval_compare.py \
  --baseline-model open/base_model \
  --candidate-models \
    models/base-distilled \
    models/compressed-l29 \
    models/base-llmc-awq \
    models/compressed-l29-distilled \
    models/compressed-l29-llmc-awq \
    models/base-distilled-llmc-awq \
    models/compressed-l29-distilled-llmc-awq \
  --report-file outputs/eval_compare_all_8cases_chatfix.json
```

```bash
# 단일 모델 검증 (빈 문자열 출력 시 실패)
uv run python scripts/02_verify_vllm.py \
  --model-dir models/base-llmc-awq \
  --prompt "Explain model compression in one short paragraph." \
  --report-file outputs/verify_base_llmc_awq_chatfix.json
```

---

## English (ENG)

### 1) Problem and Root Cause
The previous eval path used `llm.generate()` with plain prompts.
For EXAONE variants (especially compressed/quantized), that frequently caused immediate EOS and empty outputs.
The organizer-provided baseline path in this repo is `open/base_model`.

### 2) Refactor Applied
1. `scripts/08_eval_compare.py`
- Switched generation from `generate()` to `chat()`
- Uses chat messages (`[{"role":"user","content":...}]`)
- Default sampling: `temperature=0.0`, `min_tokens=8`, `max_tokens=64`
- Added `non_empty_count` and `non_empty_rate` to the report

2. `scripts/02_verify_vllm.py`
- Also switched to `chat()`
- Hard-fails on empty generation (`Generated empty output.`)

### 3) Post-fix Validation
Report:
- `/workspace/exaone-compression-clean/outputs/eval_compare_all_8cases_chatfix.json`

Result:
- **All 8 models achieved `non_empty_rate = 1.0` on all 10 fixed prompts**.
- Empty-string collapse is resolved.

### 4) Notes
- Similarity/exact-match now compare non-empty outputs, so they are more meaningful than before.
- The metric is still a proxy; final leaderboard quality should be judged on task-specific benchmark outputs.

### 5) Stronger Recovery + Better Evaluation (New)
To improve leaderboard potential above low scores, this repo now includes:
- LoRA-based distillation recovery for pruned model (`scripts/03_lora_train.py`)
- LoRA merge to standalone HF checkpoint (`scripts/04_merge_lora.py`)
- Benchmark-style quality evaluation (`scripts/09_eval_quality.py`)
- Token-level speed evaluation aligned with `sec/token` normalization (`scripts/10_eval_token_latency.py`)

Run:
```bash
# 1) Build pruned model first
make compress

# 2) Strong recovery: LoRA distill + merge
make recover-lora

# 3) Evaluate quality proxy (MMLU + BoolQ)
make eval-quality

# 4) Evaluate speed proxy (token latency / SpeedNorm-like)
make eval-token-speed
```

If `peft` is missing:
```bash
uv pip install 'peft>=0.17,<0.18'
```
