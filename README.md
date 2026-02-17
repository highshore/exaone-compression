# EXAONE Compression Toolkit

## 한국어 (KOR)

### 1) 목적
이 저장소는 `project_rules.txt`의 고정 평가 환경(`torch==2.9.0+cu128`, `transformers==4.57.3`, `vllm==0.14.1`)을 기준으로,
EXAONE-4.0-1.2B를 경량화하고 `submit.zip/model/*` 형태로 제출 가능하게 만드는 파이프라인입니다.

이번 문서는 "실제로 끝까지 돌아간 방법"만 정리했습니다.

### 2) 여기까지 온 실제 작업 경로 (성공 경로만)
1. Baseline 환경/모델 고정
- `models/base` 준비
- vLLM eager 모드 기준 로드/생성 확인

2. Structured pruning (layer drop)
- `scripts/05_awq_quantize.py`
- 30층 -> 29층 (`models/compressed-l29`)

3. Distillation-style 보정
- `scripts/07_distill_student.py`
- pruned student에 teacher 출력을 이용해 `lm_head` 보정 (`models/compressed-l29-distilled`)
- 추가로 distilled-only 케이스도 생성 (`models/base-distilled`)

4. Quantization (AWQ, llmcompressor 경로)
- AutoAWQ 대신 `llmcompressor` + `compressed-tensors(W4A16)` 경로 사용
- 생성 모델:
  - `models/base-llmc-awq`
  - `models/compressed-l29-llmc-awq`
  - `models/base-distilled-llmc-awq`
  - `models/compressed-l29-distilled-llmc-awq`

5. 8개 케이스 통합 비교
- `scripts/08_eval_compare.py`
- 리포트: `outputs/eval_compare_all_8cases.json` (RunPod 실행 산출물)

### 3) 8개 방법 조합 비교 결과
측정 기준:
- `elapsed_sec`: 모델 로드 + 10개 프롬프트 생성 총 시간
- `speedup_vs_baseline`: baseline 대비 속도 배율
- `avg_similarity`, `exact_match_rate`: baseline 출력 문자열 기준 단순 유사도 프록시
- `size`: 모델 디렉토리 실제 파일 총합

기준 모델:
- Baseline elapsed: `31.114s`
- Baseline size: `2.394 GiB`

| Case | Method | Model Dir | Elapsed (s) | Speedup | Avg Sim | Exact | Size (GiB) | Size Delta vs Base |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | original | `models/base` | 31.114 | 1.0000 | 1.0 | 1.0 | 2.394 | 0.00% |
| 2 | distilled only | `models/base-distilled` | 16.894 | 1.8417 | 1.0 | 1.0 | 2.784 | -16.31% (larger) |
| 3 | pruned only | `models/compressed-l29` | 17.287 | 1.7998 | 1.0 | 1.0 | 2.327 | +2.79% smaller |
| 4 | quantized only | `models/base-llmc-awq` | 19.376 | 1.6058 | 0.9 | 0.9 | 1.306 | +45.46% smaller |
| 5 | distilled + pruned | `models/compressed-l29-distilled` | 12.250 | 2.5399 | 1.0 | 1.0 | 2.718 | -13.53% (larger) |
| 6 | pruned + quantized | `models/compressed-l29-llmc-awq` | 11.405 | 2.7281 | 0.7 | 0.7 | 1.288 | +46.18% smaller |
| 7 | distilled + quantized | `models/base-distilled-llmc-awq` | 11.148 | 2.7910 | 0.9 | 0.9 | 1.306 | +45.46% smaller |
| 8 | all (distilled+pruned+quantized) | `models/compressed-l29-distilled-llmc-awq` | 11.602 | 2.6818 | 0.7 | 0.7 | 1.288 | +46.18% smaller |

### 4) 결과 해석 (추천 선택지)
1. 품질 보수적(quality-first) 추천
- `models/compressed-l29` (pruned only)
- 이유: 유사도/일치율 1.0 유지, 크기 감소(+2.79%), 속도 개선(1.80x)

2. 용량 절감(size-first) 추천
- `models/base-llmc-awq` 또는 `models/compressed-l29-llmc-awq`
- 이유: 약 45~46% 용량 절감
- 주의: 출력 품질 프록시 하락(0.9 또는 0.7)

3. 현재 8개 중 비추천
- distilled-only, distilled+pruned: 속도는 빠르지만 파일 크기가 baseline보다 커짐

### 5) vLLM 출력 샘플 (동일 프롬프트)
프롬프트: `hello`
- original / distilled / pruned / pruned+distilled: `hellohellohello...`
- quantized 계열 일부: 빈 문자열(`''`) 발생

중요:
- 본 리포트의 텍스트 유사도는 `scripts/08_eval_compare.py`의 문자열 비교 프록시입니다.
- 실제 리더보드 성능은 주최 측 비공개 벤치마크 점수로 최종 판단됩니다.

### 6) 재현 커맨드 (RunPod)
```bash
# baseline + pruning + distilled(pruned)
make sync
make check
make baseline
make compress
make distill

# distilled-only 생성
uv run python scripts/07_distill_student.py \
  --teacher-model models/base \
  --student-model models/base \
  --output-dir models/base-distilled \
  --num-samples 20 --epochs 1 --batch-size 2 --learning-rate 5e-6 \
  --report-file outputs/distill_report_base.json \
  --dataset-file outputs/distill_dataset_base.jsonl

# quantization variants (llmcompressor W4A16)
# base, compressed-l29, base-distilled, compressed-l29-distilled 각각 대상으로 실행
# 예시: base
python -m llmcompressor.entrypoints.oneshot \
  --model models/base \
  --recipe outputs/recipe_awq_w4a16.yaml \
  --dataset json \
  --dataset_path outputs/ptq_calib \
  --output_dir models/base-llmc-awq

# 8-case 통합 비교
uv run python scripts/08_eval_compare.py \
  --baseline-model models/base \
  --candidate-models \
    models/base-distilled \
    models/compressed-l29 \
    models/base-llmc-awq \
    models/compressed-l29-distilled \
    models/compressed-l29-llmc-awq \
    models/base-distilled-llmc-awq \
    models/compressed-l29-distilled-llmc-awq \
  --report-file outputs/eval_compare_all_8cases.json
```

---

## English (ENG)

### 1) Goal
This repo implements a practical EXAONE-4.0-1.2B compression pipeline that matches `project_rules.txt` constraints and produces HF-compatible `submit.zip/model/*` artifacts.

This README intentionally documents only the path that worked end-to-end.

### 2) What actually worked (chronological)
1. Baseline fixed and validated on vLLM eager mode
2. Structured pruning (30 -> 29 layers) with `scripts/05_awq_quantize.py`
3. Distillation-style recovery with `scripts/07_distill_student.py`
4. AWQ quantization via `llmcompressor + compressed-tensors (W4A16)`
5. Unified 8-case comparison using `scripts/08_eval_compare.py`

### 3) 8-case comparison (performance / speedup / size)
Metrics:
- `elapsed_sec`: total model load + generation time over 10 prompts
- `speedup_vs_baseline`: relative speedup vs baseline
- `avg_similarity`, `exact_match_rate`: string-level output proxy vs baseline
- `size`: on-disk model directory size

Baseline:
- elapsed: `31.114s`
- size: `2.394 GiB`

| Case | Method | Model Dir | Elapsed (s) | Speedup | Avg Sim | Exact | Size (GiB) | Size Delta vs Base |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | original | `models/base` | 31.114 | 1.0000 | 1.0 | 1.0 | 2.394 | 0.00% |
| 2 | distilled only | `models/base-distilled` | 16.894 | 1.8417 | 1.0 | 1.0 | 2.784 | -16.31% (larger) |
| 3 | pruned only | `models/compressed-l29` | 17.287 | 1.7998 | 1.0 | 1.0 | 2.327 | +2.79% smaller |
| 4 | quantized only | `models/base-llmc-awq` | 19.376 | 1.6058 | 0.9 | 0.9 | 1.306 | +45.46% smaller |
| 5 | distilled + pruned | `models/compressed-l29-distilled` | 12.250 | 2.5399 | 1.0 | 1.0 | 2.718 | -13.53% (larger) |
| 6 | pruned + quantized | `models/compressed-l29-llmc-awq` | 11.405 | 2.7281 | 0.7 | 0.7 | 1.288 | +46.18% smaller |
| 7 | distilled + quantized | `models/base-distilled-llmc-awq` | 11.148 | 2.7910 | 0.9 | 0.9 | 1.306 | +45.46% smaller |
| 8 | all (distilled+pruned+quantized) | `models/compressed-l29-distilled-llmc-awq` | 11.602 | 2.6818 | 0.7 | 0.7 | 1.288 | +46.18% smaller |

### 4) Recommendations by objective
1. Quality-first (conservative): `models/compressed-l29`
- Keeps similarity proxy at 1.0, still smaller and faster than baseline.

2. Size-first: `models/base-llmc-awq` or `models/compressed-l29-llmc-awq`
- ~45-46% size reduction.
- Accept quality-proxy regression.

3. Not recommended from this run
- Distilled-only and distilled+pruned variants: speed improved but file size increased vs baseline.

### 5) Prompt/output note
Prompt `hello` produced:
- original/distilled/pruned/pruned+distilled: repeated `hello...`
- several quantized variants: empty string outputs on this prompt

Important:
- Similarity/exact metrics here are lightweight string proxies from `scripts/08_eval_compare.py`.
- Final leaderboard quality must be judged on the organizer benchmark.

### 6) Reproduction commands (RunPod)
```bash
make sync
make check
make baseline
make compress
make distill

uv run python scripts/07_distill_student.py \
  --teacher-model models/base \
  --student-model models/base \
  --output-dir models/base-distilled \
  --num-samples 20 --epochs 1 --batch-size 2 --learning-rate 5e-6 \
  --report-file outputs/distill_report_base.json \
  --dataset-file outputs/distill_dataset_base.jsonl

python -m llmcompressor.entrypoints.oneshot \
  --model models/base \
  --recipe outputs/recipe_awq_w4a16.yaml \
  --dataset json \
  --dataset_path outputs/ptq_calib \
  --output_dir models/base-llmc-awq

uv run python scripts/08_eval_compare.py \
  --baseline-model models/base \
  --candidate-models \
    models/base-distilled \
    models/compressed-l29 \
    models/base-llmc-awq \
    models/compressed-l29-distilled \
    models/compressed-l29-llmc-awq \
    models/base-distilled-llmc-awq \
    models/compressed-l29-distilled-llmc-awq \
  --report-file outputs/eval_compare_all_8cases.json
```
