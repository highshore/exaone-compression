# EXAONE Compression Toolkit

## 한국어 (KOR)

### 1) 목적
`project_rules.txt` 고정 환경(`torch==2.9.0+cu128`, `transformers==4.57.3`, `vllm==0.14.1`) 기준으로,
EXAONE-4.0-1.2B 경량화 모델을 HF 표준 형식으로 만들고 `submit.zip/model/*`로 제출 가능하게 하는 저장소입니다.

### 2) 실제로 완주한 경로 (성공 경로만)
1. Baseline 준비 및 vLLM eager 로드/생성 검증
2. Structured pruning (30 -> 29 layers): `models/compressed-l29`
3. Distillation-style 보정: `models/compressed-l29-distilled`, `models/base-distilled`
4. AWQ quantization (llmcompressor + compressed-tensors W4A16)
5. 8개 조합 일괄 비교: `scripts/08_eval_compare.py`

### 3) 재실행 성능 비교 (8개 모델)
재실행 리포트:
- `/workspace/exaone-compression-clean/outputs/eval_compare_all_8cases_rerun.json`

기준값:
- Baseline elapsed: `26.833s`
- Baseline size: `2.394 GiB`

| Case | Method | Model Dir | Elapsed (s) | Speedup | Avg Sim | Exact | Size (GiB) | Size Delta vs Base |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | original | `models/base` | 26.833 | 1.0000 | 1.0 | 1.0 | 2.394 | 0.00% |
| 2 | distilled only | `models/base-distilled` | 15.420 | 1.7401 | 1.0 | 1.0 | 2.784 | -16.31% (larger) |
| 3 | pruned only | `models/compressed-l29` | 14.222 | 1.8867 | 1.0 | 1.0 | 2.327 | +2.79% smaller |
| 4 | quantized only | `models/base-llmc-awq` | 15.170 | 1.7688 | 0.9 | 0.9 | 1.306 | +45.46% smaller |
| 5 | distilled + pruned | `models/compressed-l29-distilled` | 10.185 | 2.6346 | 1.0 | 1.0 | 2.718 | -13.53% (larger) |
| 6 | pruned + quantized | `models/compressed-l29-llmc-awq` | 10.735 | 2.4996 | 0.7 | 0.7 | 1.288 | +46.18% smaller |
| 7 | distilled + quantized | `models/base-distilled-llmc-awq` | 10.087 | 2.6602 | 0.9 | 0.9 | 1.306 | +45.46% smaller |
| 8 | all (distilled+pruned+quantized) | `models/compressed-l29-distilled-llmc-awq` | 10.374 | 2.5866 | 0.7 | 0.7 | 1.288 | +46.18% smaller |

### 4) 10개 고정 프롬프트 출력 결과 (모델별)
프롬프트 목록:
1. Explain model compression in one paragraph.
2. List 3 practical tips to reduce LLM latency.
3. What is knowledge distillation?
4. How is structured pruning different from unstructured pruning?
5. quantization 과 pruning 의 차이를 한국어로 짧게 설명해줘.
6. 모델 경량화 검증에서 꼭 봐야 할 지표를 3개 말해줘.
7. Explica brevemente la cuantización de modelos.
8. Dame dos ideas para desplegar LLMs en dispositivos edge.
9. hello
10. Write one short sentence about transformers.

표기:
- `∅`: empty string (`''`)
- `␠`: single space (`' '`)
- `h×24`: `hello` 24회 반복 문자열
- `ko-3-loop`: `" 3은 3과 3과 3을 3으로 3으로 3으로 3으로 3으로 3으로 3으로 3으로 3으로 3으로 3으로 3으로"`

| Prompt # | base | base-distilled | compressed-l29 | base-llmc-awq | compressed-l29-distilled | compressed-l29-llmc-awq | base-distilled-llmc-awq | compressed-l29-distilled-llmc-awq |
|---:|---|---|---|---|---|---|---|---|
| 1 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |
| 2 | ∅ | ∅ | ∅ | ∅ | ∅ | ␠ | ∅ | ␠ |
| 3 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |
| 4 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |
| 5 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |
| 6 | ∅ | ∅ | ∅ | ∅ | ∅ | ko-3-loop | ∅ | ko-3-loop |
| 7 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |
| 8 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |
| 9 | h×24 | h×24 | h×24 | ∅ | h×24 | ∅ | ∅ | ∅ |
| 10 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |

해석:
- 현재 설정에서는 대부분 프롬프트가 빈 문자열로 수렴합니다.
- `avg_similarity`/`exact_match`가 높게 나오는 구간은 "빈 문자열끼리 일치" 영향이 큽니다.
- 따라서 이 지표는 임시 프록시로만 보고, 실제 대회 성능 판단은 별도 벤치마크로 확인해야 합니다.

### 5) 추천 선택지 (현 시점)
1. 품질 보수적: `models/compressed-l29`
- 문자열 프록시 보존(1.0) + 소폭 용량 절감 + 속도 개선

2. 용량 우선: `models/base-llmc-awq` 또는 `models/compressed-l29-llmc-awq`
- 약 45~46% 용량 절감
- 다만 출력 안정성/품질 프록시 하락

### 6) 재현 명령
```bash
# 8-case 비교
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
This repo builds EXAONE-4.0-1.2B compressed variants compatible with the fixed `project_rules.txt` runtime and HF-style `submit.zip/model/*` packaging.

### 2) Worked path
1. Baseline validation on vLLM eager mode
2. Structured pruning (30 -> 29 layers)
3. Distillation-style recovery
4. AWQ quantization via `llmcompressor + compressed-tensors`
5. Unified 8-case evaluation

### 3) Fresh rerun summary
Fresh report file:
- `/workspace/exaone-compression-clean/outputs/eval_compare_all_8cases_rerun.json`

Key point:
- Most prompts still collapse to empty strings across many variants.
- So current `avg_similarity` / `exact_match` should be treated as a weak proxy (empty-vs-empty matches inflate scores).

### 4) Prompt-output matrix
The full 10-prompt, 8-model matrix is listed in the Korean section above (same run, same report).

### 5) Practical recommendation
1. Quality-conservative: `models/compressed-l29`
2. Size-first: `models/base-llmc-awq` or `models/compressed-l29-llmc-awq`
3. Do not trust current proxy metrics alone for leaderboard decisions; run stronger quality checks.
