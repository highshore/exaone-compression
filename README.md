# EXAONE Compression Toolkit

## 한국어 (KOR)

### 1) 목표와 범위
이 저장소는 `project_rules.txt` 기준(고정 평가 환경 + `submit.zip` 구조)에서,
EXAONE-4.0-1.2B를 **vLLM에서 실제 구동 가능한 형태로 경량화**하기 위한 실행형 파이프라인입니다.

핵심 목표:
- 모델 크기 축소
- 토큰당 추론 시간 개선
- 품질 붕괴 최소화
- 제출 형식(`submit.zip/model/*`) 준수

### 2) 최종 채택한 전략 (실제로 동작 검증된 조합)
1. Structured Pruning (Layer Drop)
- `scripts/05_awq_quantize.py`
- 30개 레이어 중 29개 유지(`target-layers=29`, uniform)
- 결과물: `models/compressed-l29`

2. Distillation-Style Recovery Calibration
- `scripts/07_distill_student.py`
- Teacher(`models/base`) 응답을 생성해 Student(`models/compressed-l29`)의 `lm_head`만 보정
- 안정 검증된 설정: `num_samples=20`, `epochs=1`, `batch_size=2`, `lr=5e-6`
- 결과물: `models/compressed-l29-distilled`

3. vLLM 기반 비교 평가
- `scripts/08_eval_compare.py`
- baseline vs compressed vs distilled를 같은 프롬프트 세트로 비교
- 비교 지표:
  - elapsed time
  - speedup vs baseline
  - avg similarity to baseline
  - exact match rate

### 3) 평가 환경 호환성 (중요)
`project_rules.txt`의 평가 서버 버전에 맞춰 동작하도록 구성했습니다.

- Python: `3.11`
- CUDA: `12.8`
- `torch==2.9.0+cu128`
- `transformers==4.57.3`
- `vllm==0.14.1`

참고:
- `vllm` 메타데이터는 `torch==2.9.1`을 요구할 수 있지만,
  실제 런타임 검증은 `torch==2.9.0+cu128`에서 통과한 조합으로 고정했습니다.

### 4) RunPod에서 처음부터 재현하는 순서
```bash
git clone <your-repo-url>
cd exaone-compression

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
```

### 5) 전체 파이프라인 (동작 검증 완료 순서)
```bash
make sync
make check
make baseline
make compress
make verify
make verify-compressed
make verify-tfm
make verify-tfm-compressed
make distill
make verify-distilled
make verify-tfm-distilled
make evaluate
make package-compressed
make package-distilled
```

### 6) vLLM 동작 검증 예시 (프롬프트/출력)
검증 스크립트:
- `scripts/02_verify_vllm.py`

예시 프롬프트:
- `hello`

예시 출력 패턴(실제 검증 시 관찰):
- baseline: `hellohello...`
- compressed-l29: `hellohello...`
- compressed-l29-distilled: `hellohello...`

의미:
- 압축/보정 후에도 최소 생성 경로가 끊기지 않고,
- vLLM 로드 + 생성이 모두 성공함을 확인했습니다.

### 7) 산출물
- Baseline: `models/base`
- Compressed: `models/compressed-l29`
- Distilled: `models/compressed-l29-distilled`
- 평가 리포트: `outputs/eval_compare.json`
- 제출 파일:
  - `submit_compressed_l29.zip`
  - `submit_compressed_l29_distilled.zip`

### 8) 제출 형식 검증
```bash
python - <<'PY'
import zipfile
for p in ["submit_compressed_l29.zip", "submit_compressed_l29_distilled.zip"]:
    z = zipfile.ZipFile(p)
    roots = sorted({n.split('/')[0] for n in z.namelist() if n.strip()})
    print(p, roots, roots == ["model"])
PY
```

예상:
- 각 zip의 최상위 엔트리는 `model` 하나만 존재

---

## English (ENG)

### 1) Goal and Scope
This repository provides a reproducible compression pipeline for EXAONE-4.0-1.2B,
aligned with `project_rules.txt` constraints (fixed evaluation stack + `submit.zip` structure).

Primary goals:
- reduce model size
- improve inference efficiency
- keep quality degradation controlled
- satisfy packaging rules (`submit.zip/model/*`)

### 2) Final Strategy We Kept (worked path only)
1. Structured Pruning (Layer Drop)
- `scripts/05_awq_quantize.py`
- Keep 29 out of 30 layers (`target-layers=29`, uniform)
- Output: `models/compressed-l29`

2. Distillation-Style Recovery Calibration
- `scripts/07_distill_student.py`
- Generate teacher outputs from `models/base`, then calibrate only the student `lm_head`
- Stable settings: `num_samples=20`, `epochs=1`, `batch_size=2`, `lr=5e-6`
- Output: `models/compressed-l29-distilled`

3. vLLM-based Comparative Evaluation
- `scripts/08_eval_compare.py`
- Compares baseline vs compressed vs distilled on the same prompt set
- Metrics:
  - elapsed time
  - speedup vs baseline
  - average text similarity to baseline
  - exact match rate

### 3) Evaluation Compatibility (important)
Configured to match the package versions in `project_rules.txt`:

- Python: `3.11`
- CUDA: `12.8`
- `torch==2.9.0+cu128`
- `transformers==4.57.3`
- `vllm==0.14.1`

Note:
- `vllm` package metadata can still declare `torch==2.9.1`.
- We keep the runtime pinned to `torch==2.9.0+cu128` because that is the target eval environment and runtime checks pass under that pin.

### 4) Clean RunPod Setup
```bash
git clone <your-repo-url>
cd exaone-compression

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
```

### 5) Full Pipeline (validated order)
```bash
make sync
make check
make baseline
make compress
make verify
make verify-compressed
make verify-tfm
make verify-tfm-compressed
make distill
make verify-distilled
make verify-tfm-distilled
make evaluate
make package-compressed
make package-distilled
```

### 6) vLLM Validation Example (prompt/output)
Validation script:
- `scripts/02_verify_vllm.py`

Example prompt:
- `hello`

Observed output pattern:
- baseline: `hellohello...`
- compressed-l29: `hellohello...`
- compressed-l29-distilled: `hellohello...`

Interpretation:
- vLLM load and generation succeed for all three checkpoints.

### 7) Artifacts
- Baseline: `models/base`
- Compressed: `models/compressed-l29`
- Distilled: `models/compressed-l29-distilled`
- Comparison report: `outputs/eval_compare.json`
- Submission zips:
  - `submit_compressed_l29.zip`
  - `submit_compressed_l29_distilled.zip`

### 8) Submission Structure Check
```bash
python - <<'PY'
import zipfile
for p in ["submit_compressed_l29.zip", "submit_compressed_l29_distilled.zip"]:
    z = zipfile.ZipFile(p)
    roots = sorted({n.split('/')[0] for n in z.namelist() if n.strip()})
    print(p, roots, roots == ["model"])
PY
```

Expected:
- each zip has only one top-level entry: `model`
