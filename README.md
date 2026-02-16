# EXAONE Compression Toolkit

## 한국어 (KOR)

### 1) 이 저장소의 현재 상태 (처음부터가 아님)
이 저장소는 "기초 세팅" 단계가 아니라, 아래 검증을 이미 끝낸 상태를 기준으로 정리되어 있습니다.

- 기본 모델 다운로드/로딩 검증 완료: `models/base`
- 압축 모델 후보 검증 완료: `models/compressed-l29`
- 제출물 생성 검증 완료: `submit_compressed_l29.zip` (zip 최상위 `model/`만 존재)
- vLLM/Transformers 양쪽 로딩 및 생성 성공 확인

즉, 이 README는 "무에서 시작"이 아니라, 이미 통과한 경로를 재현하는 실행 가이드입니다.

검증 스냅샷(동일 프롬프트 `hello`, eager 모드 기준):

- baseline (`models/base`): `27.677s`
- compressed (`models/compressed-l29`): `23.509s`
- 속도비 (`baseline/compressed`): `1.177x`
- 샘플 출력 패턴: baseline/compressed 모두 `hellohello...` 형태 유지

### 2) 여기까지 오기 위해 확정한 기술 선택

1. 런타임 버전 고정
- 대회 평가 환경 기준으로 `torch==2.9.0+cu128`을 사용합니다.
- `vllm==0.14.1` 메타데이터는 `torch==2.9.1`을 요구하지만, 실제 런타임은 `2.9.0+cu128`에서 동작 검증했습니다.
- 이를 재현 가능하게 유지하기 위해 `pyproject.toml`의 `tool.uv.override-dependencies`를 사용합니다.

2. 압축 방식
- `scripts/05_awq_quantize.py`는 현재 AWQ가 아니라 "레이어 드롭(layer-drop)" 방식입니다.
- EXAONE-4.0-1.2B의 30개 레이어 중 균등 샘플링으로 일부만 유지합니다.

3. 기본 제출 후보 선택 기준
- 여러 레이어 수를 시험했을 때, 너무 공격적(예: 24~28)은 샘플 출력 품질 붕괴가 잦았습니다.
- `29/30`은 샘플 프롬프트 기준 출력 패턴을 유지하면서 속도 이득이 있어 기본 후보로 채택했습니다.

### 3) 권장 실행 환경 (RunPod)

- Python: `3.11.x`
- CUDA: `12.8`
- vLLM: `0.14.1`
- 핵심 Torch 스택:
  - `torch==2.9.0+cu128`
  - `torchaudio==2.9.0+cu128`
  - `torchvision==0.24.0+cu128`
  - `triton==3.5.0`

### 4) 시작 위치 (RunPod에서 처음 세팅)

```bash
git clone <your-repo-url>
cd exaone-compression

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
```

### 5) 검증된 전체 파이프라인 (순서 그대로 실행)

```bash
make sync
make check
make baseline
make compress
make verify
make verify-compressed
make verify-tfm
make verify-tfm-compressed
make package-compressed
```

각 명령의 의미:

- `make sync`: 평가 환경 호환 버전으로 의존성 동기화
- `make check`: 파이썬/torch/vllm/transformers 버전 체크
- `make baseline`: 기본 모델 다운로드 + 파일 유효성 검사
- `make compress`: 30 -> 29 레이어 압축 모델 생성 (`models/compressed-l29`)
- `make verify`: 기본 모델 vLLM 생성 테스트
- `make verify-compressed`: 압축 모델 vLLM 생성 테스트
- `make verify-tfm`: 기본 모델 Transformers 생성 테스트
- `make verify-tfm-compressed`: 압축 모델 Transformers 생성 테스트
- `make package-compressed`: 제출 zip 생성 (`submit_compressed_l29.zip`)

### 6) 결과물 경로

- 기본 모델: `models/base`
- 압축 모델: `models/compressed-l29`
- vLLM 검증 리포트: `outputs/verify_vllm_compressed.json`
- 제출 파일: `submit_compressed_l29.zip`

### 7) 제출 형식 검증 (반드시 확인)

```bash
python - <<'PY'
import zipfile
z = zipfile.ZipFile("submit_compressed_l29.zip")
roots = sorted({n.split("/")[0] for n in z.namelist() if n.strip()})
print("top_level_entries:", roots)
print("valid:", roots == ["model"])
PY
```

예상 결과:
- `top_level_entries: ['model']`
- `valid: True`

### 8) 운영 메모

- `pip check`에서 `vllm -> torch==2.9.1` 경고가 나올 수 있습니다.
- 이 프로젝트는 평가 환경 재현(`torch==2.9.0+cu128`)을 우선하며, 최종 판단은 `make verify*` 런타임 테스트 기준으로 합니다.
- 기본 모델 vLLM 검증이 불안정하면 `make verify-safe`를 사용하세요.

---

## English (ENG)

### 1) Current repository state (not a zero-state setup)
This repository is already beyond initial setup. The following path has been
validated:

- Baseline model download/load validated: `models/base`
- Compressed candidate validated: `models/compressed-l29`
- Submission packaging validated: `submit_compressed_l29.zip` (zip root is only `model/`)
- Both vLLM and Transformers load/generate checks passed

This README is a reproducible runbook for that validated path.

Validation snapshot (same prompt `hello`, eager mode):

- baseline (`models/base`): `27.677s`
- compressed (`models/compressed-l29`): `23.509s`
- speed ratio (`baseline/compressed`): `1.177x`
- sample output pattern: both baseline/compressed kept `hellohello...`

### 2) Decisions that got us here

1. Runtime pinning
- We target the competition environment and pin `torch==2.9.0+cu128`.
- `vllm==0.14.1` metadata asks for `torch==2.9.1`, but runtime was validated on `2.9.0+cu128`.
- Reproducibility is enforced via `tool.uv.override-dependencies` in `pyproject.toml`.

2. Compression method
- `scripts/05_awq_quantize.py` currently implements structured layer-drop (not AWQ yet).
- For EXAONE-4.0-1.2B, layers are selected with uniform sampling.

3. Default submission candidate
- More aggressive drops (for example 24~28) often produced unstable sample outputs.
- `29/30` gave a safer quality/speed tradeoff, so it is the default candidate.

### 3) Recommended environment (RunPod)

- Python: `3.11.x`
- CUDA: `12.8`
- vLLM: `0.14.1`
- Torch stack:
  - `torch==2.9.0+cu128`
  - `torchaudio==2.9.0+cu128`
  - `torchvision==0.24.0+cu128`
  - `triton==3.5.0`

### 4) Starting point (first setup on RunPod)

```bash
git clone <your-repo-url>
cd exaone-compression

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
```

### 5) Full validated pipeline (run in this order)

```bash
make sync
make check
make baseline
make compress
make verify
make verify-compressed
make verify-tfm
make verify-tfm-compressed
make package-compressed
```

What each step does:

- `make sync`: dependency sync with evaluation-compatible pins
- `make check`: version checks for python/torch/vllm/transformers
- `make baseline`: baseline model download + validation
- `make compress`: builds 30 -> 29 layer compressed model (`models/compressed-l29`)
- `make verify`: baseline vLLM smoke test
- `make verify-compressed`: compressed model vLLM smoke test
- `make verify-tfm`: baseline Transformers smoke test
- `make verify-tfm-compressed`: compressed model Transformers smoke test
- `make package-compressed`: build submission zip (`submit_compressed_l29.zip`)

### 6) Output locations

- Baseline model: `models/base`
- Compressed model: `models/compressed-l29`
- vLLM report: `outputs/verify_vllm_compressed.json`
- Submission zip: `submit_compressed_l29.zip`

### 7) Submission format check

```bash
python - <<'PY'
import zipfile
z = zipfile.ZipFile("submit_compressed_l29.zip")
roots = sorted({n.split("/")[0] for n in z.namelist() if n.strip()})
print("top_level_entries:", roots)
print("valid:", roots == ["model"])
PY
```

Expected:
- `top_level_entries: ['model']`
- `valid: True`

### 8) Operational notes

- `pip check` may report `vllm -> torch==2.9.1` mismatch.
- In this project, priority is evaluation-environment compatibility (`torch==2.9.0+cu128`) plus runtime smoke tests (`make verify*`).
- If baseline vLLM is unstable on your host, use `make verify-safe`.
