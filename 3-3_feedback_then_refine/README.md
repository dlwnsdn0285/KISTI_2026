# 3장 3절: 인용 오류 교정을 위한 근거 기반 피드백 정제 (Feedback-then-Refine)

인용 근거 검증(3장 2절)에서 **비지지**로 판정된 (주장 문장, 인용 참고문헌) 쌍을 대상으로,
구조화된 피드백 → 교정 조치(수정 / 재인용 / 인용 삭제) → 재검증을 최대 5회 반복합니다.
모든 모델은 오픈소스이며 vLLM으로 로컬 서빙합니다.

| 역할 | 모델 |
|---|---|
| 판정 (재검증) | `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic` + NLI `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (둘 다 지지일 때만 지지) |
| 피드백 생성 · 정제 | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` |
| 근거 문단 검색 | `BAAI/bge-large-en-v1.5` (참고문헌 원문 문단 Top-5) |

## 파일

```
scripts/
├── common.py           # 경로·서버 설정, 근거 문단 검색, NLI, vLLM 클라이언트, 판정 프롬프트
├── prepare_sample.py   # 3장 2절 비지지 쌍에서 실험 표본 1,000쌍 추출 → data/sample.jsonl
├── run_ftr.py          # Feedback-then-Refine 반복 교정 (4개 비교 조건)
├── analyze.py          # 회차별 지지 비율, 의미 보존 점수 → results/run_main/_summary.json
├── launch_servers.sh   # vLLM 서버 실행 (판정 :8101, 정제 :8102)
└── run.sh              # run_ftr.py → analyze.py
```

## 실행 방법

```bash
pip install -r requirements.txt

# 1) 모델 서버 실행 (GPU 0,1: 판정 모델, GPU 2: 정제 모델 + 검색기·NLI)
bash scripts/launch_servers.sh all

# 2) 실험 표본 생성 (FTR_POOL/unsupported.jsonl → data/sample.jsonl)
cd scripts
python prepare_sample.py

# 3) 반복 교정 + 지표 계산 (결과: results/run_main/)
bash run.sh
# 빠른 확인: python run_ftr.py --limit 20 --loops 2 --out ../results/run_test
```

경로는 환경 변수로 바꿀 수 있습니다: `FTR_SRC`(서베이별 참고문헌 OCR 폴더), `FTR_POOL`(3장 2절 결과
`unsupported.jsonl` 위치), `JUDGE_URL` / `REFINER_URL`(vLLM 주소, 쉼표로 여러 개 지정 가능).

## 입력 데이터

논문 원문과 서베이 데이터는 저작권 문제로 포함하지 않습니다.

- `unsupported.jsonl`: 3장 2절에서 언어모델과 NLI가 모두 비지지로 판정한 쌍(10,424건).
  사용 필드는 `sentence`, `marker_text`, `title`, `paper_title`, `model`, `condition`, `llm_label.label`,
  `provenance[0].reference_number`, `provenance[0].llm_file`입니다.
- 서베이 폴더: `references.jsonl`(`refNo`, `title`)과 `reference_ocr/<refNo>_*.json`(OCR 원문, 400자 이상 줄을 문단으로 사용).

## 비교 조건과 결과 (1,000쌍, 5회 반복)

| 조건 (`--conditions`) | 피드백 | 교정 조치 | 지지 비율 |
|---|---|---|---|
| `no_fb` | 없음 | 수정 | 32.6% |
| `reason_fb` | 판정 사유 | 수정 | 50.0% |
| `struct_fb` | 구조화된 피드백 | 수정 | 61.9% |
| `struct_fb_recite` (제안) | 구조화된 피드백 + 재인용 후보 3개 | 수정 / 재인용 / 삭제 | **72.8%** |

제안 방법의 해결 방식은 수정 44.6%, 재인용 28.2%, 인용 삭제 1.1%(미해결 26.1%)입니다.
판정 절차는 사람 라벨 1,000건에서 정확도 0.989, 비지지를 지지로 잘못 판정한 경우 0건입니다.
