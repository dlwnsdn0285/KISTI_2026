# 3장 1절: 세부 정보와 주장/근거 연결을 보존하는 과학기술 문헌 압축 요약

각 문헌을 문단 단위로 나눠 잠재 표현으로 압축하고, 생성 모델이 여러 문헌의 잠재 표현을 질의와 함께 읽고 답하도록 학습합니다.
인코더(Qwen3-Embedding-0.6B)는 부분 단위(문단)마다 잠재 표현을 만들고, 프로젝터가 이를 생성 모델(Qwen3.5-9B)의 입력 공간으로 변환합니다.

| 구성 요소 | 모델 | 학습 범위 |
|---|---|---|
| 인코더 | `Qwen/Qwen3-Embedding-0.6B` | 전체 파라미터. 청크 평균 슬롯 입력, 양방향 어텐션 |
| 프로젝터 | 2층 MLP | 전체 |
| 생성 모델 | `Qwen/Qwen3.5-9B` | 고정. 잠재 표현을 QKV로 변환하는 선형 층에만 LoRA (rank 32, 전체 어텐션 층 8개) |
| 학습 데이터 생성 | `Qwen/Qwen3.8-27B` | 재진술, 단일 사실 QA, 복합 QA, 요지 QA |
| 평가 모델 | `openai/gpt-oss-20b` | M3SciQA 평가 프롬프트 |

## 학습

| 단계 | 학습 내용 | 데이터 |
|---|---|---|
| 단일 문단 요약 학습 단계 | 부분 잠재 요약 하나로 원문 복원, 재진술, 질의응답(단일 사실 QA, 복합 QA, 요지 QA) | 학습 문헌의 문단 |
| 다중 문헌 요약 활용 학습 단계 | 문헌 2편의 부분 잠재 요약과 질문으로 답 생성 | IM-SciQA 학습 데이터 |

단일 문단 요약 학습 단계는 문장 단위 원문 복원(문장당 잠재 표현 2개)으로 시작해 문단 단위 학습으로 이어집니다.
`scripts/train.sh`를 한 번 실행하면 두 단계를 끝까지 이어서 학습합니다.
손실은 정답 토큰의 교차 엔트로피이며, 지시문과 질문 토큰은 제외합니다. 인코더, 프로젝터, 잠재 표현용 LoRA만 갱신합니다.

문단 중요도 기반 적응형 배분은 문단마다 1~5점의 중요도를 매기고, 소프트맥스(온도 2)로 문헌당 총 슬롯 수를 나눕니다.
중요도는 학습 QA의 정답 근거에 가까운 정도로 정하며, 오라클 배분은 다중 문헌 요약 활용 학습 단계부터 적용합니다.

## 파일

```
src/scisum/
├── model/          # 인코더, 프로젝터, 생성 모델(잠재 표현용 LoRA), 문헌 → 부분 잠재 요약
├── data/           # 부분 단위 텍스트, 단계별 학습 데이터
├── train/run.py    # 단일 문단 요약 학습 → 다중 문헌 요약 활용 학습
├── infer/answer.py # IM-SciQA 답 생성 (잠재 표현 입력 / 텍스트 입력)
├── eval/judge.py   # M3SciQA 프롬프트 채점
└── prompts.py
pipeline/
├── parse/          # PMC 원문 수집, IM-SciQA 분할, 부분 단위 구성
├── allocate/       # 문단 중요도, 슬롯 배분 (균등 압축률, 오라클)
├── synthesize/     # 재진술, 단일 사실 QA, 복합 QA, 요지 QA 생성과 검증
└── prompts/        # 학습 데이터 생성 프롬프트
baselines/          # 제목+초록, LLMLingua-2, Keynote 입력 생성
scripts/            # prepare_data.sh, synthesize.sh, train.sh, evaluate.sh, baselines.sh
tests/              # 단위 테스트, smoke/ (전체 경로 스모크 테스트)
```

## 실행 방법

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab')"

# 1) 데이터 준비: PMC 원문 수집 → IM-SciQA 분할 → 부분 단위 → 문단 중요도 → 슬롯 배분
bash scripts/prepare_data.sh

# 2) 학습 데이터 생성 (Qwen3.8-27B 서버 필요)
vllm serve Qwen/Qwen3.8-27B --reasoning-parser qwen3 --enable-prefix-caching --port 8000
bash scripts/synthesize.sh

# 3) 학습 (결과: runs/<RECIPE>/multidoc/best.pt)
NG=4 RECIPE=ours   bash scripts/train.sh   # 원문 복원 + 재진술 + 질의응답, 균등 압축률
NG=4 RECIPE=oracle bash scripts/train.sh   # + 적응형 배분 (오라클)
NG=4 RECIPE=noqa   bash scripts/train.sh   # 원문 복원 + 재진술

# 4) 평가 (gpt-oss-20b 서버 필요)
vllm serve openai/gpt-oss-20b --port 8000
RECIPE=ours bash scripts/evaluate.sh

# 5) 베이스라인: 문헌 전문, 제목+초록, LLMLingua-2, Keynote
bash scripts/baselines.sh
```

`NG`는 GPU 수, `BS`는 GPU당 배치 크기(기본 `4,4,4`)입니다. 유효 배치(128, 32, 8)는 고정되어 있어 GPU 수를 바꿔도 같은 설정으로 학습하며, GPU 수는 유효 배치를 나누어떨어지게 정합니다(1, 2, 4, 8).
생성 서버 주소는 `BASE_URL`, 평가 서버 주소는 `JUDGE_URL`로 바꿀 수 있습니다. 생성, 학습, 평가는 모두 중단된 지점부터 이어서 실행됩니다.

## 테스트

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=src pytest tests/   # 단위 테스트 (CPU)
bash tests/smoke/run_smoke.sh                         # 전체 경로 스모크 테스트 (CPU, 약 2분)
```

스모크 테스트는 IM-SciQA 학습 6 / 검증 2 / 평가 2문항과 그 문헌 20편으로 데이터 준비, 학습 데이터 생성, 학습, 평가, 제목+초록 베이스라인을 차례로 실행합니다.
모델은 작은 무작위 모델로, Qwen3.8-27B와 gpt-oss-20b 서버는 형식만 맞춘 모의 서버(`tests/smoke/mock_llm_server.py`)로 대신합니다.

## 데이터

논문 원문과 학습 데이터는 포함하지 않으며, `scripts/prepare_data.sh`와 `scripts/synthesize.sh`로 만듭니다.

- IM-SciQA: 공개 데이터 `MRAGDATASET/Pubmed-MHQA`를 문헌이 겹치지 않게 학습 9,096 / 검증 1,000 / 평가 1,000문항으로 나눈 것입니다.
  실험에 쓴 분할은 `pipeline/parse/reported_split_ids.json`에 있습니다.
- 논문 원문: NCBI BioC-PMC API로 내려받아 문단으로 나눕니다. 참고문헌, 표, 그림, 부록은 제외합니다.
- 부분 단위: 제목+초록 하나와 본문 문단들입니다. 본문 문단은 섹션 이름을 앞에 붙여 인코더에 넣습니다.
- 생성 결과: `data/synth/paraphrase.json`(재진술), `data/synth/qa.json`(단일 사실 QA, 복합 QA, 요지 QA).

## 결과

IM-SciQA 평가 데이터 1,000문항, 압축률 16배(실제 14.9배), GPT-oss-20B 채점(정답 1점, 부분 정답 0.5점, 오답 0점) 평균입니다.

| 입력 방식 | 압축 방식 | 문헌당 토큰 수 | 압축률 | 정답률 |
|---|---|---|---|---|
| 문헌 전문 | - | 6,279.6 | 1x | 0.711 |
| 제목 + 초록 | 추출 요약 | 402.2 | 15.6x | 0.212 |
| LLMLingua-2 | 추출 요약 | 485.9 | 12.9x | 0.283 |
| Keynote | 추상 요약 | 608.5 | 10.3x | 0.299 |
| Ours | 잠재 표현 | 420.9 | 14.9x | **0.564** |

| 입력 방식 | 정답률 |
|---|---|
| 문헌 전문 | 0.711 |
| Ours (원문 복원 및 재진술) | 0.444 |
| + 질의응답 | 0.564 |
| + 적응형 배분 (오라클) | **0.628** |
