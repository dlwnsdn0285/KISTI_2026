# 2장 2·3절: 압축 컨텍스트의 다중 증거 보완(Additional Memory)과 응답시간 향상

PISCO(x16 압축, `naver/pisco-mistral`)의 문서별 압축 단위 5개는 그대로 두고, 질문과 상위 5개 문서 원문으로 만든
**추가 블록 1개**를 붙여 디코더(Mistral-7B)에 입력합니다(OURS). 압축기·디코더는 고정하고 블록 생성기만 QA loss로 학습합니다.
응답시간은 문서 압축 결과를 미리 캐시하고 추가 블록만 질의 시점에 계산하는 방식으로 측정합니다.

## 실행

```bash
conda create -n pisco-ref python=3.11 -y && conda run -n pisco-ref pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
conda create -n pisco-kisti python=3.12 -y && conda run -n pisco-kisti pip install -r requirements-kisti.txt   # KISTI·응답시간용
bash run/setup.sh      # 모델 다운로드 + 공개 데이터 준비
bash run/train.sh      # 추가 블록 생성기(OURS) + 비교군 Decoder LoRA 학습 (SMOKE=1: 빠른 확인)
bash run/infer.sh      # 영문 벤치마크 추론: 원문 / PISCO / Decoder LoRA / Baseline-6 / OURS
bash run/kisti.sh      # KISTI multi-evidence QA 구축·추론·판정 (KISTI 데이터 필요)
bash run/latency.sh    # 응답시간 TTFT·TTIT·Throughput
python run/summary.py  # 성능·응답시간 표 출력
```

GPU는 `GPU=0`(기본값), KISTI 질문 생성용 32B 모델은 `GEN_GPUS=0,1,2,3`로 지정합니다. `run/`은 실행 래퍼이고,
단계별 원본 코드는 `scripts/`와 `src/pisco_research/`에 있습니다. 결과는 `artifacts/`에 저장됩니다.

## 데이터

데이터와 가중치는 포함하지 않습니다. 공개 데이터는 `run/setup.sh`가 고정 리비전으로 내려받습니다:
HotpotQA(`hotpotqa/hotpot_qa`), 2Wiki(`framolfese/2WikiMultihopQA`), MuSiQue(`dgslibisey/MuSiQue`),
MultiHop-RAG(`yixuantt/MultiHopRAG`), NQ(DPR `biencoder-nq-dev.json.gz`).

KISTI 데이터(논문 원문·QA)는 비공개입니다. 경로는 다음 스크립트 상단 상수에서 바꿉니다:
`scripts/26_kisti_embed.py`(`CORPORA`), `26_kisti_prepare_child125.py`(`KISTI_ROOT`), `26_kisti_prepare_m120.py`·
`28_kisti_multi_evidence.py`(`INDEX_DIR`, `QA_DIR`, 모델 경로), `26_kisti_score.py`·`28_score.py`(`JUDGE_DIR`).

## 결과

| 데이터셋 | 원문 | Baseline (PISCO) | Decoder LoRA | Baseline-6 memory | OURS |
|---|---|---|---|---|---|
| MuSiQue | 0.544 | 0.393 | 0.480 | 0.383 | **0.485** (+9.2%p) |
| MultiHop-RAG | 0.722 | 0.655 | 0.672 | 0.648 | **0.684** (+2.9%p) |
| NQ (단일 증거) | 0.695 | 0.647 | 0.634 | 0.616 | **0.657** (+1.0%p) |
| KISTI multi-evidence QA | 0.698 | 0.439 | 0.477 | 0.404 | **0.495** (+5.6%p) |

| KISTI 문서 수 k | 5 | 10 | 20 | 40 |
|---|---|---|---|---|
| 원문 / Baseline / OURS | 0.698 / 0.439 / 0.495 | 0.579 / 0.368 / 0.404 | 0.481 / 0.288 / 0.354 | 0.474 / 0.249 / 0.270 |

응답시간: 입력 22.6k 토큰(문서 320개)에서 원문 대비 TTFT 7.8배, 처리량 7.2배 (RTX 3090).
영문은 Match(MultiHop-RAG는 yes/no 판정), KISTI는 Qwen2.5-7B 판정 정확도이며 KISTI는 영문 학습 모델을 그대로 적용한 결과입니다.
