# 1장 1절: 과학기술 문헌 인용 네트워크를 활용한 효율적 검색, 수집 기술 개발

서베이 제목 하나를 받아, 그 서베이가 인용할 논문을 찾는 검색 에이전트.

```mermaid
flowchart LR
  A[서베이 제목] --> B[검색어 생성]
  B --> C[S2 API 검색<br/>시드]
  C --> D["기준 풀<br/>(시드 + 참고문헌)"]
  D --> E["확장<br/>(참고문헌 · 피인용)"]
  E --> F["LLM 판정<br/>(0~9 등급)"]
  F --> G[최종 목록]
```

| 단계 | 구성 요소 |
|---|---|
| 검색어 생성 · 판정 | `Qwen/Qwen3-8B` (vLLM 로컬 서빙) |
| 시드 검색 | Semantic Scholar 검색 API |
| 확장 | 인용 그래프(`datalake` 또는 `s2`) + Contextual bandit 점수 식(`priors/expansion_prior.json`) |
| 제목 관련도 | `sentence-transformers/all-MiniLM-L6-v2` |

## 사용법

### 1. 설치

```bash
pip install -r requirements.txt
pip install vllm==0.8.5.post1          # LLM 서버용
cp config.example.yaml config.yaml
```

### 2. 데이터 준비 (둘 중 하나)

```bash
python prepare_data.py --backend datalake --datalake-root /path/to/science_datalake
python prepare_data.py --backend s2 --data-dir /path/to/data
```

- `datalake`: KISTI 제공 science datalake 패키지를 변환 없이 사용
- `s2`: Semantic Scholar 벌크 데이터를 받아 적재(175GB). `S2_API_KEY` 필요

완료 시 출력되는 경로를 `config.yaml`에 기입.

### 3. 실행

```bash
export S2_API_KEY=...
bash serve_llm.sh      # 별도 터미널에서 실행

# 단일 제목 실행
python run.py --config config.yaml --backend datalake \
    --title "Graph Neural Networks for Recommender Systems" --cutoff 2024-06-01

# 목록 파일 일괄 실행
# queries.jsonl: {"qid": "q1", "title": "서베이 제목", "cutoff": "2025-02-03"}
python run.py --config config.yaml --backend datalake --queries queries.jsonl --out results/run
```

| 옵션 | 설명 |
|---|---|
| `--backend` | 인용 그래프 데이터 (`datalake` 또는 `s2`) |
| `--cutoff` | 기준 날짜 이후 출판된 논문 제외 |
| `--blocklist` | 후보에서 제외할 논문 목록 |
| `--train --gold gold.jsonl` | 검색 대신 점수 식 재학습 |


### 4. 채점

```bash
# gold.jsonl: {"qid": "q1", "corpusid": 246975848, "doi": "10.3390/informatics9010006"}
python score.py --config config.yaml --run results/run --gold gold.jsonl --report results/report.json
```

후보 전체와 최종 목록의 recall · precision 계산. 정답 대조 기준: `s2`는 `corpusid`, `datalake`는 `doi`(또는 `arxiv`).

## 파일

```
paper-search-agent/
├── requirements.txt        # 파이썬 패키지
├── config.example.yaml     # 설정 예시
├── prepare_data.py         # 데이터 준비
├── serve_llm.sh            # LLM 서버 실행
├── run.py                  # 검색 실행
├── score.py                # 채점
├── psa/
│   ├── pipeline.py         # 전체 흐름
│   ├── search.py           # 검색어 생성, S2 검색
│   ├── expand.py           # 관련도, 확장 점수 식
│   ├── judge.py            # LLM 판정, 최종 목록 규칙
│   ├── store.py            # 데이터 조회 (datalake · s2)
│   ├── train.py            # 점수 식 학습
│   ├── util.py             # 설정, 시간 측정 등 공통 도구
│   └── prompts/            # 프롬프트
├── priors/                 # 점수 식 계수
└── tests/                  # 테스트
```

## 결과

CSUR 서베이 100편, `s2`, 서베이당 평균. 확장 예산은 모두 같음.

| 확장 방식 | 후보 recall | 최종 recall | 최종 precision |
|---|---|---|---|
| 확장 없음 | 0.449 | — | — |
| 참조만 | 0.553 | 0.138 | 0.167 |
| 피인용만 | 0.555 | 0.156 | 0.158 |
| 번갈아 (고정) | 0.573 | 0.154 | 0.160 |
| 연도 규칙 | 0.529 | 0.141 | 0.165 |
| **ours** | **0.595** | **0.158** | 0.160 |
