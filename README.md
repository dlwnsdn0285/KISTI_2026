# KISTI_RAG

This repository contains the code for a joint research conducted by KAIST DM Lab and KAIST DISL Lab in collaboration with KISTI (Korea Institute of Science and Technology Information) to develop an advanced RAG system for Korea's largest scientific literature database, focusing on technologies for fast and efficient retrieval and analysis of multiple scientific and technical documents

## Project Structure
- `3-3_feedback_then_refine/`: 인용 오류 교정을 위한 근거 기반 피드백 정제 (Feedback-then-Refine). 실행 방법은 폴더 내 README 참고

## Setup Requirements

### Dataset Setup

1. Download both datasets from [KISTI AIDA](https://aida.kisti.re.kr/data/?collection=&q=%EB%85%BC%EB%AC%B8):

   - 국내 논문 전문 텍스트 데이터셋
   - 국내 논문 QA 데이터셋
2. Place both dataset folders in `data/KISTI/`

### Installation

```bash
pip install -r requirements.txt
```

## Running Experiments
