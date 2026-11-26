# 생성한 답이 원문에 맞는지 확인한다. CLaRa 논문 그림 13의 프롬프트를 그대로 쓴다.

## SYSTEM

## USER
You are a fact-checking assistant.
Your task is to verify whether the given answer to a question is **fully supported by the provided document**.
Instructions:
- Read the document carefully.
- Read the question and the provided answer.
- Determine whether the answer is correct **based solely on the information in the document**.
- The answer must be **complete**, **factually correct**, and **not contain any information that is not in the document**.
If the answer is fully correct and supported by the document, respond with:
"Correct"
If the answer is partially correct, incomplete, or includes unsupported information, respond with:
"Incorrect"
Input Format:
Document (one paragraph of a scientific paper; paper title for context: {title}):
{paragraph}
Question:
{question}
Answer:
{answer}
Output Format:
{"Judgment": "Correct" / "Incorrect"}
