# 질문 하나에 답을 생성한다. CLaRa 논문 그림 12의 프롬프트를 그대로 쓰고, 논문 제목을 맥락으로 준다.

## SYSTEM

## USER
You are a factual answering assistant.
Your task is to read the provided document and answer the given question **based only on the information explicitly stated in the document**.
Please output the answer as short as possible.
Requirements:
- Your answer must be based solely on the content of the document.
- Do not use prior knowledge or make assumptions beyond the document.
- If the document does not contain the answer, respond with: "The document does not contain this information."
- The answer should be concise, factual, and complete.
Input Format:
Document (one paragraph of a scientific paper; paper title for context: {title}):
{paragraph}
Question:
{question}
Output Format:
Answer: <YOUR ANSWER HERE>
