# 앞에서 만든 질문이 다루지 않은 사실로 단일 사실 질문을 더 만든다. CLaRa 논문 그림 14의 프롬프트에서 예시만 뺐다.

## SYSTEM

## USER
You are given a document and a set of existing question-answer pairs. Your task is to carefully compare the information covered in the QA pairs against the document and generate additional questions that cover any key information not yet addressed.
Requirements:
- Only generate questions for key facts present in the document that are **not already covered** in the existing QA pairs.
- Do **not** repeat or rephrase the information in the existing question answer pairs.
- Each question should cover **only one or two distinct key pieces of information**.
- Each question must be self-contained, meaning it should be understood by the user without seeing the document.
- All questions should require **simple factual recall only**, with no inference or reasoning.
Input Format:
Document (one paragraph of a scientific paper; paper title for context: {title}):
{paragraph}
Existing QA:
{qa_list}
Output Format:
Return your generated new supplementary questions in the following JSON format:
{
"Number of Supplementary Questions": N,
"Question1": "...",
"Question2": "...",
...,
"QuestionN": "..."
}
If all key information is already covered and no supplementary questions are needed, output an empty JSON object:
{
"Number of Supplementary Questions": 0
}
