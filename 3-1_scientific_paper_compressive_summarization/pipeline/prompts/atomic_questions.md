# 문단의 사실을 하나씩 묻는 단일 사실 질문을 만든다. CLaRa 논문 그림 10의 프롬프트에서 위키백과 예시만 뺐다.

## SYSTEM

## USER
You are given a document delimited by <doc> and </doc>. Your job is to read the given document and generate a comprehensive set of multi-hop questions that fully cover all the key information in the text.
The document is one paragraph of a scientific paper; the paper's title is given only as context.
Paper title: {title}
<doc>
{paragraph}
</doc>
Question Requirements:
You should generate as many questions as necessary to fully cover all the key facts in the document.
(1) Each question must be self-contained, meaning it should be understood by the user without seeing the document.
(2) Each question must cover only one or at most two distinct key pieces of information.
(3) The questions must be non-overlapping — no two questions should target the same piece of information.
(4) The questions should be simple factual recall only — do not require inference, reasoning, or summarization.
(5) Your output should be a list of self-contained, non-overlapping factual questions that together comprehensively cover all the key information in the document.
Your output should be a JSON object with the following format:
{
"Question1": "...",
"Question2": "...",
...,
"QuestionN": "..."
}
