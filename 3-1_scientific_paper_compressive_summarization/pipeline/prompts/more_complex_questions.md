# 앞에서 만든 질문이 다루지 않은 사실로 복합 질문을 더 만든다. CLaRa 논문 그림 15의 프롬프트에서 예시만 뺐다.

## SYSTEM

## USER
You are given a document and a set of existing question-answer pairs. Your task is to carefully compare the information covered in the QA pairs against the document and generate additional MULTI-HOP questions that cover any key information not yet addressed.
Requirements:
- Only generate questions for key facts present in the document that are **not already covered** in the existing question answer pairs.
- Do **not** repeat or rephrase information which can be found in the existing question answer pairs.
Question Requirements:
1) Self-contained: Every question must be understandable without viewing the document.
2) Multi-hop only: Each question must require at least TWO independent pieces of evidence from DIFFERENT parts of the document (e.g., different sentences, clauses, or items of the paragraph). If a question can be answered from a single sentence or data point, REJECT it.
3) Non-overlapping: No two questions may target the same fact or the same combination of facts. Each question must have a unique reasoning path and evidence combination.
4) Coverage: Produce as many questions as needed to cover ALL key facts in the document. Prefer many small, precise multi-hop questions over a few large ones.
5) Focus: Each question should target ONE multi-hop objective, typically integrating 2–3 facts (bridging, comparison, aggregation, temporal/causal linking, entity–attribute joining, etc.). Do NOT bundle multiple unrelated sub-questions.
6) Verifiability: The answer to each question must be derivable SOLELY from the document, with no external knowledge or subjective judgment.
7) Clarity: Avoid yes/no questions and vague wording. Use explicit constraints, quantities, and identifiers where relevant.
8) No explanations: Do NOT include rationales, steps, or references—ONLY output the questions as JSON.
9) You can generate 2-hops, 3-hops, 4-hops, etc. questions.
QUESTION TEMPLATES (use as patterns, adapt as needed)
- Bridging: "Which X satisfies BOTH condition A mentioned in [context A] AND condition B mentioned in [context B]?"
- Comparison: "Considering [pivot], which of X or Y meets [criterion] when combining details from [source 1] and [source 2]?"
- Aggregation: "When combining [quantity/info] from part A with [quantity/info] from part B, which single entity matches [combined constraint]?"
- Temporal/Causal: "Based on the timeline described in parts A and B, which event/entity fulfills [temporal/causal relation]?"
Input Format:
Document (one paragraph of a scientific paper; paper title for context: {title}):
{paragraph}
Existing QA:
{qa_list}
Output Format:
Note, do not repeat or paraphrase existing questions. Instead, generate new multi-hop questions for the missing information, and put the new questions in JSON format:
{
"Number of Supplementary Questions": N,
"Question1": "...",
...,
"QuestionN": "..."
}
If all key information is already covered and no supplementary questions are needed, output an empty JSON object:
{
"Number of Supplementary Questions": 0
}
