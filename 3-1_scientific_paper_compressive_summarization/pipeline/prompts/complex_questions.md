# 문단의 여러 사실을 이어 묻는 복합 질문을 만든다. CLaRa 논문 그림 11의 프롬프트에서 예시만 뺐다.

## SYSTEM

## USER
You are given a document delimited by <doc> and </doc>. Your job is to generate a set of MULTI-HOP questions that, taken together, comprehensively cover the document's key information.
The document is one paragraph of a scientific paper; the paper's title is given only as context.
Paper title: {title}
<doc>
{paragraph}
</doc>
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
OUTPUT FORMAT
Return ONLY a JSON object with keys "Question1", "Question2", ..., "QuestionN".
