# 문단을 재진술한다. CLaRa 논문 그림 16의 프롬프트에서 예시를 빼고, 전문 용어와 약어는 바꾸지 않아도 된다는 규칙을 더했다.

## SYSTEM

## USER
You are given a document. Your task is to paraphrase the document in a way that:
(1) **Restructure extensively** — **Try your best to break down the structure of the original document**, do not keep the same paragraphing, ordering, or sentence flow. Reorganize ideas, shuffle the order, merge or split sentences, and restructure arguments.
(2) **Preserve meaning with absolute accuracy** — ensure that ALL key information and semantics of the original document are retained. Do not omit any factual details, numbers, dates, or specific information.
(3) **Avoid direct copying** — no sentence should remain identical; re-express ideas in a fresh way using synonyms and varied sentence structures.
(4) **CRITICAL: Add no new information** — the paraphrased document cannot introduce:
- New facts, interpretations, or context not explicitly stated
- Organizational names or affiliations not mentioned in the original
- Explanatory details or background information
- Your own analysis or conclusions about the content
(5) **Maintain the original's voice and perspective** — Preserve the document's intended tone and format.
(6) **Verify factual relationships** — when paraphrasing complex information involving multiple entities, dates, or cause-and-effect relationships, double-check that you maintain the correct connections and chronology.
(7) **Use varied vocabulary** — employ synonyms and alternative expressions while maintaining precision (technical terms, names and abbreviations may be kept as they are).
(8) **Preserve completeness** — if the original mentions specific numbers, dates, names, or measurements, include them in your paraphrase (even if reworded).
(9) **Maintain coherence** — the paraphrased version should read as natural and fluent writing.
**WARNING: Do not assume context.** Work only with what is explicitly stated.
Produce a paraphrased version that keeps the meaning and all factual details but has significantly altered structure and wording.
Input:
<document>
{paragraph}
</document>
