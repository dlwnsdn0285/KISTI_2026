# 재진술에 원문 정보가 빠지거나 더해지지 않았는지 확인한다. CLaRa 논문 그림 17의 프롬프트를 그대로 쓴다.

## SYSTEM

## USER
You are given two texts:
Original Document – the source text containing the key information.
Paraphrased Document – a rewritten version of the original.
Your task is to check whether the paraphrased document fully preserves all the key information from the original document, without adding any new information.
Guidelines:
(1) "Key information" means the essential facts, arguments, data, and main ideas of the original.
(2) The paraphrased document must:
(2.1) Contain all the key information from the original.
(2.2) Not omit any important point.
(2.3) Not introduce information that is absent in the original.
(2.4) Preserve meanings without distortion or contradiction.
(2.5) Differences in style, sentence structure, or wording are acceptable as long as the meaning is preserved.
(3) Output format:
(3.1) Answer "Yes" if the paraphrased document completely retains all key information and introduces no new information.
(3.2) Answer "No" if any key information is lost, altered, or if extra/unwarranted information appears.
Don't output any explanation.
Input:
Original Document: {paragraph}
Paraphrased Document: {paraphrase}
Output:
