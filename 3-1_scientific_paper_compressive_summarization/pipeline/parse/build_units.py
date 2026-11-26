"""내려받은 논문 원문을 부분 단위(문단) 파일로 나눈다.

참고문헌, 부록, 표, 그림, 각주는 버리고, 문단으로 된 선언 섹션(저자 기여, 이해충돌, 연구비, 윤리)은 남긴다.
선언이 각주로만 실린 논문은 그 내용이 빠진다(실험에 쓴 데이터와 같은 규칙).

입력: data/raw/pmc_bioc/
출력: data/units/<PMCID>.json (제목, 초록, 문단 목록)
실행: python -m pipeline.parse.build_units
"""
import ast
import glob
import os

from pipeline.common.io import load_json, save_json

BIOC_DIR = os.path.join("data", "raw", "pmc_bioc")
UNITS_DIR = os.path.join("data", "units")

DROP_SECTION = {"REF", "APPENDIX", "ABBR", "SUPPL", "SUPPLEMENTARY", "TABLE", "FIG"}
DROP_TYPE = {"TABLE", "TABLE_CAPTION", "TABLE_FOOTNOTE", "FIG", "FIGURE",
             "FIG_CAPTION", "FIGURE_CAPTION", "REF", "FOOTNOTE"}


def load_bioc(pid: str) -> dict:
    path = os.path.join(BIOC_DIR, f"{pid}.json")
    return load_json(path) if os.path.exists(path) else {"passages": []}


def infons(passage: dict) -> dict:
    inf = passage.get("infons", {})
    return ast.literal_eval(inf) if isinstance(inf, str) else inf


def bioc_title(rec: dict) -> str:
    """section_type TITLE인 첫 passage의 텍스트 (공백 정규화). 없으면 ""."""
    for p in rec.get("passages", []):
        if (infons(p).get("section_type") or "").upper() == "TITLE":
            return " ".join((p.get("text") or "").split())
    return ""


def paper_units(pid: str, rec: dict) -> dict:
    title, abstract, paras, section = "", [], [], "OTHER"
    for p in rec.get("passages", []):
        inf = infons(p)
        st = (inf.get("section_type") or "").upper()
        ty = (inf.get("type") or "").upper()
        text = (p.get("text") or "").strip()
        if not text:
            continue
        if st == "TITLE":
            title = title or " ".join(text.split())
            continue
        if st in DROP_SECTION or ty in DROP_TYPE:
            continue
        if "TITLE" in ty:  # 소제목: 이어지는 문단의 섹션 이름
            section = " ".join(text.split())
            continue
        if st == "ABSTRACT":
            abstract.append(text)
            continue
        paras.append({"idx": len(paras), "section": section, "text": " ".join(text.split())})
    return {"paper_id": pid, "title": title, "abstract": " ".join(abstract), "paragraphs": paras}


def main() -> None:
    files = sorted(glob.glob(os.path.join(BIOC_DIR, "*.json")))
    n = empty = n_para = 0
    for f in files:
        pid = os.path.basename(f)[:-5]
        u = paper_units(pid, load_json(f))
        if not (u["title"] or u["abstract"] or u["paragraphs"]):
            empty += 1
            continue
        save_json(u, os.path.join(UNITS_DIR, f"{pid}.json"))
        n += 1
        n_para += len(u["paragraphs"])
    print(f"[units] {n}편 저장 (내용 없음 {empty}편), 문헌당 문단 {n_para / max(n, 1):.1f}개 -> {UNITS_DIR}")


if __name__ == "__main__":
    main()
