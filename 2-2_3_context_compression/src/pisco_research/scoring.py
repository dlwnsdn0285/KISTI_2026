"""Match scorer and run accounting shared by every system (Raw / A / B / C).

Match (paper metric): a prediction is correct if ANY normalized reference
answer is a substring of the normalized prediction. Normalization: lowercase,
remove punctuation, remove articles (a/an/the), collapse whitespace
(SQuAD-style order). This is neither EM nor token F1.

EM / token F1 (SQuAD definitions, same normalization, max over aliases) are
reported alongside Match: Match rewards long answers and gives false
positives such as reference "no" inside "not" (Step 09).

Empty-reference policy: aliases that normalize to "" are dropped (an empty
string is a substring of everything). A sample with no non-empty alias left
is a data error and raises instead of scoring as correct.
"""

import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = set(string.punctuation)


def normalize_answer(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def match(prediction, references):
    pred = normalize_answer(prediction)
    return any(r in pred for r in _usable_refs(references))


def _usable_refs(references):
    refs = [normalize_answer(r) for r in references]
    refs = [r for r in refs if r]
    if not refs:
        raise ValueError(f"no non-empty reference answer after normalization: {references!r}")
    return refs


_YES_NO = re.compile(r"\b(yes|no)\b")


def yes_no_match(prediction, references):
    """Match variant for yes/no references (Step 10b).

    Substring Match is unreliable for "no" (it is inside "not", "know", ...)
    and for "yes" (any explanation that mentions both words counts). Here the
    FIRST whole-word yes/no in the normalized prediction must equal a reference.
    A prediction with no yes/no word is wrong. For references that are not
    yes/no, this is plain match().
    """
    refs = _usable_refs(references)
    if not any(r in ("yes", "no") for r in refs):
        return match(prediction, references)
    found = _YES_NO.search(normalize_answer(prediction))
    return found is not None and found.group(1) in refs


def exact_match(prediction, references):
    pred = normalize_answer(prediction)
    return any(pred == r for r in _usable_refs(references))


def token_f1(prediction, references):
    """Max over aliases of the SQuAD token-overlap F1."""
    pred_tokens = normalize_answer(prediction).split()
    best = 0.0
    for ref in _usable_refs(references):
        ref_tokens = ref.split()
        common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in set(ref_tokens))
        if common:
            precision, recall = common / len(pred_tokens), common / len(ref_tokens)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def score_run(manifest, predictions):
    """manifest: eval-input manifest; predictions: list of prediction records of ONE run.

    Every manifest sample is in the denominator. A missing or failed prediction
    counts as not matched; it is never silently dropped.
    """
    by_id = {p["sample_id"]: p for p in predictions}
    rows = []
    for sample in manifest["samples"]:
        sid = sample["sample_id"]
        pred = by_id.get(sid)
        row = {"sample_id": sid, "status": "missing" if pred is None else pred["status"]}
        if pred is not None and pred["status"] == "ok":
            row["prediction"] = pred["prediction"]
            refs = sample["evaluation"]["reference_answers"]
            row["match"] = match(pred["prediction"], refs)
            row["em"] = exact_match(pred["prediction"], refs)
            row["f1"] = token_f1(pred["prediction"], refs)
        else:
            row["match"], row["em"], row["f1"] = False, False, 0.0
        rows.append(row)

    counts = {"expected": len(manifest["samples"]),
              "attempted": sum(r["status"] != "missing" for r in rows),
              "succeeded": sum(r["status"] == "ok" for r in rows),
              "failed": sum(r["status"] == "failed" for r in rows),
              "matched": sum(r["match"] for r in rows)}
    counts["match_over_expected"] = counts["matched"] / counts["expected"] if counts["expected"] else None
    counts["em_over_expected"] = sum(r["em"] for r in rows) / counts["expected"] if counts["expected"] else None
    counts["f1_over_expected"] = sum(r["f1"] for r in rows) / counts["expected"] if counts["expected"] else None
    counts["complete"] = counts["succeeded"] == counts["expected"]
    return {"rows": rows, "summary": counts}
