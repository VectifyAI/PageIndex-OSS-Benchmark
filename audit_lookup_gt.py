"""Flag questions whose reference answer is not actually stated in the document.

A lookup answer should appear verbatim on one of its evidence pages. When it
does not, the answer had to be computed — two figures summed, a difference
taken, a percentage worked out — which makes the question a `derive`, not a
`lookup`, however it is labelled. NETFLIX_2015_10K is the case that motivated
this: its reference answer 253.3 is 133.2 + 120.1, and neither the sum nor the
phrase appears anywhere in the source.

Absence is a flag, not a verdict. Text extraction differs from what a reader
sees, numbers get written "1,358,000" or "1.4 billion", and a Str answer may be
a faithful paraphrase. Everything flagged here needs eyes on it before removal.

    python audit_lookup_gt.py [QUESTIONS_JSON]
"""
import json
import re
import sys
import warnings

import fitz

DOCS = "MMLongBench-Doc/data/documents"


def page_text(doc_id, pages):
    """Concatenated text of the 1-indexed evidence pages, plus neighbours.

    Neighbours are included because evidence page numbering and the PDF's own
    pagination disagree often enough to cause false alarms on their own.
    """
    with fitz.open(f"{DOCS}/{doc_id}") as d:
        wanted = set()
        for p in pages:
            wanted |= {p - 1, p, p + 1}
        return "\n".join(d[i - 1].get_text()
                         for i in sorted(wanted) if 1 <= i <= len(d))


def norm(s):
    return re.sub(r"[\s,$%]", "", s).lower()


def numbers_in(text):
    return {norm(m) for m in re.findall(r"\d[\d,]*\.?\d*", text)}


def stated(answer, fmt, text):
    """Is this answer present as written, rather than derived from the page?"""
    if fmt in ("Int", "Float"):
        a = norm(answer)
        if a in numbers_in(text):
            return True
        # 253.3 stated as "253.30", or 1383 as "1,383.0"
        try:
            val = float(a)
        except ValueError:
            return False
        return any(abs(float(n) - val) < 1e-9
                   for n in numbers_in(text) if re.fullmatch(r"[\d.]+", n))
    if fmt == "List":
        items = re.findall(r"'([^']*)'|\"([^\"]*)\"", answer)
        items = [a or b for a, b in items] or [answer]
        return all(norm(i) in norm(text) for i in items)
    return norm(answer) in norm(text)


def main():
    warnings.filterwarnings("ignore")
    path = sys.argv[1] if len(sys.argv) > 1 else "mmlb-v2-puretext-lookup/questions.json"
    rows = json.load(open(path))

    flagged = []
    for r in rows:
        pages = json.loads(r["evidence_pages"])
        if not pages:
            continue
        text = page_text(r["doc_id"], pages)
        if not stated(r["answer"], r["answer_format"], text):
            flagged.append({**r, "evidence_text_chars": len(text)})

    print(f"{len(rows)} questions | {len(flagged)} reference answers not found "
          f"verbatim on their evidence pages\n")
    for r in flagged:
        print(f"--- {r['doc_id'][:44]}  p{r['evidence_pages']}  [{r['answer_format']}]")
        print(f"    Q : {r['question'][:150]}")
        print(f"    GT: {r['answer'][:120]}")
    json.dump(flagged, open("data/gt_audit_flagged.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote data/gt_audit_flagged.json — inspect before removing anything")


if __name__ == "__main__":
    main()
