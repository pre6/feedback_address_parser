#!/usr/bin/env python
"""Evaluate a trained model on a .spacy file.

    python trainer/evaluate.py --model models/v1/model-best --data data/base/dev.spacy
    python trainer/evaluate.py --model models/v1/model-best --data data/test/test.spacy

Reports, per label:
  - entity-level precision / recall / F1 (an entity counts only if label AND
    exact span match)
  - exact-match rate: fraction of docs where the predicted text for that label
    equals the gold text (the metric your notebook used), counting "no entity"
    matching "no entity"

Use dev.spacy while iterating. Use test.spacy sparingly, for final comparisons
between model versions.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from spacy.training import Example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=0, help="evaluate on first N docs only")
    ap.add_argument("--errors", type=int, default=20, help="print this many mismatches")
    ap.add_argument("--out", default="", help="write JSON report here (default: next to model)")
    args = ap.parse_args()

    nlp = spacy.load(args.model)
    gold_docs = list(DocBin().from_disk(args.data).get_docs(nlp.vocab))
    if args.limit:
        gold_docs = gold_docs[: args.limit]

    examples = [Example(nlp.make_doc(d.text), d) for d in gold_docs]
    scores = nlp.evaluate(examples)

    labels = sorted({e.label_ for d in gold_docs for e in d.ents})
    exact = defaultdict(int)
    shown = 0
    for ex in examples:
        pred = nlp(ex.reference.text)
        for label in labels:
            g = [e.text for e in ex.reference.ents if e.label_ == label]
            p = [e.text for e in pred.ents if e.label_ == label]
            if g == p:
                exact[label] += 1
            elif shown < args.errors:
                shown += 1
                print(f"[{label}] gold={g} pred={p}\n    {ex.reference.text!r}")

    n = len(examples)
    report = {
        "model": args.model,
        "data": args.data,
        "n_docs": n,
        "overall": {k: round(scores[k], 4) for k in ("ents_p", "ents_r", "ents_f")},
        "per_label": {
            lab: {
                "p": round(scores["ents_per_type"].get(lab, {}).get("p", 0.0), 4),
                "r": round(scores["ents_per_type"].get(lab, {}).get("r", 0.0), 4),
                "f": round(scores["ents_per_type"].get(lab, {}).get("f", 0.0), 4),
                "exact_match": round(exact[lab] / n, 4),
            }
            for lab in labels
        },
    }

    print(f"\n{n} docs from {args.data}")
    print(f"{'label':<10}{'P':>8}{'R':>8}{'F1':>8}{'exact':>8}")
    for lab, m in report["per_label"].items():
        print(f"{lab:<10}{m['p']:>8.3f}{m['r']:>8.3f}{m['f']:>8.3f}{m['exact_match']:>8.3f}")
    o = report["overall"]
    print(f"{'ALL':<10}{o['ents_p']:>8.3f}{o['ents_r']:>8.3f}{o['ents_f']:>8.3f}")

    out = Path(args.out) if args.out else Path(args.model).parent / f"eval_{Path(args.data).stem}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
