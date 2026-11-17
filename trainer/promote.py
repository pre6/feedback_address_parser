#!/usr/bin/env python
"""Promote a candidate model if it beats the current one on the frozen test set.

    python trainer/promote.py --candidate models/v2/model-best
    python trainer/promote.py --candidate models/v2/model-best --dry-run
    python trainer/promote.py --candidate models/v2/model-best --force

State lives in models/registry.json:

    {"current": "v1/model-best",
     "history": [{"timestamp": ..., "candidate": ..., "decision": ..., ...}]}

Paths in the registry are relative to the models dir, so the same file works
on Windows, Linux and inside Docker.

Rules for promotion (all must hold):
  1. candidate overall F1 >= current overall F1 + --min-gain
  2. no label's F1 drops by more than --max-label-drop

If there is no current model yet, the candidate is promoted unconditionally.
Rollback = edit "current" in registry.json (or promote an old version with --force).
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from spacy.training import Example

ROOT = Path(__file__).resolve().parents[1]


def evaluate(model_path: Path, data_path: Path) -> dict:
    nlp = spacy.load(model_path)
    docs = list(DocBin().from_disk(data_path).get_docs(nlp.vocab))
    examples = [Example(nlp.make_doc(d.text), d) for d in docs]
    scores = nlp.evaluate(examples)
    per_label = {lab: round(m["f"], 4) for lab, m in scores["ents_per_type"].items()}
    return {"f": round(scores["ents_f"], 4), "p": round(scores["ents_p"], 4),
            "r": round(scores["ents_r"], 4), "per_label": per_label, "n_docs": len(docs)}


def load_registry(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"current": None, "history": []}


def save_registry(path: Path, reg: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=2))
    os.replace(tmp, path)  # atomic, so the API never reads a half-written file


def decide(cur, cand, min_gain, max_label_drop):
    """Return (promote: bool, reasons: list[str])."""
    if cur is None:
        return True, ["no current model"]
    reasons, ok = [], True
    gain = cand["f"] - cur["f"]
    if gain < min_gain:
        ok = False
        reasons.append(f"overall F1 gain {gain:+.4f} < required {min_gain:+.4f}")
    for lab, f_cur in cur["per_label"].items():
        drop = f_cur - cand["per_label"].get(lab, 0.0)
        if drop > max_label_drop:
            ok = False
            reasons.append(f"{lab} F1 dropped by {drop:.4f} (> {max_label_drop})")
    return ok, reasons or [f"overall F1 gain {gain:+.4f}, no label regressed"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True, help="path to the candidate model dir, e.g. models/v2/model-best")
    ap.add_argument("--test", default="data/test/test.spacy")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--min-gain", type=float, default=0.0, help="required overall F1 improvement")
    ap.add_argument("--max-label-drop", type=float, default=0.01, help="max allowed F1 drop on any single label")
    ap.add_argument("--dry-run", action="store_true", help="evaluate and report, don't change the registry")
    ap.add_argument("--force", action="store_true", help="promote regardless of scores")
    args = ap.parse_args()

    models_dir = (ROOT / args.models_dir).resolve()
    test_path = ROOT / args.test
    reg_path = models_dir / "registry.json"
    if not test_path.exists():
        sys.exit(f"Missing {test_path}. Run trainer/build_dataset.py first.")

    cand_abs = Path(args.candidate)
    cand_abs = (cand_abs if cand_abs.is_absolute() else Path.cwd() / cand_abs).resolve()
    if not cand_abs.exists():
        cand_abs = (ROOT / args.candidate).resolve()
    try:
        cand_rel = cand_abs.relative_to(models_dir).as_posix()
    except ValueError:
        sys.exit(f"Candidate {cand_abs} is not inside {models_dir}")

    reg = load_registry(reg_path)
    cur_rel = reg["current"]

    print(f"Evaluating candidate: {cand_rel}")
    cand = evaluate(cand_abs, test_path)
    cur = None
    if cur_rel and cur_rel != cand_rel:
        print(f"Evaluating current:   {cur_rel}")
        cur = evaluate(models_dir / cur_rel, test_path)
    elif cur_rel == cand_rel:
        print("Candidate is already the current model.")
        cur = cand

    def line(name, m):
        labs = "  ".join(f"{k}={v:.4f}" for k, v in sorted(m["per_label"].items()))
        return f"  {name:<10} F1={m['f']:.4f}  P={m['p']:.4f}  R={m['r']:.4f}  | {labs}"

    print(f"\nTest set: {test_path} ({cand['n_docs']} docs)")
    if cur is not None:
        print(line("current", cur))
    print(line("candidate", cand))

    promote, reasons = decide(cur, cand, args.min_gain, args.max_label_drop)
    if args.force:
        promote, reasons = True, ["--force"] + reasons
    decision = "promoted" if promote else "rejected"
    print(f"\nDecision: {decision.upper()} ({'; '.join(reasons)})")

    if args.dry_run:
        print("(dry run, registry unchanged)")
        return

    reg["history"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidate": cand_rel,
        "previous": cur_rel,
        "decision": decision,
        "reasons": reasons,
        "candidate_scores": cand,
        "current_scores": cur,
    })
    if promote:
        reg["current"] = cand_rel
    models_dir.mkdir(parents=True, exist_ok=True)
    save_registry(reg_path, reg)
    print(f"Registry updated: {reg_path}" + (f" (current = {cand_rel})" if promote else ""))
    sys.exit(0 if promote else 1)  # non-zero lets a Makefile/script stop on rejection


if __name__ == "__main__":
    main()
