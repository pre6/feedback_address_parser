#!/usr/bin/env python
"""Build spaCy NER training data from labelled address CSVs.

Usage (from the repo root):

    python trainer/build_dataset.py --input data/raw/addresses.csv

    # different column names / labels:
    python trainer/build_dataset.py --input data/raw/*.csv \
        --text-col LOCATION \
        --label-col ADDRESS=ADDRESS --label-col CITY=CITY \
        --label-col STATE=STATE --label-col COUNTY=COUNTY --label-col ZIP=ZIP

Outputs (under --out-dir, default ./data):
    base/train.spacy, base/dev.spacy
    base/rejected.csv            rows we could not label safely, with reasons
    test/test.spacy              FROZEN held-out set (created once)
    test/test_hashes.txt         text hashes of the test set (for leakage checks)
    test/test_manifest.json      sha256 of test.spacy, to detect tampering

The test set is created the first time you run this and is never regenerated.
On later runs (e.g. with new data) any row whose text is already in the test
set is excluded from train/dev, so nothing leaks. Use --force-new-test only if
you really want to start over (this invalidates comparisons with old models).
"""
import argparse
import glob
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from spacy.tokens import DocBin
from spacy.util import filter_spans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.spans import find_span, has_overlap, is_missing  # noqa: E402
from shared.text import normalize_text, text_hash  # noqa: E402

DEFAULT_LABEL_COLS = ["ADDRESS=ADDRESS", "CITY=CITY", "ZIP=ZIP"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="+", required=True, help="CSV file(s); globs allowed")
    p.add_argument("--text-col", default="LOCATION", help="column with the raw location string")
    p.add_argument(
        "--label-col",
        action="append",
        help="COLUMN=LABEL mapping, repeatable (default: ADDRESS, CITY, ZIP)",
    )
    p.add_argument("--out-dir", default="data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--dev-frac", type=float, default=0.10)
    p.add_argument(
        "--overlap",
        choices=["reject", "longest"],
        default="reject",
        help="rows whose entities overlap: reject the row (default) or keep the longest spans",
    )
    p.add_argument("--drop-empty", action="store_true", help="drop rows with no labelled fields (default: keep as negatives)")
    p.add_argument("--force-new-test", action="store_true", help="regenerate the frozen test set (dangerous)")
    return p.parse_args()


def parse_label_cols(items):
    mapping = {}
    for item in items or DEFAULT_LABEL_COLS:
        if "=" not in item:
            sys.exit(f"--label-col must look like COLUMN=LABEL, got: {item}")
        col, label = item.split("=", 1)
        mapping[col.strip()] = label.strip().upper()
    return mapping


# --------------------------------------------------------------------------- #
# Loading / cleaning
# --------------------------------------------------------------------------- #
def load_inputs(patterns, text_col, label_cols):
    paths = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(pat)) or [pat])
    frames = []
    for path in paths:
        # keep_default_na=False so a value like "NA" isn't silently turned into NaN
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = [c for c in [text_col, *label_cols] if c not in df.columns]
        if missing:
            sys.exit(f"{path}: missing columns {missing}. Found: {list(df.columns)}")
        df = df[[text_col, *label_cols]].copy()
        df["_source"] = Path(path).name
        df["_row"] = range(len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def normalize_frame(df, text_col, label_cols):
    df = df.copy()
    df["_raw_text"] = df[text_col]
    df[text_col] = df[text_col].map(normalize_text)
    for col in label_cols:
        df[col] = df[col].map(lambda v: "" if is_missing(v) else normalize_text(v))
    return df


# --------------------------------------------------------------------------- #
# Building examples
# --------------------------------------------------------------------------- #
def build_examples(df, text_col, label_cols, nlp, overlap_mode):
    """Return (examples, rejected). examples are dicts with text, hash, doc."""
    examples, rejected = [], []

    def reject(row, reason, detail=""):
        rejected.append(
            {
                "source": row["_source"],
                "row": row["_row"],
                "text": row["_raw_text"],
                "reason": reason,
                "detail": detail,
            }
        )

    for _, row in df.iterrows():
        text = row[text_col]
        spans, problems = [], []

        for col, label in label_cols.items():
            span, err = find_span(text, row[col], label)
            if err:
                problems.append(f"{col}:{err}:{row[col]}")
            elif span:
                spans.append(span)

        if problems:
            reason = "|".join(sorted({p.split(":")[1] for p in problems}))
            reject(row, reason, "; ".join(problems))
            continue

        doc = nlp.make_doc(text)
        ents = []
        misaligned = []
        for start, end, label in spans:
            ent = doc.char_span(start, end, label=label, alignment_mode="strict")
            if ent is None:
                misaligned.append(f"{label}:{text[start:end]}")
            else:
                ents.append(ent)
        if misaligned:
            reject(row, "token_misaligned", "; ".join(misaligned))
            continue

        if has_overlap([(e.start_char, e.end_char, e.label_) for e in ents]):
            if overlap_mode == "reject":
                reject(row, "overlap", ", ".join(f"{e.label_}:{e.text}" for e in ents))
                continue
            ents = filter_spans(ents)

        doc.ents = ents
        examples.append({"text": text, "hash": text_hash(text), "doc": doc, "n_ents": len(ents)})

    return examples, rejected


def dedupe(examples):
    """One example per normalized text; keep the version with the most entities."""
    best = {}
    for ex in examples:
        cur = best.get(ex["hash"])
        if cur is None or ex["n_ents"] > cur["n_ents"]:
            best[ex["hash"]] = ex
    return list(best.values())


# --------------------------------------------------------------------------- #
# Splitting / writing
# --------------------------------------------------------------------------- #
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_docbin(docs, path):
    db = DocBin(attrs=["ENT_IOB", "ENT_TYPE"], store_user_data=False)
    for d in docs:
        db.add(d)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    db.to_disk(path)


def label_counts(examples):
    c = Counter()
    for ex in examples:
        for ent in ex["doc"].ents:
            c[ent.label_] += 1
    return c


def report(name, examples):
    c = label_counts(examples)
    empties = sum(1 for e in examples if e["n_ents"] == 0)
    counts = ", ".join(f"{k}={v}" for k, v in sorted(c.items())) or "-"
    print(f"  {name:<6} {len(examples):>7} docs ({empties} with no entities) | {counts}")


def main():
    args = parse_args()
    label_cols = parse_label_cols(args.label_col)
    out = Path(args.out_dir)
    base_dir, test_dir = out / "base", out / "test"
    test_path = test_dir / "test.spacy"
    hashes_path = test_dir / "test_hashes.txt"
    manifest_path = test_dir / "test_manifest.json"

    print(f"Labels: {label_cols}")
    df = load_inputs(args.input, args.text_col, label_cols)
    print(f"Loaded {len(df)} rows")

    df = normalize_frame(df, args.text_col, label_cols)
    blank = df[args.text_col] == ""
    if blank.any():
        print(f"Dropping {int(blank.sum())} rows with empty text")
        df = df[~blank]

    nlp = spacy.blank("en")
    examples, rejected = build_examples(df, args.text_col, label_cols, nlp, args.overlap)

    if args.drop_empty:
        before = len(examples)
        examples = [e for e in examples if e["n_ents"] > 0]
        print(f"Dropped {before - len(examples)} rows with no labelled fields (--drop-empty)")

    n_before = len(examples)
    examples = dedupe(examples)
    print(f"Deduplicated by text hash: {n_before} -> {len(examples)}")

    # ---- frozen test set -------------------------------------------------- #
    test_exists = test_path.exists() and hashes_path.exists() and not args.force_new_test
    if test_exists:
        frozen = set(hashes_path.read_text().split())
        if manifest_path.exists():
            recorded = json.loads(manifest_path.read_text()).get("sha256")
            if recorded and recorded != file_sha256(test_path):
                sys.exit("test.spacy does not match test_manifest.json - it was modified. Aborting.")
        leaked = [e for e in examples if e["hash"] in frozen]
        examples = [e for e in examples if e["hash"] not in frozen]
        print(f"Existing frozen test set found ({len(frozen)} docs); excluded {len(leaked)} matching new rows")
        test_examples = None
        pool = examples
    else:
        if args.force_new_test and test_path.exists():
            print("WARNING: regenerating the test set; old model scores are no longer comparable")
        pool, test_examples = train_test_split(examples, test_size=args.test_frac, random_state=args.seed)

    dev_size = args.dev_frac / (1 - args.test_frac) if test_examples is not None else args.dev_frac
    train_examples, dev_examples = train_test_split(pool, test_size=dev_size, random_state=args.seed)

    # ---- write ------------------------------------------------------------ #
    write_docbin([e["doc"] for e in train_examples], base_dir / "train.spacy")
    write_docbin([e["doc"] for e in dev_examples], base_dir / "dev.spacy")

    if test_examples is not None:
        write_docbin([e["doc"] for e in test_examples], test_path)
        hashes_path.write_text("\n".join(e["hash"] for e in test_examples) + "\n")
        manifest_path.write_text(
            json.dumps(
                {
                    "n_docs": len(test_examples),
                    "sha256": file_sha256(test_path),
                    "created": datetime.now(timezone.utc).isoformat(),
                    "seed": args.seed,
                    "labels": sorted(set(label_cols.values())),
                },
                indent=2,
            )
        )
        print(f"Created NEW frozen test set at {test_path}")

    base_dir.mkdir(parents=True, exist_ok=True)
    rej_path = base_dir / "rejected.csv"
    pd.DataFrame(rejected, columns=["source", "row", "text", "reason", "detail"]).to_csv(rej_path, index=False)

    # ---- report ----------------------------------------------------------- #
    print("\nSummary")
    report("train", train_examples)
    report("dev", dev_examples)
    if test_examples is not None:
        report("test", test_examples)
    else:
        print(f"  test   (unchanged, frozen at {test_path})")

    total = len(df)
    print(f"\nRejected {len(rejected)} of {total} rows ({100 * len(rejected) / max(total, 1):.1f}%) -> {rej_path}")
    if rejected:
        for reason, n in Counter(r["reason"] for r in rejected).most_common():
            print(f"  {reason:<28} {n}")


if __name__ == "__main__":
    main()
