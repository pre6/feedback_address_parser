#!/usr/bin/env python
"""Train a spaCy NER model on the output of build_dataset.py.

Examples (from the repo root):

    # 1) quick end-to-end check on a 50k subset
    python trainer/train.py --subset 50000

    # 2) full run
    python trainer/train.py

    # extra spaCy overrides go after --, e.g. limit steps
    python trainer/train.py --subset 20000 -- --training.max_steps 2000

Each run writes to models/v1, models/v2, ... (never overwrites). The best
checkpoint by dev F1 is in models/vN/model-best. This script only looks at the
DEV set. The frozen test set is only touched by evaluate.py.
"""
import argparse
import random
import subprocess
import sys
from pathlib import Path

import spacy
from spacy.tokens import DocBin

ROOT = Path(__file__).resolve().parents[1]


def next_version_dir(models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name[1:]) for p in models_dir.glob("v*") if p.name[1:].isdigit()]
    return models_dir / f"v{max(nums, default=0) + 1}"


def make_subset(src: Path, n: int, seed: int, dst: Path) -> Path:
    nlp = spacy.blank("en")
    docs = list(DocBin().from_disk(src).get_docs(nlp.vocab))
    if n >= len(docs):
        return src
    random.Random(seed).shuffle(docs)
    db = DocBin(attrs=["ENT_IOB", "ENT_TYPE"])
    for d in docs[:n]:
        db.add(d)
    dst.parent.mkdir(parents=True, exist_ok=True)
    db.to_disk(dst)
    print(f"Using random subset: {n} of {len(docs)} training docs")
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/base/train.spacy")
    ap.add_argument("--dev", default="data/base/dev.spacy")
    ap.add_argument("--config", default="trainer/config.cfg")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--subset", type=int, default=0, help="train on a random subset of N docs (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu-id", type=int, default=-1)
    ap.add_argument("--optimize", choices=["efficiency", "accuracy"], default="efficiency",
                    help="efficiency = CPU-friendly tok2vec. 'accuracy' uses a transformer and wants a GPU.")
    args, extra = ap.parse_known_args()
    extra = [e for e in extra if e != "--"]

    train_path, dev_path, cfg = ROOT / args.train, ROOT / args.dev, ROOT / args.config
    for p in (train_path, dev_path):
        if not p.exists():
            sys.exit(f"Missing {p}. Run trainer/build_dataset.py first.")

    if not cfg.exists():
        print(f"Creating {cfg}")
        subprocess.run(
            [sys.executable, "-m", "spacy", "init", "config", str(cfg),
             "--lang", "en", "--pipeline", "ner", "--optimize", args.optimize, "--force"],
            check=True,
        )

    out_dir = next_version_dir(ROOT / args.models_dir)
    if args.subset:
        train_path = make_subset(train_path, args.subset, args.seed, out_dir.parent / f"_{out_dir.name}_train_subset.spacy")

    cmd = [
        sys.executable, "-m", "spacy", "train", str(cfg),
        "--paths.train", str(train_path), "--paths.dev", str(dev_path),
        "--output", str(out_dir), "--gpu-id", str(args.gpu_id),
        "--system.seed", str(args.seed), *extra,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nDone. Best model: {out_dir / 'model-best'}")
    print(f"Next: python trainer/evaluate.py --model {out_dir / 'model-best'} --data data/base/dev.spacy")


if __name__ == "__main__":
    main()
