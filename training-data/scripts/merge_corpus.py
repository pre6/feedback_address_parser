"""
merge_corpus.py

Assembles a clean, deduplicated training corpus from:
  - corpus/train.spacy  (base training set, converted from raw JSONL)
  - corrections/*.spacy (analyst corrections, append-only by batch)

Corrections always win over the base corpus for the same address string.
dev.spacy and test.spacy are never touched.

Run directly:
    python training-data/scripts/merge_corpus.py

Or import and call merge() from train.py before spacy train.
"""

import glob
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import spacy
from spacy.tokens import DocBin

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — relative to training-data/ directory
# ---------------------------------------------------------------------------

TRAINING_DATA_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = TRAINING_DATA_DIR / "corpus"
CORRECTIONS_DIR = TRAINING_DATA_DIR / "corrections"
META_PATH = TRAINING_DATA_DIR / "meta.json"

TRAIN_OUT = CORPUS_DIR / "train.spacy"


# ---------------------------------------------------------------------------
# Core merge logic
# ---------------------------------------------------------------------------

def load_docbin(path: Path, vocab) -> dict[int, object]:
    """Load a .spacy DocBin file and return {hash(doc.text): doc}."""
    if not path.exists():
        log.warning("File not found, skipping: %s", path)
        return {}

    db = DocBin().from_disk(path)
    docs = list(db.get_docs(vocab))
    log.info("  Loaded %d docs from %s", len(docs), path.name)
    return {hash(doc.text): doc for doc in docs}


def merge(
    corpus_dir: Path = CORPUS_DIR,
    corrections_dir: Path = CORRECTIONS_DIR,
    meta_path: Path = META_PATH,
    output_path: Path = TRAIN_OUT,
) -> int:
    """
    Merge base corpus and all correction batches into a single train.spacy.

    Returns the number of docs in the merged corpus.
    """
    nlp = spacy.blank("en")

    # ------------------------------------------------------------------
    # 1. Load base corpus — these are the lowest-priority annotations
    # ------------------------------------------------------------------
    log.info("Loading base corpus...")
    seen: dict[int, object] = load_docbin(corpus_dir / "train.spacy", nlp.vocab)
    base_count = len(seen)
    log.info("Base corpus: %d docs", base_count)

    # ------------------------------------------------------------------
    # 2. Load correction batches — sorted so order is deterministic
    #    Corrections overwrite base entries for the same address string
    # ------------------------------------------------------------------
    correction_files = sorted(corrections_dir.glob("*.spacy"))

    if not correction_files:
        log.info("No correction files found — using base corpus unchanged.")
    else:
        log.info("Loading %d correction batch(es)...", len(correction_files))
        new_from_corrections = 0
        overwritten = 0

        for path in correction_files:
            batch = load_docbin(path, nlp.vocab)
            for key, doc in batch.items():
                if key in seen:
                    overwritten += 1
                else:
                    new_from_corrections += 1
                seen[key] = doc  # corrections always win

        log.info(
            "Corrections applied: %d overwritten, %d new addresses added",
            overwritten,
            new_from_corrections,
        )

    total = len(seen)

    # ------------------------------------------------------------------
    # 3. Validate — warn if corpus shrank (sign of a bad corrections file)
    # ------------------------------------------------------------------
    if total < base_count:
        log.warning(
            "Merged corpus (%d) is smaller than base corpus (%d). "
            "Check correction files for empty or malformed DocBins.",
            total,
            base_count,
        )

    # ------------------------------------------------------------------
    # 4. Write merged corpus
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_db = DocBin(docs=list(seen.values()))
    merged_db.to_disk(output_path)
    log.info("Wrote merged corpus (%d docs) → %s", total, output_path)

    # ------------------------------------------------------------------
    # 5. Update meta.json
    # ------------------------------------------------------------------
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        log.warning("meta.json not found at %s — creating a new one.", meta_path)
        meta = {}

    meta["train_count"] = total
    meta["last_merged"] = datetime.now(timezone.utc).isoformat()
    meta["correction_batches"] = [f.name for f in correction_files]

    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Updated meta.json (train_count=%d, last_merged=%s)", total, meta["last_merged"])

    return total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("=== merge_corpus.py starting ===")
    log.info("Training data root: %s", TRAINING_DATA_DIR)

    try:
        total = merge()
        log.info("=== Done. Merged corpus contains %d docs. ===", total)
        sys.exit(0)
    except Exception as exc:
        log.error("Merge failed: %s", exc, exc_info=True)
        sys.exit(1)