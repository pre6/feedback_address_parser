"""
trainer/train.py

Entry point for the address parser training pipeline.

Steps:
  1. Load and validate meta.json (entity labels, corpus counts)
  2. Merge base train.spacy + all corrections/*.spacy into a fresh train.spacy
  3. Run spacy train against dev.spacy for early stopping
  4. Evaluate the trained model against the locked test.spacy
  5. If F1 >= threshold (and >= current model), write versioned model and update symlink
  6. Update meta.json with new version metadata

Environment variables (from .env / docker-compose):
  TRAINING_DATA_DIR   path to training-data/ mount   (default: /training-data)
  MODELS_DIR          path to models/ volume          (default: /models)
  MIN_F1              minimum F1 to accept new model  (default: 0.80)
  SPACY_CONFIG        path to config.cfg              (default: /app/config.cfg)
"""

import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from spacy.training import Example

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (all overridable via environment)
# ---------------------------------------------------------------------------

TRAINING_DATA_DIR = Path(os.getenv("TRAINING_DATA_DIR", "/training-data"))
MODELS_DIR        = Path(os.getenv("MODELS_DIR",        "/models"))
SPACY_CONFIG      = Path(os.getenv("SPACY_CONFIG",      "/app/config.cfg"))
MIN_F1            = float(os.getenv("MIN_F1",           "0.80"))

CORPUS_DIR      = TRAINING_DATA_DIR / "corpus"
CORRECTIONS_DIR = TRAINING_DATA_DIR / "corrections"
META_JSON       = TRAINING_DATA_DIR / "meta.json"

TRAIN_SPACY = CORPUS_DIR / "train.spacy"
DEV_SPACY   = CORPUS_DIR / "dev.spacy"
TEST_SPACY  = CORPUS_DIR / "test.spacy"
MERGED_SPACY = CORPUS_DIR / "train_merged.spacy"

CURRENT_SYMLINK = MODELS_DIR / "current"


# ---------------------------------------------------------------------------
# 1. Meta validation
# ---------------------------------------------------------------------------

def load_meta() -> dict:
    if not META_JSON.exists():
        log.error("meta.json not found at %s", META_JSON)
        sys.exit(1)
    with META_JSON.open() as f:
        meta = json.load(f)
    required = {"entity_labels", "spacy_version"}
    missing = required - meta.keys()
    if missing:
        log.error("meta.json is missing keys: %s", missing)
        sys.exit(1)
    log.info("Entity labels: %s", meta["entity_labels"])
    return meta


def validate_config_labels(meta: dict) -> None:
    """Warn if config.cfg NER labels differ from meta.json."""
    try:
        cfg_text = SPACY_CONFIG.read_text()
    except FileNotFoundError:
        log.warning("config.cfg not found at %s — skipping label validation", SPACY_CONFIG)
        return
    for label in meta["entity_labels"]:
        if label not in cfg_text:
            log.warning("Label '%s' from meta.json not found in config.cfg", label)


# ---------------------------------------------------------------------------
# 2. Corpus merge
# ---------------------------------------------------------------------------

def load_docbin(path: Path, vocab) -> list:
    """Load a .spacy file and return a list of Doc objects."""
    db = DocBin().from_disk(path)
    return list(db.get_docs(vocab))


def merge_corpus(nlp) -> int:
    """
    Merge train.spacy + all corrections/*.spacy into train_merged.spacy.
    Deduplicates by exact text. Returns the number of training examples.
    """
    log.info("Merging corpus...")

    seen_texts: set[str] = set()
    merged_db  = DocBin()

    def add_docs(docs: list, source: str) -> int:
        added = 0
        for doc in docs:
            if doc.text in seen_texts:
                continue
            seen_texts.add(doc.text)
            merged_db.add(doc)
            added += 1
        log.info("  %s → %d new docs (%d duplicates skipped)", source, added, len(docs) - added)
        return added

    # Base training corpus
    if not TRAIN_SPACY.exists():
        log.error("Base train.spacy not found at %s", TRAIN_SPACY)
        sys.exit(1)
    base_docs = load_docbin(TRAIN_SPACY, nlp.vocab)
    add_docs(base_docs, "train.spacy")

    # Corrections — sorted for reproducibility
    correction_files = sorted(CORRECTIONS_DIR.glob("*.spacy"))
    if not correction_files:
        log.info("No correction files found in %s", CORRECTIONS_DIR)
    for cf in correction_files:
        if cf.name == "merged.spacy":
            continue  # skip any stale merged file
        docs = load_docbin(cf, nlp.vocab)
        add_docs(docs, cf.name)

    merged_db.to_disk(MERGED_SPACY)
    total = len(seen_texts)
    log.info("Merged corpus written: %d unique examples → %s", total, MERGED_SPACY)
    return total


# ---------------------------------------------------------------------------
# 3. spaCy train
# ---------------------------------------------------------------------------

def next_version() -> str:
    """Determine the next version directory name (v1, v2, v3, ...)."""
    existing = sorted(
        [d for d in MODELS_DIR.iterdir() if d.is_dir() and d.name.startswith("v")],
        key=lambda d: int(d.name[1:]) if d.name[1:].isdigit() else 0,
    )
    if not existing:
        return "v1"
    last_num = int(existing[-1].name[1:])
    return f"v{last_num + 1}"


def run_training(output_dir: Path) -> None:
    """
    Call `spacy train` as a subprocess so it streams live output.
    Using subprocess rather than the Python API gives us real-time
    epoch logs in docker compose logs.
    """
    log.info("Starting spaCy training → %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "spacy", "train",
        str(SPACY_CONFIG),
        "--output",    str(output_dir),
        "--paths.train", str(MERGED_SPACY),
        "--paths.dev",   str(DEV_SPACY),
    ]

    log.info("Command: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        log.error("spacy train exited with code %d", result.returncode)
        shutil.rmtree(output_dir, ignore_errors=True)
        sys.exit(1)

    # spacy train writes model-best/ and model-last/ inside output_dir
    best_path = output_dir / "model-best"
    if not best_path.exists():
        log.error("model-best not found in %s after training", output_dir)
        shutil.rmtree(output_dir, ignore_errors=True)
        sys.exit(1)

    log.info("Training complete. Best model at %s", best_path)


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model_path: Path, meta: dict) -> dict:
    """
    Load the trained model and evaluate on test.spacy.
    Returns a dict of per-entity and overall F1 scores.
    """
    log.info("Evaluating on test set: %s", TEST_SPACY)

    if not TEST_SPACY.exists():
        log.error("test.spacy not found at %s — cannot evaluate", TEST_SPACY)
        sys.exit(1)

    nlp = spacy.load(model_path / "model-best")
    test_docs = load_docbin(TEST_SPACY, nlp.vocab)

    examples = []
    for gold_doc in test_docs:
        pred_doc = nlp(gold_doc.text)
        examples.append(Example(pred_doc, gold_doc))

    scores = nlp.evaluate(examples)

    ents_f  = scores.get("ents_f",  0.0)
    ents_p  = scores.get("ents_p",  0.0)
    ents_r  = scores.get("ents_r",  0.0)
    per_ent = scores.get("ents_per_type", {})

    log.info("Overall   — F1: %.4f  P: %.4f  R: %.4f", ents_f, ents_p, ents_r)
    for label in meta["entity_labels"]:
        label_scores = per_ent.get(label, {})
        log.info(
            "  %-14s  F1: %.4f  P: %.4f  R: %.4f",
            label,
            label_scores.get("f", 0.0),
            label_scores.get("p", 0.0),
            label_scores.get("r", 0.0),
        )

    return {"overall_f1": ents_f, "overall_p": ents_p, "overall_r": ents_r, "per_entity": per_ent}


def get_current_f1() -> float:
    """Return the F1 of the currently deployed model, or 0 if none exists."""
    if not CURRENT_SYMLINK.exists():
        return 0.0
    score_file = CURRENT_SYMLINK / "eval_scores.json"
    if not score_file.exists():
        return 0.0
    with score_file.open() as f:
        data = json.load(f)
    return data.get("overall_f1", 0.0)


# ---------------------------------------------------------------------------
# 5. Promote
# ---------------------------------------------------------------------------

def promote_model(version: str, output_dir: Path, scores: dict, meta: dict, train_count: int) -> None:
    """
    Write eval_scores.json into the versioned model dir and
    update the `current` symlink to point to it.
    """
    version_dir = MODELS_DIR / version

    # Write evaluation scores alongside the model
    score_path = version_dir / "eval_scores.json"
    score_path.write_text(json.dumps(scores, indent=2))

    # Update (or create) the `current` symlink atomically
    tmp_link = MODELS_DIR / "_current_tmp"
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(version_dir)
    tmp_link.rename(CURRENT_SYMLINK)

    log.info("Promoted %s as current model (F1: %.4f)", version, scores["overall_f1"])


# ---------------------------------------------------------------------------
# 6. Update meta.json
# ---------------------------------------------------------------------------

def update_meta(meta: dict, version: str, scores: dict, train_count: int) -> None:
    meta["current_version"]  = version
    meta["last_trained"]     = datetime.now(timezone.utc).isoformat()
    meta["train_count"]      = train_count
    meta["last_f1"]          = round(scores["overall_f1"], 4)
    meta["correction_batches"] = sorted(
        [f.name for f in CORRECTIONS_DIR.glob("*.spacy") if f.name != "merged.spacy"]
    )
    META_JSON.write_text(json.dumps(meta, indent=2))
    log.info("meta.json updated → version=%s  F1=%.4f", version, scores["overall_f1"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info("Address parser training pipeline starting")
    log.info("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Validate meta and config
    meta = load_meta()
    validate_config_labels(meta)

    # Blank NLP for vocab (no pipeline needed just for DocBin loading)
    nlp = spacy.blank("en")

    # 2. Merge corpus
    train_count = merge_corpus(nlp)

    # 3. Determine version and train
    version    = next_version()
    output_dir = MODELS_DIR / version
    run_training(output_dir)

    # 4. Evaluate
    scores = evaluate_model(output_dir, meta)

    # 5. Decide whether to promote
    current_f1 = get_current_f1()
    new_f1     = scores["overall_f1"]

    if new_f1 < MIN_F1:
        log.warning(
            "New model F1 %.4f is below MIN_F1 %.4f — NOT promoting. "
            "Check corrections for annotation errors.",
            new_f1, MIN_F1,
        )
        shutil.rmtree(output_dir, ignore_errors=True)
        sys.exit(2)

    if new_f1 < current_f1:
        log.warning(
            "New model F1 %.4f is lower than current model F1 %.4f — NOT promoting. "
            "Retrain with more corrections or review recent annotation batch.",
            new_f1, current_f1,
        )
        shutil.rmtree(output_dir, ignore_errors=True)
        sys.exit(2)

    # 6. Promote and update meta
    promote_model(version, output_dir, scores, meta, train_count)
    update_meta(meta, version, scores, train_count)

    log.info("=" * 60)
    log.info("Pipeline complete. Active model: %s  F1: %.4f", version, new_f1)
    log.info("=" * 60)


if __name__ == "__main__":
    main()