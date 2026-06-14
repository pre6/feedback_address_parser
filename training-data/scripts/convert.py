"""
convert.py

Reads labeled address data from a Postgres table and converts it to
spaCy .spacy (DocBin) format, split into train / dev / test sets.

EXPECTED TABLE SCHEMA
---------------------
Each training table must have at least these columns:

    id          SERIAL PRIMARY KEY
    text        TEXT          -- the raw address string
    address     TEXT          -- the labeled ADDRESS span text (or NULL)
    city        TEXT          -- the labeled CITY span text (or NULL)

Example row:
    id=1, text="742 Evergreen Terrace, Springfield", address="742 Evergreen Terrace", city="Springfield"

USAGE
-----
# Convert a specific table:
    python convert.py --table training_batch_001

# Convert and point to a different DB or output dir:
    python convert.py --table training_batch_001 \
        --db-url postgresql://user:pass@localhost:5432/mydb \
        --output-dir ./training-data/corpus

# Override the default 80/10/10 split:
    python convert.py --table training_batch_001 --train 0.75 --dev 0.15 --test 0.10

ENVIRONMENT VARIABLES
---------------------
If --db-url is not passed, the script reads DATABASE_URL from the environment
or from a .env file in the same directory (via python-dotenv if installed).

OUTPUT
------
Writes three files to --output-dir (default: ./training-data/corpus):
    train.spacy   -- 80% of data, used for gradient updates
    dev.spacy     -- 10% of data, used for epoch-level evaluation + early stopping
    test.spacy    -- 10% of data, locked away; only used for final honest evaluation

Also updates training-data/meta.json with counts and timestamp.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import spacy
from spacy.tokens import Doc, DocBin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_span(text: str, fragment: str) -> tuple[int, int] | None:
    """
    Find the character start/end of `fragment` inside `text`.
    Returns (start, end) or None if not found.

    Uses a case-sensitive search first, then falls back to lowercased.
    This handles minor capitalisation differences between the raw text
    and the labeled span.
    """
    if not fragment:
        return None

    idx = text.find(fragment)
    if idx != -1:
        return idx, idx + len(fragment)

    # fallback: case-insensitive
    idx = text.lower().find(fragment.lower())
    if idx != -1:
        return idx, idx + len(fragment)

    return None


def row_to_doc(nlp, text: str, address: str | None, city: str | None) -> Doc | None:
    """
    Convert one database row into a spaCy Doc with .ents set.

    Returns None (and prints a warning) if:
    - the text is empty
    - a labeled span cannot be located in the text
    - two spans overlap (spaCy rejects overlapping entities)
    """
    text = (text or "").strip()
    if not text:
        return None

    doc = nlp.make_doc(text)
    spans = []

    for label, fragment in [("ADDRESS", address), ("CITY", city)]:
        if not fragment or not fragment.strip():
            continue
        fragment = fragment.strip()
        result = find_span(text, fragment)
        if result is None:
            print(
                f"  [WARN] Could not find {label!r} span {fragment!r} in: {text!r}",
                file=sys.stderr,
            )
            return None
        start_char, end_char = result
        span = doc.char_span(start_char, end_char, label=label)
        if span is None:
            # char_span returns None when the offsets don't align with token boundaries.
            # This usually means a tokenisation edge case — flag it rather than silently skip.
            print(
                f"  [WARN] Span alignment failed for {label!r} ({start_char}:{end_char}) in: {text!r}",
                file=sys.stderr,
            )
            return None
        spans.append(span)

    # Check for overlaps before assigning
    spans_sorted = sorted(spans, key=lambda s: s.start)
    for i in range(len(spans_sorted) - 1):
        if spans_sorted[i].end > spans_sorted[i + 1].start:
            print(
                f"  [WARN] Overlapping spans in: {text!r} — skipping row",
                file=sys.stderr,
            )
            return None

    doc.ents = spans
    return doc


def split_data(docs: list, train_ratio: float, dev_ratio: float):
    """
    Shuffle and split docs into (train, dev, test) lists.
    test_ratio is implicitly 1 - train_ratio - dev_ratio.
    """
    random.shuffle(docs)
    n = len(docs)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)
    return docs[:train_end], docs[train_end:dev_end], docs[dev_end:]


def write_docbin(docs: list, path: Path):
    db = DocBin(docs=docs)
    db.to_disk(path)
    print(f"  Wrote {len(docs)} docs → {path}")


def update_meta(meta_path: Path, table: str, train: int, dev: int, test: int):
    """
    Read existing meta.json (if present) and update it with the new counts.
    Creates meta.json if it doesn't exist yet.
    """
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    meta.setdefault("entity_labels", ["ADDRESS", "CITY"])
    meta["last_converted"] = datetime.now(timezone.utc).isoformat()
    meta["source_table"] = table
    meta["train_count"] = train
    meta["dev_count"] = dev
    meta["test_count"] = test

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Updated {meta_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert Postgres address data to spaCy DocBin")
    parser.add_argument(
        "--table", required=True,
        help="Name of the Postgres table to read from (e.g. training_batch_001)"
    )
    parser.add_argument(
        "--db-url", default=None,
        help="Postgres connection string. Defaults to DATABASE_URL env var."
    )
    parser.add_argument(
        "--output-dir", default="training-data/corpus",
        help="Directory to write train.spacy, dev.spacy, test.spacy (default: training-data/corpus)"
    )
    parser.add_argument("--train", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    parser.add_argument("--dev",   type=float, default=0.1, help="Dev split ratio (default: 0.1)")
    parser.add_argument("--test",  type=float, default=0.1, help="Test split ratio (default: 0.1)")
    parser.add_argument("--seed",  type=int,   default=42,  help="Random seed for reproducible splits")
    args = parser.parse_args()

    # Validate ratios
    total = round(args.train + args.dev + args.test, 6)
    if abs(total - 1.0) > 1e-4:
        print(f"[ERROR] train + dev + test must equal 1.0 (got {total})", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)

    # -----------------------------------------------------------------------
    # Resolve DB URL
    # -----------------------------------------------------------------------
    db_url = args.db_url or os.getenv("DATABASE_URL")
    if not db_url:
        # Try loading from .env in the script's directory
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).parent / ".env")
            db_url = os.getenv("DATABASE_URL")
        except ImportError:
            pass

    if not db_url:
        print(
            "[ERROR] No database URL found.\n"
            "  Pass --db-url or set DATABASE_URL in your environment or .env file.\n"
            "  Example: DATABASE_URL=postgresql://user:password@localhost:5432/mydb",
            file=sys.stderr,
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Fetch rows from Postgres
    # -----------------------------------------------------------------------
    print(f"\nConnecting to Postgres...")
    try:
        conn = psycopg2.connect(db_url)
    except psycopg2.OperationalError as e:
        print(f"[ERROR] Could not connect to database:\n  {e}", file=sys.stderr)
        sys.exit(1)

    cursor = conn.cursor()

    # Verify the table exists before running the full query
    cursor.execute(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
        (args.table,)
    )
    if not cursor.fetchone()[0]:
        print(f"[ERROR] Table '{args.table}' does not exist in the database.", file=sys.stderr)
        # Show available tables to help the user
        cursor.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        )
        tables = [r[0] for r in cursor.fetchall()]
        if tables:
            print(f"  Available tables: {', '.join(tables)}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading from table: {args.table}")
    # Use parameterized identifier via psycopg2's AsIs — table names can't be
    # passed as %s parameters so we validate the name first to prevent injection.
    if not args.table.replace("_", "").isalnum():
        print(f"[ERROR] Table name '{args.table}' contains invalid characters.", file=sys.stderr)
        sys.exit(1)

    cursor.execute(f'SELECT text, address, city FROM "{args.table}" WHERE text IS NOT NULL')
    rows = cursor.fetchall()
    conn.close()
    print(f"  Fetched {len(rows)} rows")

    if len(rows) == 0:
        print("[ERROR] Table is empty — nothing to convert.", file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Convert rows to spaCy Docs
    # -----------------------------------------------------------------------
    print("\nConverting rows to spaCy docs...")
    nlp = spacy.blank("en")
    docs = []
    skipped = 0

    for text, address, city in rows:
        doc = row_to_doc(nlp, text, address, city)
        if doc is None:
            skipped += 1
        else:
            docs.append(doc)

    print(f"  Converted: {len(docs)}  |  Skipped (warnings above): {skipped}")

    if len(docs) < 10:
        print(
            f"[ERROR] Only {len(docs)} usable docs — too few to split meaningfully. "
            "Fix the warnings above and re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Split and write
    # -----------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_docs, dev_docs, test_docs = split_data(docs, args.train, args.dev)

    print(f"\nSplit: {len(train_docs)} train / {len(dev_docs)} dev / {len(test_docs)} test")
    print(f"Writing to {output_dir}/\n")

    write_docbin(train_docs, output_dir / "train.spacy")
    write_docbin(dev_docs,   output_dir / "dev.spacy")
    write_docbin(test_docs,  output_dir / "test.spacy")

    # Update meta.json one directory up from corpus/
    meta_path = output_dir.parent / "meta.json"
    update_meta(meta_path, args.table, len(train_docs), len(dev_docs), len(test_docs))

    print(f"\nDone. Run `spacy train` using the files in {output_dir}/")


if __name__ == "__main__":
    main()