#!/usr/bin/env python
"""Parse addresses from a Postgres table, flag uncertain ones, write results.

    set DATABASE_URL=postgresql://user:pass@localhost:5432/mydb      (Windows cmd)

    # look first, write nothing:
    python api/batch_parse.py --source-table addresses --id-col id --text-col location --dry-run --limit 500

    # for real:
    python api/batch_parse.py --source-table addresses --id-col id --text-col location

    # after promoting a new model, re-parse everything except human-reviewed rows:
    python api/batch_parse.py --source-table addresses --id-col id --text-col location --reparse

Results go to the address_parsed table (created automatically):
  status = 'auto'          confident, no flags
  status = 'needs_review'  flagged, shows up in the analyst queue
  status = 'reviewed'      fixed/confirmed by a human (never overwritten here)
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import psycopg
import spacy
from psycopg import sql
from psycopg.types.json import Jsonb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.confidence import analyze, beam_scores  # noqa: E402
from shared.db import database_url, ensure_schema  # noqa: E402
from shared.text import normalize_text  # noqa: E402

UPSERT = """
INSERT INTO address_parsed
  (source_id, location, address, city, zip, status, min_conf, flags, entity_conf, model_version, parsed_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s::text[], %s, %s, now())
ON CONFLICT (source_id) DO UPDATE SET
  location = EXCLUDED.location, address = EXCLUDED.address, city = EXCLUDED.city, zip = EXCLUDED.zip,
  status = EXCLUDED.status, min_conf = EXCLUDED.min_conf, flags = EXCLUDED.flags,
  entity_conf = EXCLUDED.entity_conf, model_version = EXCLUDED.model_version, parsed_at = now()
WHERE address_parsed.status <> 'reviewed'
"""


def resolve_model(args):
    """--model wins; otherwise use whatever models/registry.json says is current."""
    if args.model:
        p = Path(args.model).resolve()
        return p, (p.parent.name if p.name == "model-best" else p.name)
    models_dir = (ROOT / args.models_dir).resolve()
    reg_path = models_dir / "registry.json"
    if not reg_path.exists():
        sys.exit(f"No {reg_path}. Run promote.py once, or pass --model models/v1/model-best")
    rel = json.loads(reg_path.read_text())["current"]
    return models_dir / rel, rel.split("/")[0]


def process_chunk(nlp, rows, args, required, use_beam, version):
    """rows: [(source_id, location)] -> list of tuples ready for UPSERT."""
    texts = [normalize_text(loc or "") for _, loc in rows]
    idx = [i for i, t in enumerate(texts) if t]
    docs = list(nlp.pipe([texts[i] for i in idx]))
    scores = beam_scores(nlp, docs) if (use_beam and docs) else [None] * len(docs)
    by_idx = {i: analyze(d, s, args.threshold, required, args.zip_regex) for i, d, s in zip(idx, docs, scores)}

    out = []
    for i, (source_id, location) in enumerate(rows):
        r = by_idx.get(i) or {
            "address": None, "city": None, "zip": None, "min_conf": 0.0,
            "entity_conf": {}, "flags": ["empty_text"], "status": "needs_review",
        }
        out.append((source_id, location or "", r["address"], r["city"], r["zip"], r["status"],
                    r["min_conf"], r["flags"], Jsonb(r["entity_conf"]), version))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-table", required=True, help="e.g. addresses or public.addresses")
    ap.add_argument("--id-col", required=True)
    ap.add_argument("--text-col", required=True)
    ap.add_argument("--threshold", type=float, default=0.90, help="entities below this confidence get flagged")
    ap.add_argument("--required", default="ADDRESS,CITY,ZIP", help="labels that must be present, else flagged")
    ap.add_argument("--zip-regex", default=None, help=r'optional, e.g. "\d{5}(-\d{4})?" for US ZIPs')
    ap.add_argument("--rules-only", action="store_true", help="skip beam confidence, use only rule checks")
    ap.add_argument("--reparse", action="store_true", help="re-parse already-parsed rows (except reviewed ones)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--dry-run", action="store_true", help="parse and report, write nothing")
    ap.add_argument("--model", default="", help="model dir; default = current model in models/registry.json")
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    model_path, version = resolve_model(args)
    print(f"Model: {model_path} (version {version})")
    nlp = spacy.load(model_path)
    required = {x.strip().upper() for x in args.required.split(",") if x.strip()}
    use_beam = not args.rules_only

    if use_beam:  # fail loudly now rather than silently producing no confidence
        try:
            d = nlp("123 Main St, Springfield, IL 62704")
            beam_scores(nlp, [d])
        except Exception as e:
            sys.exit(f"Beam scoring failed on this model: {e!r}\nRe-run with --rules-only to use rule checks only.")

    cond = sql.SQL("p.status IS DISTINCT FROM 'reviewed'") if args.reparse else sql.SQL("p.source_id IS NULL")
    query = sql.SQL(
        "SELECT s.{id}::text AS source_id, s.{text} AS location FROM {table} s "
        "LEFT JOIN address_parsed p ON p.source_id = s.{id}::text "
        "WHERE s.{text} IS NOT NULL AND {cond}"
    ).format(
        id=sql.Identifier(args.id_col),
        text=sql.Identifier(args.text_col),
        table=sql.Identifier(*args.source_table.split(".")),
        cond=cond,
    )
    if args.limit:
        query += sql.SQL(" LIMIT {}").format(sql.Literal(args.limit))

    url = database_url()
    statuses, reasons, confs, samples = Counter(), Counter(), [], []
    total = 0
    with psycopg.connect(url) as rconn, psycopg.connect(url) as wconn:
        ensure_schema(wconn)
        wconn.commit()
        with rconn.cursor(name="src_rows") as cur:
            cur.execute(query)
            while True:
                rows = cur.fetchmany(args.chunk)
                if not rows:
                    break
                out = process_chunk(nlp, rows, args, required, use_beam, version)
                if not args.dry_run:
                    with wconn.cursor() as wcur:
                        wcur.executemany(UPSERT, out)
                    wconn.commit()
                for t in out:
                    statuses[t[5]] += 1
                    reasons.update(f.split(":")[0] + (":" + f.split(":")[1] if ":" in f else "") for f in t[7])
                    if t[6] is not None:
                        confs.append(t[6])
                    if t[5] == "needs_review" and len(samples) < 10:
                        samples.append(t)
                total += len(out)
                print(f"  processed {total}", end="\r", flush=True)

    print(f"\n\nParsed {total} rows" + (" (DRY RUN, nothing written)" if args.dry_run else ""))
    if not total:
        return
    for s, n in statuses.most_common():
        print(f"  {s:<14}{n:>8}  ({100 * n / total:.1f}%)")
    print("\nFlag reasons:")
    for r, n in reasons.most_common():
        print(f"  {r:<22}{n:>8}")
    if confs:
        confs.sort()
        q = lambda p: confs[min(int(p * len(confs)), len(confs) - 1)]
        print(f"\nmin_conf percentiles: p5={q(.05):.3f}  p25={q(.25):.3f}  p50={q(.5):.3f}  p75={q(.75):.3f}")
    if samples:
        print("\nSample flagged rows:")
        for t in samples:
            print(f"  {t[1]!r}\n     -> addr={t[2]!r} city={t[3]!r} zip={t[4]!r} conf={t[6]} flags={t[7]}")


if __name__ == "__main__":
    main()
