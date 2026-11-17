"""Review-queue endpoints: list parsed addresses, save many corrections at once.

Included from api/main.py with:
    from api.review import router as review_router
    app.include_router(review_router)
"""
import csv
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import psycopg
from fastapi import APIRouter, HTTPException, Query
from psycopg.rows import dict_row
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.db import database_url  # noqa: E402

router = APIRouter(prefix="/review", tags=["review"])

CORRECTIONS_DIR = Path(os.getenv("CORRECTIONS_DIR", "/data/corrections"))
CORRECTIONS_FILE = CORRECTIONS_DIR / "corrections.csv"
COLUMNS = ["LOCATION", "ADDRESS", "CITY", "ZIP", "model_version", "timestamp"]
_csv_lock = threading.Lock()


class Update(BaseModel):
    source_id: str
    address: str = ""
    city: str = ""
    zip: str = ""


class BatchUpdate(BaseModel):
    reviewer: str = "unknown"
    updates: list[Update]


@router.get("")
def list_rows(
    status: Literal["needs_review", "auto", "reviewed", "all"] = "needs_review",
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    where = "" if status == "all" else "WHERE status = %s"
    params = [] if status == "all" else [status]
    with psycopg.connect(database_url(), row_factory=dict_row) as conn:
        total = conn.execute(f"SELECT count(*) AS n FROM address_parsed {where}", params).fetchone()["n"]
        rows = conn.execute(
            "SELECT source_id, location, address, city, zip, status, min_conf, flags, model_version "
            f"FROM address_parsed {where} ORDER BY min_conf ASC NULLS FIRST, source_id LIMIT %s OFFSET %s",
            [*params, limit, offset],
        ).fetchall()
    return {"total": total, "rows": rows}


@router.post("/batch")
def save_batch(req: BatchUpdate):
    if not req.updates:
        raise HTTPException(400, "no updates")
    saved, not_found = [], []
    with psycopg.connect(database_url(), row_factory=dict_row) as conn:  # commits on clean exit
        for u in req.updates:
            row = conn.execute(
                "UPDATE address_parsed SET address = %s, city = %s, zip = %s, status = 'reviewed', "
                "reviewed_by = %s, reviewed_at = now() WHERE source_id = %s "
                "RETURNING location, model_version",
                [u.address.strip() or None, u.city.strip() or None, u.zip.strip() or None,
                 req.reviewer, u.source_id],
            ).fetchone()
            if row is None:
                not_found.append(u.source_id)
            else:
                saved.append((u, row))

    # Feed the training loop: same format as data/raw/ plus extra columns.
    csv_ok = True
    try:
        CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()
        with _csv_lock:
            is_new = not CORRECTIONS_FILE.exists()
            with open(CORRECTIONS_FILE, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=COLUMNS)
                if is_new:
                    w.writeheader()
                for u, row in saved:
                    w.writerow({
                        "LOCATION": row["location"], "ADDRESS": u.address.strip(), "CITY": u.city.strip(),
                        "ZIP": u.zip.strip(), "model_version": row["model_version"], "timestamp": now,
                    })
    except OSError as e:
        csv_ok = False
        print(f"WARNING: DB updated but could not write corrections CSV: {e}")

    return {"updated": len(saved), "not_found": not_found, "corrections_csv_written": csv_ok}
