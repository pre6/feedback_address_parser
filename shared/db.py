"""Postgres helpers shared by the API and the batch job."""
import os


def database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("Set DATABASE_URL, e.g. postgresql://user:pass@localhost:5432/dbname")
    return url


# source_id is TEXT so this works whatever type your source table's id column is.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS address_parsed (
    source_id     TEXT PRIMARY KEY,
    location      TEXT NOT NULL,
    address       TEXT,
    city          TEXT,
    zip           TEXT,
    status        TEXT NOT NULL CHECK (status IN ('auto', 'needs_review', 'reviewed')),
    min_conf      REAL,
    flags         TEXT[] NOT NULL DEFAULT '{}',
    entity_conf   JSONB,
    model_version TEXT,
    parsed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewed_by   TEXT,
    reviewed_at   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS address_parsed_queue_idx
    ON address_parsed (min_conf NULLS FIRST) WHERE status = 'needs_review';
"""


def ensure_schema(conn):
    conn.execute(SCHEMA_SQL)
