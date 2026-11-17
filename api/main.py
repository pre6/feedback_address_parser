"""Address parser API.

Run locally (from repo root):
    MODEL_PATH=models/v1/model-best CORRECTIONS_DIR=data/corrections \
        uvicorn api.main:app --reload

In Docker, mount ./models at /models and ./data at /data, and point
MODEL_PATH at /models/current (a symlink or copy of the promoted model).
"""
import csv
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.review import router as review_router

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.text import normalize_text  # noqa: E402  (same normalization as training)

MODEL_PATH = os.getenv("MODEL_PATH", "/models/current")
CORRECTIONS_DIR = Path(os.getenv("CORRECTIONS_DIR", "/data/corrections"))
CORRECTIONS_FILE = CORRECTIONS_DIR / "corrections.csv"
COLUMNS = ["LOCATION", "ADDRESS", "CITY", "ZIP", "model_version", "timestamp"]

app = FastAPI(title="Address Parser")
_write_lock = threading.Lock()
nlp = None
model_version = "unknown"
app.include_router(review_router)


def _version_from_path(path: str) -> str:
    """models/v3/model-best -> v3 ; models/current -> whatever it resolves to."""
    resolved = Path(path).resolve()
    return resolved.parent.name if resolved.name == "model-best" else resolved.name


@app.on_event("startup")
def load_model():
    global nlp, model_version
    nlp = spacy.load(MODEL_PATH)
    model_version = _version_from_path(MODEL_PATH)


class ParseRequest(BaseModel):
    text: str


class Entity(BaseModel):
    label: str
    text: str
    start: int
    end: int


class ParseResponse(BaseModel):
    text: str  # the normalized text the offsets refer to
    entities: list[Entity]
    model_version: str


class CorrectionRequest(BaseModel):
    text: str
    address: str = ""
    city: str = ""
    zip: str = ""


@app.get("/health")
def health():
    return {"status": "ok" if nlp is not None else "loading", "model_version": model_version}


@app.post("/parse", response_model=ParseResponse)
def parse(req: ParseRequest):
    text = normalize_text(req.text)
    if not text:
        raise HTTPException(400, "text is empty")
    doc = nlp(text)
    return ParseResponse(
        text=text,
        entities=[Entity(label=e.label_, text=e.text, start=e.start_char, end=e.end_char) for e in doc.ents],
        model_version=model_version,
    )


@app.post("/corrections")
def add_correction(req: CorrectionRequest):
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    # Same columns as data/raw/ so build_dataset.py can read it unchanged.
    # LOCATION is stored raw; build_dataset normalizes it.
    row = {
        "LOCATION": req.text,
        "ADDRESS": req.address,
        "CITY": req.city,
        "ZIP": req.zip,
        "model_version": model_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        is_new = not CORRECTIONS_FILE.exists()
        with open(CORRECTIONS_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            if is_new:
                w.writeheader()
            w.writerow(row)
    return {"saved": True, "file": str(CORRECTIONS_FILE)}
