# Feedback Address Parser

An address parser with a human-in-the-loop retraining cycle. A spaCy NER model extracts **ADDRESS**, **CITY** and **ZIP** from free-text location strings. Users correct mistakes in a web UI, the corrections are folded into the training data, and a new model is promoted only if it beats the current one on a frozen test set.

```
 parse  ->  review / correct  ->  corrections.csv  ->  retrain  ->  evaluate  ->  promote  ->  reload API
   ^                                                                                              |
   +----------------------------------------------------------------------------------------------+
```

## Components

| Folder | What it holds |
|---|---|
| `api/` | FastAPI service: parsing endpoint, `/health`, model loading, `batch_parse.py` for scoring a database table |
| `ui/` | Streamlit app: Parse page, Review queue, "Submit correction" |
| `trainer/` | Dataset build, training, evaluation, promotion, and the one-command retrain loop |
| `shared/` | Code used by both the API and the trainer (`text.py` normalization, `spans.py` span matching) |
| `models/` | Trained models (`v1/`, `v2/`, ...) and `registry.json` |
| `data/` | `raw/` (labelled CSVs), `corrections/` (user feedback), `base/` (train/dev), `test/` (frozen test set) |
| `raw_data/` | TODO: describe what lives here |

## Quick start (Docker)

Requires Docker Desktop and a `.env` file in the repo root with `DATABASE_URL` set.

```
docker compose build
docker compose up
```

- API health check: http://localhost:8000/health (returns `{"status":"ok","model_version":"..."}`)
- UI: http://localhost:8501

Stop with `Ctrl+C`, then `docker compose down`.

> Do not run plain `docker compose config`; it prints your database password. Use `docker compose config --quiet` to validate the file.

### Database

On startup the API creates one table, `address_parsed`, in the database named by `DATABASE_URL`. It never modifies your source tables.

To parse an existing table in bulk and fill the review queue:

```
docker compose run --rm api python api/batch_parse.py --source-table YOURTABLE --id-col YOURIDCOLUMN --text-col YOURTEXTCOLUMN --dry-run --limit 500
```

Drop `--dry-run` to write results. Start with `--limit 500` so the review queue stays small.

## The feedback loop

1. **Parse** an address in the UI.
2. **Correct** any wrong field and click *Submit correction*. This appends a row to `data/corrections/corrections.csv`.
3. **Retrain** (see below). Corrections are picked up automatically.
4. If the candidate model is **promoted**, `models/registry.json` is updated and the API is told to reload.

### Correction file format

`data/corrections/*.csv` must have these columns, because the retrain script uses the defaults:

```
LOCATION, ADDRESS, CITY, ZIP
```

Each labelled field must appear verbatim (after normalization) inside `LOCATION`. Rows where it doesn't are written to `data/base/rejected.csv` with a reason instead of being trained on.

## Retraining

Run from the repo root, in the Python environment that has spaCy, pandas and scikit-learn installed (not inside Docker; `data/` and `models/` are shared with the containers through the filesystem).

```
python trainer/retrain.py
```

This runs four stages:

| Stage | Script | What it does |
|---|---|---|
| 1. Build | `build_dataset.py` | Reads `data/raw/*.csv` and `data/corrections/*.csv`, normalizes, labels, dedupes, splits into `data/base/train.spacy` and `dev.spacy` |
| 2. Train | `train.py` | Trains a new model into the next `models/vN/` folder using `trainer/config.cfg` |
| 3. Evaluate | `evaluate.py` | Scores the candidate on the dev set and prints mismatches |
| 4. Promote | `promote.py` | Compares candidate and current on the frozen test set and updates the registry if the candidate passes |

If the candidate is promoted and the API is reachable, `retrain.py` calls `POST /reload` on it.

### Useful flags

```
python trainer/retrain.py --skip-build          # reuse existing train/dev files
python trainer/retrain.py --dry-run             # run everything, leave registry.json alone
python trainer/retrain.py --subset 2000 -- --training.max_steps 500   # quick plumbing test
python trainer/retrain.py --raw-glob "data/raw_sample/*.csv"          # build from a smaller sample
```

Anything after `--` is passed to `spacy train` as a config override.

### Resource notes

- `--subset` only limits **training**. The build stage always processes every input row.
- The build holds every document in memory. With ~900k input rows expect several GB of RAM and no progress output until it finishes. Close other applications first.
- Training runs on CPU by default (`--gpu-id -1`), so a full run takes a long time.
- The promote stage scores two models on the whole test set and is slow even on a quick run.

## Model registry and promotion

`models/registry.json` records which model is live and the history of every promotion decision:

```json
{
  "current": "v1/model-best",
  "history": [{"timestamp": "...", "candidate": "...", "previous": "...",
               "decision": "promoted", "reasons": ["..."],
               "candidate_scores": {}, "current_scores": {}}]
}
```

Paths are relative to `models/`, so the file works on Windows, Linux and inside Docker. It is written atomically so the API never reads a half-written file.

A candidate is promoted only if **both** hold:

1. Overall F1 is at least the current model's F1 plus `--min-gain` (default `0.0`).
2. No label's F1 drops by more than `--max-label-drop` (default `0.01`).

If there is no current model, the candidate is promoted unconditionally. A rejection exits with code 1 and is a normal outcome, not an error.

Promote a specific model manually:

```
python trainer/promote.py --candidate models/v2/model-best --dry-run   # report only
python trainer/promote.py --candidate models/v2/model-best --force     # skip the gate
python trainer/promote.py --candidate models/v2/model-best --min-gain -0.05 --max-label-drop 0.05
```

Evaluate any model on any `.spacy` file:

```
python trainer/evaluate.py --model models/v1/model-best --data data/base/dev.spacy
```

### Rolling back

Edit `"current"` in `models/registry.json` (or restore a backup copy), then reload the API:

```
copy /y models\registry.json.bak models\registry.json
curl -X POST http://localhost:8000/reload
```

Back up before experimenting: `copy models\registry.json models\registry.json.bak`.

## Frozen test set

`data/test/test.spacy` is created the first time `build_dataset.py` runs and is **never regenerated**. On later builds, any row whose normalized text matches a test document is excluded from train and dev, so nothing leaks. `test_manifest.json` stores a SHA-256 of the file, and the build aborts if it has been modified.

Don't delete anything in `data/test/`. Regenerating it (`--force-new-test`) makes old and new model scores incomparable.

## Testing the loop without a long run

1. Back up the registry and `data/base/`.
2. Build from a small sample so the build takes a minute instead of many:
   ```
   python trainer/retrain.py --raw-glob "data/raw_sample/*.csv" --dry-run -- --training.max_steps 500
   ```
3. To exercise promotion and reload, loosen the gate for one run (see the `promote.py` flags above), then check `curl http://localhost:8000/health` for the new `model_version`.
4. Roll back using the backup.

A model trained on a tiny subset will normally be **rejected**. That confirms the gate works.

## Known limitations

- **Corrections can lose to older data.** `build_dataset.py` keeps one row per normalized text, preferring the row with more entities. On a tie, the first row loaded wins, and raw data is loaded before corrections. A correction for an address already in the raw data may therefore be ignored unless it labels more fields. Giving corrections priority in `dedupe()` would fix this.
- `retrain.py` does not pass column or label options to `build_dataset.py`, so it only supports `LOCATION`, `ADDRESS`, `CITY`, `ZIP`.
- The confidence logic used to flag rows for review is a placeholder and is known to be weak.

## Troubleshooting

| Symptom | What to do |
|---|---|
| `error during connect` / `dockerDesktopLinuxEngine` | Docker Desktop isn't running. Start it and wait. |
| API restarts repeatedly | `docker compose logs api`. A message about `registry.json` means `models/` or the registry isn't where Docker expects it. |
| `could not translate host name` | The host in `DATABASE_URL` is wrong or unreachable from inside Docker (use `host.docker.internal` for a database on your own machine). |
| UI can't reach the API | The API isn't healthy yet. Check `docker compose ps` and the API logs. |
| Build fails with "missing columns" | The corrections CSV headers don't match `LOCATION, ADDRESS, CITY, ZIP`. |
| Corrections not affecting the model | Check `data/base/rejected.csv` for your rows, and see the dedupe limitation above. |
| Build seems frozen | It prints nothing until it finishes. Check that `python.exe` memory is still growing. |

### Transparency - AI was used in the production of this repo
