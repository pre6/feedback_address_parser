#!/usr/bin/env python
"""Run the whole loop: build dataset -> train -> evaluate (dev) -> promote.

    python trainer/retrain.py                    # full run
    python trainer/retrain.py --subset 2000      # quick end-to-end check
    python trainer/retrain.py --skip-build       # reuse existing train/dev files
    python trainer/retrain.py --dry-run          # evaluate, but don't change the registry
    python trainer/retrain.py --subset 2000 -- --training.max_steps 500   # spaCy overrides

Inputs: every CSV in data/raw/ plus data/corrections/*.csv (if any).
If the candidate is promoted and --api-url is reachable, the API is told to reload.
"""
import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(title, cmd, allow_codes=(0,)):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}\n$ {' '.join(map(str, cmd))}\n", flush=True)
    code = subprocess.run(cmd, cwd=ROOT).returncode
    if code not in allow_codes:
        sys.exit(f"\nStep failed: {title} (exit code {code}). Stopping.")
    return code


def latest_version_dir(models_dir: Path) -> Path:
    nums = [int(p.name[1:]) for p in models_dir.glob("v*") if p.name[1:].isdigit() and p.is_dir()]
    if not nums:
        sys.exit(f"No vN folders found in {models_dir}")
    return models_dir / f"v{max(nums)}"


def reload_api(api_url):
    try:
        req = urllib.request.Request(f"{api_url}/reload", method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            print(f"API reloaded: {r.read().decode()}")
    except Exception as e:  # API not running is fine
        print(f"(Could not reload API at {api_url}: {e}. Restart it or call POST /reload yourself.)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-glob", default="data/raw/*.csv")
    ap.add_argument("--corrections-glob", default="data/corrections/*.csv")
    ap.add_argument("--subset", type=int, default=0, help="train on a random subset of N docs")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="don't modify registry.json")
    ap.add_argument("--api-url", default="http://127.0.0.1:8000", help="API to reload after promotion ('' to skip)")
    ap.add_argument("--models-dir", default="models")
    args, extra = ap.parse_known_args()
    extra = [e for e in extra if e != "--"]

    models_dir = ROOT / args.models_dir

    # 1. build
    if not args.skip_build:
        inputs = [args.raw_glob]
        if list(ROOT.glob(args.corrections_glob)):
            inputs.append(args.corrections_glob)
        else:
            print(f"No corrections found at {args.corrections_glob}; building from raw data only.")
        run("1/4 Build dataset", [PY, "trainer/build_dataset.py", "--input", *inputs])

    # 2. train
    train_cmd = [PY, "trainer/train.py"]
    if args.subset:
        train_cmd += ["--subset", str(args.subset)]
    if extra:
        train_cmd += ["--", *extra]
    run("2/4 Train", train_cmd)

    candidate = latest_version_dir(models_dir) / "model-best"
    rel = candidate.relative_to(ROOT).as_posix()

    # 3. evaluate on dev (the test set is only touched by promote.py)
    run("3/4 Evaluate on dev", [PY, "trainer/evaluate.py", "--model", rel, "--data", "data/base/dev.spacy"])

    # 4. promote (exit code 1 = rejected, which is a normal outcome, not a crash)
    promote_cmd = [PY, "trainer/promote.py", "--candidate", rel]
    if args.dry_run:
        promote_cmd.append("--dry-run")
    code = run("4/4 Promote (compare on frozen test set)", promote_cmd, allow_codes=(0, 1))

    print("\n" + "=" * 70)
    if args.dry_run:
        print(f"DRY RUN finished. Candidate: {rel}")
    elif code == 0:
        print(f"PROMOTED: {rel} is now the live model.")
        if args.api_url:
            reload_api(args.api_url)
    else:
        print(f"NOT PROMOTED: {rel} did not beat the current model. Live model unchanged.")


if __name__ == "__main__":
    main()
