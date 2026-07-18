"""Resolve the exact HuggingFace commit hash for each model in manifests/models.csv.

This turns the PIN_COMMIT_HASH placeholders into the specific revisions actually used,
so "BGE-base" becomes a fully reproducible pointer (repo + commit), addressing the reviewer
note that model families are not reproducible identifiers on their own.

Run on the analysis machine AFTER the models have been downloaded/cached:
    python env/resolve_revisions.py
Writes manifests/models_resolved.csv with a real commit hash per repo.
"""
from __future__ import annotations

import csv
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    raise SystemExit("pip install huggingface-hub, then re-run")

MAN = Path(__file__).resolve().parents[1] / "manifests" / "models.csv"
OUT = Path(__file__).resolve().parents[1] / "manifests" / "models_resolved.csv"


def latest_commit(api: HfApi, repo: str) -> str:
    if not repo:
        return ""
    try:
        info = api.model_info(repo)
        return info.sha or "UNKNOWN"
    except Exception as e:  # network / auth / gated
        return f"UNRESOLVED({e.__class__.__name__})"


def main():
    api = HfApi()
    rows = list(csv.DictReader(MAN.open()))
    for r in rows:
        r["revision"] = latest_commit(api, r["hf_repo"])
        if r.get("hf_repo_secondary"):
            r["revision_secondary"] = latest_commit(api, r["hf_repo_secondary"])
        else:
            r["revision_secondary"] = ""
    fields = list(rows[0].keys())
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} with resolved commit hashes for {len(rows)} models")
    print("Review it, confirm the hashes match what you actually ran, then commit it.")


if __name__ == "__main__":
    main()
