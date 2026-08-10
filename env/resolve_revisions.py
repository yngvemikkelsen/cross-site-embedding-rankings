"""Resolve the exact HuggingFace commit hash ACTUALLY USED for each model in manifests/models.csv.

Primary source is the LOCAL HuggingFace cache on this machine, i.e. the commit you actually
ran, read from ~/.cache/huggingface/hub/models--<org>--<name>/refs/<ref> (default ref: main).
This is the truthful provenance. If a model is not found in the local cache, the row is marked
NOT_CACHED(...) rather than silently substituting the Hub's current HEAD, so the output never
claims a revision you did not run.

Optionally, with --hub-fallback, uncached models are resolved to the Hub's CURRENT HEAD and
clearly labelled HUB_HEAD(<sha>) so you can decide whether that is acceptable.

Run on the analysis machine (models must be cached from your run):
    python env/resolve_revisions.py                 # local cache only (recommended)
    python env/resolve_revisions.py --hub-fallback  # also query Hub HEAD for uncached repos
Writes manifests/models_resolved.csv with the resolved commit per repo.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

MAN = Path(__file__).resolve().parents[1] / "manifests" / "models.csv"
OUT = Path(__file__).resolve().parents[1] / "manifests" / "models_resolved.csv"

HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
HUB = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", HF_HOME / "hub"))


def cache_commit(repo: str, ref: str = "main") -> str | None:
    """Return the commit sha this machine has checked out for `repo`, or None if not cached."""
    if not repo:
        return ""
    folder = HUB / ("models--" + repo.replace("/", "--"))
    ref_file = folder / "refs" / ref
    if ref_file.is_file():
        return ref_file.read_text().strip()
    # fall back to whatever snapshot exists on disk (single snapshot => unambiguous)
    snap = folder / "snapshots"
    if snap.is_dir():
        snaps = [p.name for p in snap.iterdir() if p.is_dir()]
        if len(snaps) == 1:
            return snaps[0]
        if len(snaps) > 1:
            return f"AMBIGUOUS({len(snaps)}_snapshots)"
    return None


def hub_head(repo: str) -> str:
    try:
        from huggingface_hub import HfApi
        sha = HfApi().model_info(repo).sha or "UNKNOWN"
        return f"HUB_HEAD({sha})"
    except Exception as e:
        return f"UNRESOLVED({e.__class__.__name__})"


def resolve(repo: str, hub_fallback: bool) -> str:
    if not repo:
        return ""
    c = cache_commit(repo)
    if c:
        return c
    return hub_head(repo) if hub_fallback else "NOT_CACHED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub-fallback", action="store_true",
                    help="for repos not in the local cache, record the Hub's current HEAD "
                         "(labelled HUB_HEAD) instead of NOT_CACHED")
    a = ap.parse_args()

    rows = list(csv.DictReader(MAN.open()))
    resolved = 0
    for r in rows:
        rev = resolve(r.get("hf_repo", ""), a.hub_fallback)
        r["revision"] = rev
        if rev and not rev.startswith(("NOT_CACHED", "UNRESOLVED", "HUB_HEAD", "AMBIGUOUS")):
            resolved += 1
        sec = r.get("hf_repo_secondary", "")
        r["revision_secondary"] = resolve(sec, a.hub_fallback) if sec else ""

    fields = list(rows[0].keys())
    if "revision_secondary" not in fields:
        fields.append("revision_secondary")
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {OUT}")
    print(f"resolved {resolved}/{len(rows)} primary repos from the local cache "
          f"({HUB})")
    unresolved = [r["key"] for r in rows
                  if str(r["revision"]).startswith(("NOT_CACHED", "UNRESOLVED", "AMBIGUOUS"))]
    if unresolved:
        print("NOT resolved from cache:", ", ".join(unresolved))
        print("  -> re-run those models so they are cached, or use --hub-fallback "
              "(records Hub HEAD, labelled) and verify manually.")
    print("Review the file, confirm the hashes match what you ran, then commit it.")


if __name__ == "__main__":
    main()
