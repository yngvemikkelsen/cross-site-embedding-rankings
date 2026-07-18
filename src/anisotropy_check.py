"""Anisotropy diagnostic — determines whether the length-norm scorer is trustworthy.

WHY THIS EXISTS
---------------
chunk_sensitivity2.py's length-norm scorer subtracts an empirical null curve
E[max-sim | wrong query] from each document's score. That null came out at ~0.77-0.81,
which is suspiciously high: if random (mismatched) query-document pairs already sit at
cosine ~0.8, the embedding space is ANISOTROPIC (all vectors point in a similar direction),
and the length-norm correction may be measuring anisotropy rather than the chunk-count
order statistic it was meant to remove. In that case its genre η²=0.157 is NOT trustworthy.

This script measures anisotropy directly, so the length-norm result can be trusted or
discarded on evidence rather than assumption. It does NOT re-embed; it reads cached vectors.

WHAT IT REPORTS, per cell and model
-----------------------------------
  * mean off-diagonal cosine between DIFFERENT chunks (the anisotropy number)
  * mean vector norm (should be ~1.0 if L2-normalised; sanity check)
  * mean cosine to the corpus centroid (another anisotropy view)
  * the null-vs-k slope: does E[max-sim|wrong query] actually rise with chunk count k,
    AND how much of the 0.77 baseline is the constant anisotropy floor vs the k-slope

HOW TO READ IT
--------------
  off-diagonal cosine near 0.0-0.2  -> isotropic; length-norm is measuring chunk count;
                                       its genre η²=0.157 deserves weight.
  off-diagonal cosine 0.5+          -> anisotropic; length-norm subtracts a large near-
                                       constant dominated by anisotropy, and the between-
                                       genre differential is an anisotropy artefact, NOT a
                                       clean chunk-count correction -> DISCARD length-norm's
                                       0.157, and treat the genre/model ordering as
                                       scorer-fragile (unresolved).

  The KEY quantity is the SLOPE of null(k), isolated from the anisotropy FLOOR:
    null(k) = floor + slope_term(k)
  If floor >> slope_term, length-norm is dominated by the floor (bad). If slope_term is
  comparable to floor, the k-correction is real.

Usage:
    python anisotropy_check.py
"""
from __future__ import annotations

import json
from collections import defaultdict
import os
from pathlib import Path

import numpy as np

RESULTS = Path(os.environ.get("RESULTS_DIR", str(Path.home() / "paper19_results")))
CACHE = RESULTS / "two_site_v2_emb"
CELLCACHE = RESULTS / "two_site_v2_cells.json"
SEED = 42
CONTRA = ["bge", "gte", "e5", "nomic", "mpnet", "minilm", "medcpt", "biolord"]
SITES, GENRES = ["BIDMC", "UCSF"], ["discharge", "imaging"]


def load_q(tag, key):
    f = CACHE / f"{tag}_chunk_{key}_q.npy"
    return np.load(f) if f.exists() else None


def load_chunks(tag, key):
    f = CACHE / f"{tag}_chunk_{key}_d_chunk.npz"
    if not f.exists():
        return None, None
    z = np.load(f)
    return z["v"], z["ids"]


def anisotropy(X, rng, n=2000):
    """Mean off-diagonal cosine among a random sample of chunk vectors."""
    if len(X) > n:
        X = X[rng.choice(len(X), n, replace=False)]
    # X assumed L2-normalised; if not, normalise for the cosine
    norms = np.linalg.norm(X, axis=1)
    Xn = X / norms[:, None].clip(1e-9)
    S = Xn @ Xn.T
    m = ~np.eye(len(Xn), dtype=bool)
    centroid = Xn.mean(0)
    centroid /= np.linalg.norm(centroid).clip(1e-9)
    return {
        "offdiag_cos": float(S[m].mean()),
        "offdiag_cos_p95": float(np.percentile(S[m], 95)),
        "mean_norm": float(norms.mean()),
        "mean_cos_to_centroid": float((Xn @ centroid).mean()),
    }


def null_slope(Q, Dv, Did, n_docs, rng, n_sample=800):
    """E[max-sim | wrong query] by chunk count k. Returns floor (k=1) and slope to high k."""
    dc = defaultdict(list)
    for c in range(len(Did)):
        dc[Did[c]].append(c)
    docs = list(dc.keys())
    sample = rng.choice(docs, min(n_sample, len(docs)), replace=False)
    by_k = defaultdict(list)
    for d in sample:
        qi = rng.integers(n_docs)
        if qi == d:
            qi = (qi + 1) % n_docs
        rows = dc[d]
        by_k[len(rows)].append(float((Dv[rows] @ Q[qi]).max()))
    ks = sorted(by_k)
    floor = np.mean(by_k[ks[0]]) if ks else float("nan")
    high = np.mean([v for k in ks if k >= 5 for v in by_k[k]]) if any(k >= 5 for k in ks) else float("nan")
    return floor, high, {k: float(np.mean(by_k[k])) for k in ks}


def main():
    if not CELLCACHE.exists():
        raise SystemExit(f"missing {CELLCACHE}")
    cells = json.loads(CELLCACHE.read_text())["cells"]
    rng = np.random.default_rng(SEED)

    print("=" * 96)
    print("ANISOTROPY DIAGNOSTIC — is length-norm measuring chunk count, or anisotropy?")
    print("=" * 96)
    print("Reading cached document chunk vectors (no re-embedding).\n")

    print(f"  {'cell':<18}{'model':<10}{'offdiag_cos':>12}{'p95':>8}{'norm':>7}"
          f"{'cos→centroid':>14}")
    print("  " + "-" * 90)
    agg = defaultdict(list)
    for s in SITES:
        for g in GENRES:
            tag = f"{s}_{g}"
            for m in CONTRA:
                Dv, Did = load_chunks(tag, m)
                if Dv is None:
                    continue
                a = anisotropy(Dv, np.random.default_rng(SEED))
                agg[m].append(a["offdiag_cos"])
                print(f"  {s+'/'+g:<18}{m:<10}{a['offdiag_cos']:>12.3f}"
                      f"{a['offdiag_cos_p95']:>8.3f}{a['mean_norm']:>7.2f}"
                      f"{a['mean_cos_to_centroid']:>14.3f}")

    print("\n  Mean off-diagonal cosine by model (across cells):")
    for m in CONTRA:
        if agg[m]:
            mv = np.mean(agg[m])
            tag = ("ISOTROPIC — length-norm OK" if mv < 0.25 else
                   "MILD" if mv < 0.5 else
                   "ANISOTROPIC — length-norm confounded, discard its 0.157")
            print(f"    {m:<10} {mv:>6.3f}   {tag}")

    print("\n" + "=" * 96)
    print("NULL CURVE: floor (anisotropy) vs k-slope (the real chunk-count part)")
    print("=" * 96)
    print(f"  {'cell':<18}{'model':<10}{'floor(k=1)':>12}{'high(k≥5)':>11}{'slope':>9}"
          f"   slope/floor")
    print("  " + "-" * 88)
    for s in SITES:
        for g in GENRES:
            if g != "discharge":       # slope only meaningful where docs have many chunks
                continue
            tag = f"{s}_{g}"
            for m in CONTRA[:4]:        # a few models is enough to see the pattern
                Q = load_q(tag, m); Dv, Did = load_chunks(tag, m)
                if Q is None or Dv is None:
                    continue
                nd = len(cells[f"{s}|{g}"])
                floor, high, _ = null_slope(Q, Dv, Did, nd, np.random.default_rng(SEED))
                slope = high - floor
                ratio = slope / floor if floor else float("nan")
                print(f"  {s+'/'+g:<18}{m:<10}{floor:>12.3f}{high:>11.3f}{slope:>9.3f}"
                      f"   {ratio:>8.2f}")

    print("\nINTERPRETATION")
    print("-" * 96)
    print("  If off-diagonal cosine is HIGH (>0.5), the null's 0.77-0.81 is mostly the")
    print("  anisotropy FLOOR, not the k-slope. length-norm then subtracts a near-constant")
    print("  that differs only slightly by genre, and its genre η²=0.157 is an anisotropy")
    print("  artefact -> DISCARD it. Conclude: genre/model ordering is scorer-fragile and")
    print("  unresolved; build the paper on the robust core (transfer holds, site small).")
    print()
    print("  If off-diagonal cosine is LOW (<0.25) and slope/floor is large, length-norm is")
    print("  genuinely removing chunk count -> its 0.157 deserves weight, and the genre")
    print("  effect really is largely chunk-count-driven.")
    print()
    print("  Either way, the transfer thesis (τ positive, model×site ~0.01) is untouched;")
    print("  this only decides how to frame the RQ2 decomposition.")


if __name__ == "__main__":
    main()
