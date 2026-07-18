"""Mean-centering robustness check — does the genre effect survive removing the anisotropy
floor, without changing the retrieval task?

WHY CENTERING, NOT WHITENING
----------------------------
The embeddings are strongly anisotropic (off-diagonal cosine 0.65-0.84 for the top
retrievers). The concern is whether the genre η²≈0.55 is an artefact of that geometry.

Whitening (ZCA) is the WRONG tool: it re-weights dimensions and re-projects vectors, which
changes which document is nearest to a query — i.e. it produces a DIFFERENT retrieval
system, not a corrected view of the current one. Fit per-cell, it also gives each cell a
different transform, breaking the cross-cell comparison that is the whole point.

Mean-CENTERING is the minimal, correct intervention. Subtract the per-cell mean document
vector, then re-normalise. This removes the shared "common direction" that produces the
anisotropy floor, WITHOUT re-weighting dimensions. Crucially, we test whether the genre
effect and the transfer τ survive it.

A SUBTLETY THAT MATTERS
-----------------------
Plain max-sim MRR is already argsort-invariant to a CONSTANT added to all cosines, so
anisotropy does not mechanically bias the decomposition. Centering is a stronger operation
than subtracting a scalar: it removes a per-DIMENSION mean, which CAN change argsort. So
this is a genuine robustness test, not a no-op:
  - if genre η² and τ survive centering -> the effect is not an anisotropy artefact. Done.
  - if they change materially -> the geometry mattered, and we report the centered result.

We fit the centering mean on each cell's DOCUMENT chunks and apply it to both that cell's
document chunks and its queries (queries live in the same space). We report the decomposition
and cross-site τ under: (a) raw (paper's maxsim), (b) centered. Both use max-sim over chunks.

Uses ONLY cached chunk vectors. No re-embedding.

Usage:
    python centering_check.py
"""
from __future__ import annotations

import json
from collections import defaultdict
import os
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

RESULTS = Path(os.environ.get("RESULTS_DIR", str(Path.home() / "paper19_results")))
CACHE = RESULTS / "two_site_v2_emb"
CELLCACHE = RESULTS / "two_site_v2_cells.json"
TOP_K, SEED = 10, 42
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


def l2(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-9)


def maxsim_mrr(Q, Dv, Did, n_docs):
    sim = Q @ Dv.T
    dc = defaultdict(list)
    for c in range(len(Did)):
        dc[Did[c]].append(c)
    ds = np.full((len(Q), n_docs), -np.inf, np.float32)
    for d, rows in dc.items():
        ds[:, d] = sim[:, rows].max(axis=1)
    n = len(Q); k = min(TOP_K, n_docs - 1)
    rr = np.zeros(n, np.float32)
    for i in range(n):
        t = np.argpartition(-ds[i], k)[:k]
        for r, j in enumerate(t[np.argsort(-ds[i, t])]):
            if j == i:
                rr[i] = 1.0 / (r + 1); break
    return rr


def decompose(mrr, models):
    vals = [mrr[(s, g, m)] for s in SITES for g in GENRES for m in models]
    grand = np.mean(vals); sstot = sum((v - grand) ** 2 for v in vals)
    def fss(f):
        grp = defaultdict(list)
        for s in SITES:
            for g in GENRES:
                for m in models:
                    grp[f(m, s, g)].append(mrr[(s, g, m)])
        return sum(len(v) * (np.mean(v) - grand) ** 2 for v in grp.values())
    ssm = fss(lambda m, s, g: m); sss = fss(lambda m, s, g: s); ssg = fss(lambda m, s, g: g)
    msi = fss(lambda m, s, g: (m, s)) - ssm - sss
    mg = fss(lambda m, s, g: (m, g)) - ssm - ssg
    return {"model": ssm/sstot, "site": sss/sstot, "genre": ssg/sstot,
            "model×site": msi/sstot, "model×genre": mg/sstot}


def run(center):
    mrr = {}
    aniso_after = []
    for s in SITES:
        for g in GENRES:
            tag = f"{s}_{g}"; nd = len(json.loads(CELLCACHE.read_text())["cells"][f"{s}|{g}"])
            for m in CONTRA:
                Q = load_q(tag, m); Dv, Did = load_chunks(tag, m)
                if Q is None or Dv is None:
                    continue
                if center:
                    mu = Dv.mean(axis=0, keepdims=True)   # fit on document chunks
                    Dv = l2(Dv - mu)
                    Q = l2(Q - mu)                         # same space, same shift
                mrr[(s, g, m)] = float(maxsim_mrr(Q, Dv, Did, nd).mean())
                if center and m == "gte":
                    Xn = Dv[np.random.default_rng(0).choice(len(Dv), min(1500, len(Dv)), replace=False)]
                    S = Xn @ Xn.T
                    aniso_after.append(S[~np.eye(len(Xn), dtype=bool)].mean())
    return mrr, (np.mean(aniso_after) if aniso_after else None)


def taus(mrr):
    out = {}
    for g in GENRES:
        a = [mrr[('BIDMC', g, m)] for m in CONTRA if ('BIDMC', g, m) in mrr]
        b = [mrr[('UCSF', g, m)] for m in CONTRA if ('UCSF', g, m) in mrr]
        if len(a) == len(b) == len(CONTRA):
            out[g] = kendalltau(a, b)[0]
    return out


def main():
    if not CELLCACHE.exists():
        raise SystemExit(f"missing {CELLCACHE}")
    print("=" * 92)
    print("MEAN-CENTERING ROBUSTNESS — does genre survive removing the anisotropy floor?")
    print("=" * 92)

    raw, _ = run(center=False)
    cen, aniso = run(center=True)
    if len(raw) < len(CONTRA) * 4 or len(cen) < len(CONTRA) * 4:
        raise SystemExit(f"incomplete chunk cache under {CACHE}; verify with:\n"
                         f"  ls {CACHE} | grep chunk | head")

    dr, dc = decompose(raw, CONTRA), decompose(cen, CONTRA)
    tr, tc = taus(raw), taus(cen)

    print(f"\n  {'term':<16}{'raw maxsim':>12}{'centered':>12}")
    for k in ["genre", "model", "model×genre", "model×site", "site"]:
        print(f"  {k:<16}{dr[k]:>12.3f}{dc[k]:>12.3f}")
    print(f"\n  {'τ discharge':<16}{tr.get('discharge', float('nan')):>12.3f}"
          f"{tc.get('discharge', float('nan')):>12.3f}")
    print(f"  {'τ imaging':<16}{tr.get('imaging', float('nan')):>12.3f}"
          f"{tc.get('imaging', float('nan')):>12.3f}")
    if aniso is not None:
        print(f"\n  off-diagonal cosine after centering (GTE): {aniso:.3f}  "
              f"(was ~0.79 raw; centering should lower it)")

    print("\nINTERPRETATION")
    print("-" * 92)
    g_raw, g_cen = dr["genre"], dc["genre"]
    m_cen = dc["model"]
    print(f"  genre η²: raw {g_raw:.3f} -> centered {g_cen:.3f}")
    if g_cen >= 0.30 and g_cen >= m_cen:
        print("  -> genre SURVIVES centering as the largest factor. The effect is NOT an")
        print("     anisotropy artefact. Report maxsim as primary; genre is a genuine leading")
        print("     factor. Anisotropy is a property of the embeddings, not a confound here.")
    elif g_cen >= 0.25:
        print("  -> genre survives, comparable to model. Report both as major; note the")
        print("     ordering is sensitive but genre is not an artefact of the geometry.")
    else:
        print("  -> genre drops under centering: the geometry mattered. Report the centered")
        print("     decomposition as primary and revise the genre claim accordingly.")
    print(f"\n  transfer τ (raw vs centered): disch {tr.get('discharge',0):.2f}/"
          f"{tc.get('discharge',0):.2f}, imag {tr.get('imaging',0):.2f}/{tc.get('imaging',0):.2f}")
    print("  -> τ should stay positive under centering; the transfer thesis does not depend")
    print("     on the geometry and must survive this check.")


if __name__ == "__main__":
    main()
