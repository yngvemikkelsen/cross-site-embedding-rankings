"""Chunk-count sensitivity, done correctly: length-NORMALISED max-similarity.

WHY THE FIRST VERSION WAS WRONG
-------------------------------
The first sensitivity script used 'first-chunk' and 'capped-k=1' as the equalisers. But
imaging's median chunk count is 1, so capped-k=1 == first-chunk, and the first 512 tokens
of a long discharge note are its HEADER (Name/Service/Allergies/Chief-Complaint) — the very
boilerplate we removed from queries. The query is the HPI, buried mid-note. So first-chunk
discards the relevant span for long docs; its genre eta2 of 0.348 UNDERSTATES genre by
header-truncation, it does not cleanly correct for chunk count. That was an over-claim.

THE CORRECT CONTROL
-------------------
The order-statistic advantage is: a document with k chunks gets k independent draws at
matching the query, so E[max similarity] rises with k even for an IRRELEVANT query. Measure
that null directly on the real embeddings: for each document, score it against MISMATCHED
queries and record max-similarity vs its chunk count k. The fitted curve null(k) IS the
order-statistic advantage. Then:

    normalised_score(doc) = observed_max_sim(doc, its_query) - null(k_doc)

This removes the chunk-count boost WITHOUT discarding any chunk — the HPI stays in play.
If the genre effect survives this normalisation, it is genuine content difference. If it
collapses, it was chunk count.

We report FOUR scorers so the reader sees the full dependence:
    maxsim        — the paper's scorer (all chunks, best match)         [order-stat inflated]
    length-norm   — maxsim minus the empirical null(k) curve            [THE control]
    mean-pool     — whole-doc mean vector, one per doc                  [no order stat, but
                                                                          surfaces site level]
    single-random — max over ONE randomly chosen chunk per doc          [equal k=1, but keeps
                                                                          a RELEVANT-eligible
                                                                          chunk, unlike first]

Uses ONLY cached chunk vectors. No re-embedding.

Usage:
    python chunk_sensitivity2.py
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


def doc_chunk_rows(Did, n_docs):
    dc = defaultdict(list)
    for c in range(len(Did)):
        dc[Did[c]].append(c)
    return dc


def null_curve(Q, Dv, dc, n_docs, rng, n_sample=600):
    """Empirical E[max sim | MISMATCHED query] as a function of chunk count k.
    Returns a dict k -> mean null max-sim, plus a fallback interpolator."""
    ks, vals = [], []
    docs = list(dc.keys())
    sample = rng.choice(docs, min(n_sample, len(docs)), replace=False)
    for d in sample:
        qi = rng.integers(n_docs)
        if qi == d:
            qi = (qi + 1) % n_docs
        rows = dc[d]
        s = (Dv[rows] @ Q[qi]).max()
        ks.append(len(rows)); vals.append(s)
    ks, vals = np.array(ks), np.array(vals)
    # bin by k, mean per bin; interpolate for unseen k
    order = np.argsort(ks)
    ks_s, vals_s = ks[order], vals[order]
    uks = np.unique(ks_s)
    curve = {int(k): float(vals_s[ks_s == k].mean()) for k in uks}
    return curve, (uks, np.array([curve[int(k)] for k in uks]))


def null_at(k, interp):
    uks, uv = interp
    if k <= uks[0]:
        return uv[0]
    if k >= uks[-1]:
        return uv[-1]
    return float(np.interp(k, uks, uv))


def mrr_from_scores(S):
    n = S.shape[0]; k = min(TOP_K, S.shape[1] - 1)
    rr = np.zeros(n, np.float32)
    for i in range(n):
        t = np.argpartition(-S[i], k)[:k]
        for r, j in enumerate(t[np.argsort(-S[i, t])]):
            if j == i:
                rr[i] = 1.0 / (r + 1); break
    return rr


def score(Q, Dv, Did, n_docs, policy, rng):
    dc = doc_chunk_rows(Did, n_docs)
    if policy == "mean":
        D = np.zeros((n_docs, Dv.shape[1]), np.float32); cnt = np.zeros(n_docs)
        for c in range(len(Did)):
            D[Did[c]] += Dv[c]; cnt[Did[c]] += 1
        D /= np.maximum(cnt, 1)[:, None]
        D /= np.linalg.norm(D, axis=1, keepdims=True).clip(1e-9)
        return mrr_from_scores(Q @ D.T)

    sim = Q @ Dv.T                                       # (nq, nchunks)
    ds = np.full((len(Q), n_docs), -np.inf, np.float32)
    if policy == "maxsim":
        for d, rows in dc.items():
            ds[:, d] = sim[:, rows].max(axis=1)
    elif policy == "single-random":
        for d, rows in dc.items():
            pick = rows[rng.integers(len(rows))]
            ds[:, d] = sim[:, pick]
    elif policy == "length-norm":
        curve, interp = null_curve(Q, Dv, dc, n_docs, rng)
        for d, rows in dc.items():
            ds[:, d] = sim[:, rows].max(axis=1) - null_at(len(rows), interp)
    return mrr_from_scores(ds)


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


def main():
    if not CELLCACHE.exists():
        raise SystemExit(f"missing {CELLCACHE}")
    cells = json.loads(CELLCACHE.read_text())["cells"]
    rng = np.random.default_rng(SEED)

    print("=" * 98)
    print("CHUNK-COUNT SENSITIVITY v2 — length-normalised, no relevant-span discarding")
    print("=" * 98)

    # show the null curve for one representative cell so the reader sees the order statistic
    tag = "BIDMC_discharge"
    Q = load_q(tag, "gte"); Dv, Did = load_chunks(tag, "gte")
    if Q is None:
        raise SystemExit(f"chunk cache not found under {CACHE}. Verify with:\n"
                         f"  ls {CACHE} | grep chunk | head")
    n_docs = len(cells["BIDMC|discharge"])
    dc = doc_chunk_rows(Did, n_docs)
    curve, interp = null_curve(Q, Dv, dc, n_docs, np.random.default_rng(1))
    print("Empirical NULL order-statistic curve (BIDMC/discharge, GTE): E[max sim | WRONG query]")
    for lo, hi in [(1, 1), (2, 4), (5, 8), (9, 15), (16, 60)]:
        keys = [k for k in curve if lo <= k <= hi]
        if keys:
            print(f"  k {lo:>2}-{hi:<2}: {np.mean([curve[k] for k in keys]):+.4f}")
    print("  -> rising with k = the order-statistic advantage, measured on real embeddings.")
    print("  length-norm subtracts this from each doc's observed max-sim.\n")

    policies = ["maxsim", "length-norm", "mean", "single-random"]
    summary = {}
    for policy in policies:
        mrr = {}
        for s in SITES:
            for g in GENRES:
                t = f"{s}_{g}"; nd = len(cells[f"{s}|{g}"])
                for m in CONTRA:
                    Qm = load_q(t, m); Dvm, Didm = load_chunks(t, m)
                    if Qm is None or Dvm is None:
                        continue
                    mrr[(s, g, m)] = float(score(Qm, Dvm, Didm, nd, policy,
                                                 np.random.default_rng(SEED)).mean())
        if len(mrr) < len(CONTRA) * 4:
            print(f"  {policy}: incomplete cache")
            continue
        dec = decompose(mrr, CONTRA)
        gap = np.mean([mrr[(s, 'discharge', m)] - mrr[(s, 'imaging', m)]
                       for s in SITES for m in CONTRA])
        taus = {g: kendalltau([mrr[('BIDMC', g, m)] for m in CONTRA],
                              [mrr[('UCSF', g, m)] for m in CONTRA])[0] for g in GENRES}
        summary[policy] = (dec, gap, taus)

    print("VARIANCE DECOMPOSITION BY SCORER (contrastive models)")
    print("-" * 98)
    print(f"  {'scorer':<16}{'genre':>8}{'model':>8}{'m×genre':>9}{'m×site':>8}{'site':>7}"
          f"{'  D−I gap':>10}{'  τ disch':>9}{'  τ imag':>9}")
    for policy in policies:
        if policy not in summary:
            continue
        dec, gap, taus = summary[policy]
        print(f"  {policy:<16}{dec['genre']:>8.3f}{dec['model']:>8.3f}{dec['model×genre']:>9.3f}"
              f"{dec['model×site']:>8.3f}{dec['site']:>7.3f}{gap:>10.3f}"
              f"{taus['discharge']:>9.3f}{taus['imaging']:>9.3f}")

    print("\nINTERPRETATION")
    print("-" * 98)
    if "maxsim" in summary and "length-norm" in summary:
        g_max = summary["maxsim"][0]["genre"]
        g_ln = summary["length-norm"][0]["genre"]
        m_ln = summary["length-norm"][0]["model"]
        print(f"  genre η²: maxsim {g_max:.3f} -> length-normalised {g_ln:.3f}")
        drop = (g_max - g_ln) / g_max if g_max else 0
        print(f"  chunk-count accounts for ~{drop:.0%} of the maxsim genre effect (length-norm).")
        if g_ln >= 0.30 and g_ln >= m_ln:
            print("  -> genre SURVIVES as the largest factor after correction: effect is genuine.")
        elif g_ln >= 0.25:
            print("  -> genre survives but is COMPARABLE to model after correction: report both")
            print("     as co-dominant, neither cleanly largest, magnitude scorer-dependent.")
        else:
            print("  -> genre COLLAPSES after correction: reframe as document-length effect.")
        print("\n  Compare length-norm (keeps all chunks, removes only the order statistic) to")
        print("  mean-pool (removes order statistic but surfaces a site level effect) and to")
        print("  single-random (equal k=1 but keeps a relevance-eligible chunk). Convergence")
        print("  across these bounds the true genre magnitude better than any single scorer.")
        print("\n  Transfer check: cross-site τ should stay POSITIVE across scorers (it is about")
        print("  ranking, not level). Read the τ columns to confirm the thesis is scorer-robust.")


if __name__ == "__main__":
    main()
