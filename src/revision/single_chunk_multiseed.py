#!/usr/bin/env python3
"""Paper 19, editorial comment 12: single-random-chunk scoring across multiple seeds.

The submitted single-random sensitivity used one fixed-seed realisation. This
repeats it across S independent draws and reports the expected value and spread
of every quantity, so no point estimate rests on a single random draw.

Scoring matches two_site_v2_analyze.py: L2-normalised embeddings, cosine via dot
product, MRR@10, target i is document i. One chunk is drawn uniformly per
document per seed; documents with a single chunk are unaffected.

Usage:
    python3 single_chunk_multiseed.py               # 20 seeds
    python3 single_chunk_multiseed.py --seeds 50
    RESULTS_DIR=/path python3 single_chunk_multiseed.py
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

RESULTS = Path(os.environ.get("RESULTS_DIR", "/Users/yngve/projects/paper13/results"))
CACHE = RESULTS / "two_site_v2_emb"

TOP_K = 10
CONTRA = ["bge", "gte", "e5", "nomic", "mpnet", "minilm", "medcpt", "biolord"]
MLM = ["bert-base", "biobert", "clinicalbert", "pubmedbert", "scibert"]
ALL = CONTRA + MLM
SITES, GENRES = ["BIDMC", "UCSF"], ["discharge", "imaging"]
TERMS = ["model", "genre", "model x genre", "model x site", "site",
         "site x genre", "model x site x genre"]


def rr_from_D(Q, D):
    S = Q @ D.T
    n = len(D)
    rr = np.zeros(len(Q), np.float32)
    k = min(TOP_K, n - 1)
    for i in range(len(Q)):
        t = np.argpartition(-S[i], k)[:k]
        for r, j in enumerate(t[np.argsort(-S[i, t])]):
            if j == i:
                rr[i] = 1.0 / (r + 1)
                break
    return rr


def chunk_index(ids, n_docs):
    """Positions of each document's chunks."""
    per = defaultdict(list)
    for pos, d in enumerate(ids):
        per[int(d)].append(pos)
    return [np.array(per[d]) for d in range(n_docs)]


def decompose(mrr, models):
    vals = [mrr[(s, g, m)] for s in SITES for g in GENRES for m in models]
    grand = float(np.mean(vals))
    sstot = sum((v - grand) ** 2 for v in vals)
    if sstot <= 0:
        return {t: float("nan") for t in TERMS}

    def fss(f):
        grp = defaultdict(list)
        for s in SITES:
            for g in GENRES:
                for m in models:
                    grp[f(m, s, g)].append(mrr[(s, g, m)])
        return sum(len(v) * (np.mean(v) - grand) ** 2 for v in grp.values())

    ssm, sss, ssg = fss(lambda m, s, g: m), fss(lambda m, s, g: s), fss(lambda m, s, g: g)
    msi = fss(lambda m, s, g: (m, s)) - ssm - sss
    mg = fss(lambda m, s, g: (m, g)) - ssm - ssg
    sg = fss(lambda m, s, g: (s, g)) - sss - ssg
    msg = sstot - ssm - sss - ssg - msi - mg - sg
    return {"model": ssm / sstot, "site": sss / sstot, "genre": ssg / sstot,
            "model x site": msi / sstot, "model x genre": mg / sstot,
            "site x genre": sg / sstot, "model x site x genre": msg / sstot}


def taus(mrr, models):
    out = {}
    for genre in GENRES:
        a = [mrr[("BIDMC", genre, m)] for m in models]
        b = [mrr[("UCSF", genre, m)] for m in models]
        out[genre] = kendalltau(a, b)[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()

    if not CACHE.exists():
        sys.exit("FAIL: no embedding cache at %s" % CACHE)

    print("=" * 92)
    print("SINGLE-RANDOM-CHUNK SCORING ACROSS %d SEEDS (editorial comment 12)" % a.seeds)
    print("RESULTS_DIR = %s" % RESULTS)
    print("=" * 92)

    print("\nloading cached embeddings ...")
    data, panel = {}, []
    for m in ALL:
        ok = True
        for s in SITES:
            for g in GENRES:
                tag = "%s_%s_chunk_%s" % (s, g, m)
                qf, df = CACHE / ("%s_q.npy" % tag), CACHE / ("%s_d_chunk.npz" % tag)
                if not (qf.exists() and df.exists()):
                    ok = False
                    continue
                z = np.load(df)
                Q = np.load(qf).astype(np.float32)
                data[(s, g, m)] = (Q, z["v"].astype(np.float32), z["ids"].astype(int))
        if ok:
            panel.append(m)
    missing = [m for m in ALL if m not in panel]
    if missing:
        print("  NOTE: no complete cache for %s - excluded" % ", ".join(missing))
    panel_con = [m for m in CONTRA if m in panel]
    if len(panel_con) < 3:
        sys.exit("FAIL: fewer than 3 contrastive models cached")
    print("  %d models, %d cells" % (len(panel), len(SITES) * len(GENRES)))

    n_docs = {}
    idx = {}
    for k, (Q, Dv, Did) in data.items():
        n_docs[k] = len(Q)
        idx[k] = chunk_index(Did, len(Q))
    multi = {k: int(sum(1 for c in idx[k] if len(c) > 1)) for k in idx}
    print("\n  documents with more than one chunk (seed choice only matters for these):")
    for m in panel_con[:1]:
        for s in SITES:
            for g in GENRES:
                k = (s, g, m)
                print("    %-20s %5d / %5d (%.1f%%)"
                      % ("%s|%s" % (s, g), multi[k], n_docs[k],
                         100.0 * multi[k] / n_docs[k]))

    per_seed_mrr = []
    per_seed_dec = {"contrastive": [], "full": []}
    per_seed_tau = {"contrastive": [], "full": []}
    print("\nrunning seeds ...")
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        mrr = {}
        for k, (Q, Dv, Did) in data.items():
            pick = np.array([c[rng.integers(0, len(c))] for c in idx[k]])
            mrr[k] = float(rr_from_D(Q, Dv[pick]).mean())
        per_seed_mrr.append(mrr)
        per_seed_dec["contrastive"].append(decompose(mrr, panel_con))
        per_seed_tau["contrastive"].append(taus(mrr, panel_con))
        if len(panel) >= 3:
            per_seed_dec["full"].append(decompose(mrr, panel))
            per_seed_tau["full"].append(taus(mrr, panel))
        print("  seed %2d done" % seed, end="\r")
    print(" " * 30, end="\r")

    def summarize(vals):
        a_ = np.asarray(vals, dtype=float)
        return a_.mean(), a_.std(ddof=1), np.percentile(a_, 2.5), np.percentile(a_, 97.5)

    print("\nMRR@10 PER CELL AND MODEL (mean over seeds, SD)")
    print("-" * 92)
    print("  %-14s%22s%22s%22s" % ("model", "BIDMC disch", "BIDMC imag", "UCSF disch"))
    for m in panel:
        row = []
        for s, g in [("BIDMC", "discharge"), ("BIDMC", "imaging"), ("UCSF", "discharge")]:
            mu, sd, _, _ = summarize([d[(s, g, m)] for d in per_seed_mrr])
            row.append("%.4f (SD %.4f)" % (mu, sd))
        print("  %-14s%22s%22s%22s" % (m, *row))
    print("  (UCSF imaging omitted from this table for width; it is in the JSON)")

    for scope in ["contrastive", "full"]:
        if not per_seed_dec[scope]:
            continue
        label = "CONTRASTIVE-%d" % len(panel_con) if scope == "contrastive" else "FULL-%d" % len(panel)
        print("\nVARIANCE DECOMPOSITION, %s - single-random scorer over %d seeds"
              % (label, a.seeds))
        print("-" * 92)
        print("  %-24s%10s%10s%22s" % ("term", "mean", "SD", "2.5-97.5 pct"))
        for t in TERMS:
            mu, sd, lo, hi = summarize([d[t] for d in per_seed_dec[scope]])
            print("  %-24s%10.3f%10.3f     [%.3f, %.3f]" % (t, mu, sd, lo, hi))
        print("\n  cross-site tau, %s" % label)
        for genre in GENRES:
            mu, sd, lo, hi = summarize([d[genre] for d in per_seed_tau[scope]])
            print("    %-12s%10.3f%10.3f     [%.3f, %.3f]" % (genre, mu, sd, lo, hi))

    out = RESULTS / "single_chunk_multiseed.json"
    out.write_text(json.dumps({
        "seeds": a.seeds,
        "panel": panel,
        "panel_contrastive": panel_con,
        "multi_chunk_docs": {"%s|%s|%s" % k: v for k, v in multi.items()},
        "mrr_mean": {"%s|%s|%s" % k: float(np.mean([d[k] for d in per_seed_mrr]))
                     for k in per_seed_mrr[0]},
        "mrr_sd": {"%s|%s|%s" % k: float(np.std([d[k] for d in per_seed_mrr], ddof=1))
                   for k in per_seed_mrr[0]},
        "decomposition": {scope: {t: summarize([d[t] for d in per_seed_dec[scope]])
                                  for t in TERMS}
                          for scope in per_seed_dec if per_seed_dec[scope]},
        "tau": {scope: {g: summarize([d[g] for d in per_seed_tau[scope]]) for g in GENRES}
                for scope in per_seed_tau if per_seed_tau[scope]},
    }, indent=1, default=lambda x: [float(v) for v in x] if hasattr(x, "__iter__") else float(x)),
        encoding="utf-8")
    print("\nWritten: %s" % out)


if __name__ == "__main__":
    main()
