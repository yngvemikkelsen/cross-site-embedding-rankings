#!/usr/bin/env python3
"""Paper 19: every corrected number needed for the manuscript propagation pass.

Derives Tables 1-5, the paired eta-squared contrast, and the transfer/regret
intervals from the corrected rr vectors. Hit@1 and Hit@10 are exact functions
of the rr vector (rr = 1/rank for rank <= 10, else 0), so no rescoring is needed.

Output is formatted as it should appear in the manuscript.

Usage:
    python3 propagation_pack.py
    RESULTS_DIR=/path python3 propagation_pack.py

Bootstrap: 2000 replicates, seed 42, one patient draw per site applied jointly
to both of that site's genre cells, matching the manuscript's stated scheme.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

RESULTS = Path(os.environ.get("RESULTS_DIR", "/Users/yngve/projects/paper13/results"))
RRCACHE = RESULTS / "two_site_v2_rr_chunk"
CELLS = RESULTS / "two_site_v2_cells.json"

SEED, N_BOOT = 42, 2000
CONTRA = ["bge", "gte", "e5", "nomic", "mpnet", "minilm", "medcpt", "biolord"]
MLM = ["bert-base", "biobert", "clinicalbert", "pubmedbert", "scibert"]
ALL = CONTRA + MLM
SITES, GENRES = ["BIDMC", "UCSF"], ["discharge", "imaging"]
CELLKEYS = [(s, g) for s in SITES for g in GENRES]
TERMS = ["model", "genre", "model x genre", "model x site", "site",
         "site x genre", "model x site x genre"]


def load():
    if not RRCACHE.exists():
        sys.exit("FAIL: no rr cache at %s" % RRCACHE)
    cells = json.loads(CELLS.read_text())["cells"]
    pats = {c: [r["patient"] for r in recs] for c, recs in cells.items()}
    rr = {f.stem: np.load(f).astype(np.float64) for f in RRCACHE.glob("*.npy")}
    return rr, pats


def site_index(pats):
    """Patient -> per-genre index arrays, so one draw per site applies jointly to
    that site's two genre cells (the scheme the manuscript states)."""
    idx = {}
    for s in SITES:
        per = defaultdict(lambda: {g: [] for g in GENRES})
        for g in GENRES:
            for i, p in enumerate(pats["%s|%s" % (s, g)]):
                per[p][g].append(i)
        idx[s] = {p: {g: np.array(v[g], dtype=int) for g in GENRES} for p, v in per.items()}
    return idx


def decompose(mrr, models):
    vals = [mrr[(s, g, m)] for s, g in CELLKEYS for m in models]
    grand = float(np.mean(vals))
    sstot = sum((v - grand) ** 2 for v in vals)
    if sstot <= 0:
        return {t: float("nan") for t in TERMS}

    def fss(f):
        grp = defaultdict(list)
        for s, g in CELLKEYS:
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


def transfer(mrr, models_full, models_con):
    out = {}
    for genre in GENRES:
        for lbl, sub in [("full", models_full), ("contrastive", models_con)]:
            a = [mrr[("BIDMC", genre, m)] for m in sub]
            b = [mrr[("UCSF", genre, m)] for m in sub]
            out[(genre, lbl)] = kendalltau(a, b)[0]
        a = {m: mrr[("BIDMC", genre, m)] for m in models_con}
        b = {m: mrr[("UCSF", genre, m)] for m in models_con}
        bestA, bestB = max(a, key=a.get), max(b, key=b.get)
        out[(genre, "regret_BU")] = b[bestB] - b[bestA]
        out[(genre, "regret_UB")] = a[bestA] - a[bestB]
        out[(genre, "best_BIDMC")] = bestA
        out[(genre, "best_UCSF")] = bestB
    return out


def ci(x):
    return float(np.mean(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def main():
    rr, pats = load()
    key = lambda s, g, m: "%s_%s__%s" % (s, g, m)
    panel = [m for m in ALL if all(key(s, g, m) in rr for s, g in CELLKEYS)]
    panel_con = [m for m in CONTRA if m in panel]
    print("=" * 96)
    print("PROPAGATION PACK - corrected values for the manuscript")
    print("RESULTS_DIR = %s   models = %d" % (RESULTS, len(panel)))
    print("=" * 96)

    sidx = site_index(pats)
    pmap = {(s, g, m): float(rr[key(s, g, m)].mean()) for s, g in CELLKEYS for m in panel}

    # ---- bootstrap: one patient resample per site, shared across genres ----
    rng = np.random.default_rng(SEED)
    boot_mrr, boot_hit1, boot_hit10 = defaultdict(list), defaultdict(list), defaultdict(list)
    boot_dec = {"con": defaultdict(list), "full": defaultdict(list)}
    boot_tr = defaultdict(list)
    boot_delta = []
    boot_delta_full = []
    boot_cell = {"mrr": defaultdict(list), "hit1": defaultdict(list),
                 "hit10": defaultdict(list)}
    for _ in range(N_BOOT):
        sel = {}
        for s in SITES:
            plist = list(sidx[s])
            draw = rng.integers(0, len(plist), len(plist))
            chosen = [plist[i] for i in draw]
            for g in GENRES:
                parts = [sidx[s][p][g] for p in chosen if len(sidx[s][p][g])]
                sel[(s, g)] = (np.concatenate(parts) if parts
                               else np.array([], dtype=int))
        m_rep = {}
        for s, g in CELLKEYS:
            ix = sel[(s, g)]
            for m in panel:
                v = rr[key(s, g, m)][ix]
                m_rep[(s, g, m)] = float(v.mean())
                boot_mrr[(s, g, m)].append(v.mean())
                boot_hit1[(s, g, m)].append((v == 1.0).mean())
                boot_hit10[(s, g, m)].append((v > 0).mean())
        for s_, g_ in CELLKEYS:
            ix = sel[(s_, g_)]
            vs = [rr[key(s_, g_, m)][ix] for m in panel_con]
            boot_cell["mrr"][(s_, g_)].append(np.mean([v.mean() for v in vs]))
            boot_cell["hit1"][(s_, g_)].append(np.mean([(v == 1.0).mean() for v in vs]))
            boot_cell["hit10"][(s_, g_)].append(np.mean([(v > 0).mean() for v in vs]))
        dc = decompose(m_rep, panel_con)
        df = decompose(m_rep, panel)
        for t in TERMS:
            boot_dec["con"][t].append(dc[t])
            boot_dec["full"][t].append(df[t])
        boot_delta.append(dc["model x genre"] - dc["model x site"])
        boot_delta_full.append(df["model x genre"] - df["model x site"])
        for k, v in transfer(m_rep, panel, panel_con).items():
            if isinstance(v, float):
                boot_tr[k].append(v)

    # ---- Table 1 and 2 ----
    print("\nTABLE 1 - MRR@10 (point, 95% patient-clustered CI)")
    print("-" * 96)
    print("  %-14s%24s%24s%24s%24s" % ("model", "BIDMC discharge", "BIDMC imaging",
                                       "UCSF discharge", "UCSF imaging"))
    for m in panel:
        cells_ = []
        for s, g in CELLKEYS:
            _, lo, hi = ci(boot_mrr[(s, g, m)])
            cells_.append("%.3f (%.3f-%.3f)" % (pmap[(s, g, m)], lo, hi))
        print("  %-14s%24s%24s%24s%24s" % (m, *cells_))

    for label, store in [("TABLE 2a - Hit@1", boot_hit1), ("TABLE 2b - Hit@10", boot_hit10)]:
        print("\n%s (point, 95%% CI)" % label)
        print("-" * 96)
        print("  %-14s%24s%24s%24s%24s" % ("model", "BIDMC discharge", "BIDMC imaging",
                                           "UCSF discharge", "UCSF imaging"))
        for m in panel:
            cells_ = []
            for s, g in CELLKEYS:
                v = rr[key(s, g, m)]
                pt = (v == 1.0).mean() if store is boot_hit1 else (v > 0).mean()
                _, lo, hi = ci(store[(s, g, m)])
                cells_.append("%.3f (%.3f-%.3f)" % (pt, lo, hi))
            print("  %-14s%24s%24s%24s%24s" % (m, *cells_))

    # ---- Table 2 as printed: contrastive-panel means per cell ----
    print("\nTABLE 2 AS PRINTED - mean over the %d contrastive models, per cell" % len(panel_con))
    print("-" * 96)
    print("  %-20s%28s%28s" % ("Cell", "Hit@1 (95% CI)", "Hit@10 (95% CI)"))
    for s_, g_ in CELLKEYS:
        h1 = np.mean([(rr[key(s_, g_, m)] == 1.0).mean() for m in panel_con])
        h10 = np.mean([(rr[key(s_, g_, m)] > 0).mean() for m in panel_con])
        _, l1, u1 = ci(boot_cell["hit1"][(s_, g_)])
        _, l10, u10 = ci(boot_cell["hit10"][(s_, g_)])
        print("  %-20s%28s%28s"
              % ("%s %s" % (s_, g_), "%.3f (%.3f-%.3f)" % (h1, l1, u1),
                 "%.3f (%.3f-%.3f)" % (h10, l10, u10)))
    print("\n  contrastive-panel mean MRR@10 per cell:")
    for s_, g_ in CELLKEYS:
        mm = np.mean([pmap[(s_, g_, m)] for m in panel_con])
        _, lo, hi = ci(boot_cell["mrr"][(s_, g_)])
        print("    %-20s%.3f (%.3f-%.3f)" % ("%s %s" % (s_, g_), mm, lo, hi))

    # ---- Tables 3, 4 ----
    tr = transfer(pmap, panel, panel_con)
    print("\nTABLE 3 - CROSS-SITE RANK TRANSFER")
    print("-" * 96)
    for genre in GENRES:
        for lbl in ["full", "contrastive"]:
            _, lo, hi = ci(boot_tr[(genre, lbl)])
            print("  tau  %-10s %-13s %+.3f  (95%% CI %+.3f to %+.3f)"
                  % (genre, lbl, tr[(genre, lbl)], lo, hi))
    print("\nTABLE 4 - CROSS-SITE SELECTION REGRET (contrastive)")
    print("-" * 96)
    for genre in GENRES:
        _, lo1, hi1 = ci(boot_tr[(genre, "regret_BU")])
        _, lo2, hi2 = ci(boot_tr[(genre, "regret_UB")])
        print("  %-10s BIDMC-best=%-8s at UCSF  regret %+.4f (%.4f-%.4f)"
              % (genre, tr[(genre, "best_BIDMC")], tr[(genre, "regret_BU")], lo1, hi1))
        print("  %-10s UCSF-best=%-9s at BIDMC regret %+.4f (%.4f-%.4f)"
              % ("", tr[(genre, "best_UCSF")], tr[(genre, "regret_UB")], lo2, hi2))

    # ---- Table 5 + paired contrast ----
    dcon, dfull = decompose(pmap, panel_con), decompose(pmap, panel)
    print("\nTABLE 5 - VARIANCE DECOMPOSITION (point, 95% CI)")
    print("-" * 96)
    print("  %-24s%32s%32s" % ("term", "contrastive %d" % len(panel_con),
                               "full %d" % len(panel)))
    for t in TERMS:
        _, lo1, hi1 = ci(boot_dec["con"][t])
        _, lo2, hi2 = ci(boot_dec["full"][t])
        print("  %-24s%32s%32s"
              % (t, "%.3f (%.3f-%.3f)" % (dcon[t], lo1, hi1),
                 "%.3f (%.3f-%.3f)" % (dfull[t], lo2, hi2)))
    d_pt = dcon["model x genre"] - dcon["model x site"]
    _, dlo, dhi = ci(boot_delta)
    print("\n  Paired contrast, contrastive panel:")
    print("    delta eta2 = eta2(model x genre) - eta2(model x site) = %+.3f  (95%% CI %+.3f to %+.3f)"
          % (d_pt, dlo, dhi))
    print("    CI excludes zero: %s" % ("YES" if dlo > 0 or dhi < 0 else "NO"))
    df_pt = dfull["model x genre"] - dfull["model x site"]
    _, flo, fhi = ci(boot_delta_full)
    print("\n  Paired contrast, full %d-model panel:" % len(panel))
    print("    delta eta2 = %+.3f  (95%% CI %+.3f to %+.3f)" % (df_pt, flo, fhi))
    print("    CI excludes zero: %s" % ("YES" if flo > 0 or fhi < 0 else "NO"))

    out = RESULTS / "propagation_pack.json"
    out.write_text(json.dumps({
        "table1_mrr": {"%s|%s|%s" % k: v for k, v in pmap.items()},
        "table1_ci": {"%s|%s|%s" % k: list(ci(v)[1:]) for k, v in boot_mrr.items()},
        "table2_hit1": {"%s|%s|%s" % (s, g, m): float((rr[key(s, g, m)] == 1.0).mean())
                        for s, g in CELLKEYS for m in panel},
        "table2_hit10": {"%s|%s|%s" % (s, g, m): float((rr[key(s, g, m)] > 0).mean())
                         for s, g in CELLKEYS for m in panel},
        "table2_hit1_ci": {"%s|%s|%s" % k: list(ci(v)[1:]) for k, v in boot_hit1.items()},
        "table2_hit10_ci": {"%s|%s|%s" % k: list(ci(v)[1:]) for k, v in boot_hit10.items()},
        "table3_tau": {"%s|%s" % k: v for k, v in tr.items() if isinstance(v, float)},
        "table3_ci": {"%s|%s" % k: list(ci(v)[1:]) for k, v in boot_tr.items()},
        "table5_contrastive": dcon, "table5_full": dfull,
        "table5_contrastive_ci": {t: list(ci(boot_dec["con"][t])[1:]) for t in TERMS},
        "table5_full_ci": {t: list(ci(boot_dec["full"][t])[1:]) for t in TERMS},
        "delta_eta2": {"point": d_pt, "ci": [dlo, dhi]},
        "delta_eta2_full_panel": {"point": dfull["model x genre"] - dfull["model x site"],
                                  "ci": list(ci(boot_delta_full)[1:])},
        "table2_as_printed": {
            "%s|%s" % (s_, g_): {
                "hit1": float(np.mean([(rr[key(s_, g_, m)] == 1.0).mean() for m in panel_con])),
                "hit1_ci": list(ci(boot_cell["hit1"][(s_, g_)])[1:]),
                "hit10": float(np.mean([(rr[key(s_, g_, m)] > 0).mean() for m in panel_con])),
                "hit10_ci": list(ci(boot_cell["hit10"][(s_, g_)])[1:]),
                "mrr": float(np.mean([pmap[(s_, g_, m)] for m in panel_con])),
                "mrr_ci": list(ci(boot_cell["mrr"][(s_, g_)])[1:]),
            } for s_, g_ in CELLKEYS},
    }, indent=1), encoding="utf-8")
    print("\nWritten: %s" % out)


if __name__ == "__main__":
    main()
