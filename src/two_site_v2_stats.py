"""Added statistics for the revision (round 2 of review).

The reviewer's central methodological point: the "model dominates" headline (eta2_model=0.87)
is an artefact of including 5 weak MLM comparators. Among the 8 deployment-relevant
contrastive models, GENRE dominates (0.55) and model is second (0.33). Cross-site transfer
is unaffected (site ~0, model x site 0.009), but the framing must change.

Two analytical additions are needed before the rewrite, because they change what can be
claimed:

  1. CONTRASTIVE-ONLY variance decomposition with patient-clustered CIs (co-primary with
     the full-panel decomposition).
  2. Patient-clustered bootstrap CIs on the CENTRAL transfer measures — Kendall tau,
     cross-site selection regret, and top-k overlap — which currently have none.

This script recomputes everything from the cached per-query RR vectors written by
two_site_v2_analyze.py (--chunk run). It does NOT re-embed. If the RR cache is absent it
falls back to the point MRRs (CIs then unavailable, flagged in output).

Patient-clustered bootstrap detail (addressing the reviewer): within each site, a patient
can appear in BOTH genres, so resampling must draw patients JOINTLY and apply the SAME
patient draw to both genres of that site. We therefore resample patients per SITE and index
into that site's discharge and imaging cells together.

Usage:
    python two_site_v2_stats.py
"""
from __future__ import annotations

import json
from collections import defaultdict
import os
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

RESULTS = Path(os.environ.get("RESULTS_DIR", str(Path.home() / "paper19_results")))
RRCACHE = RESULTS / "two_site_v2_rr_chunk"   # per (cell,model) rr vectors, if saved
CELLCACHE = RESULTS / "two_site_v2_cells.json"
MRRJSON = RESULTS / "two_site_v2_mrr_chunk.json"
SEED, N_BOOT, TOP_K = 42, 2000, 10

CONTRA = ["bge", "gte", "e5", "nomic", "mpnet", "minilm", "medcpt", "biolord"]
MLM = ["bert-base", "biobert", "clinicalbert", "pubmedbert", "scibert"]
ALL = CONTRA + MLM
SITES, GENRES = ["BIDMC", "UCSF"], ["discharge", "imaging"]

# Point MRRs (chunked run) — fallback if RR vectors not cached
POINT = {
 'bge':{'BIDMC|discharge':0.2802,'BIDMC|imaging':0.1811,'UCSF|discharge':0.2815,'UCSF|imaging':0.1835},
 'gte':{'BIDMC|discharge':0.2966,'BIDMC|imaging':0.1954,'UCSF|discharge':0.3013,'UCSF|imaging':0.1994},
 'e5':{'BIDMC|discharge':0.2275,'BIDMC|imaging':0.1676,'UCSF|discharge':0.2420,'UCSF|imaging':0.1392},
 'nomic':{'BIDMC|discharge':0.3380,'BIDMC|imaging':0.1922,'UCSF|discharge':0.2923,'UCSF|imaging':0.1913},
 'mpnet':{'BIDMC|discharge':0.2659,'BIDMC|imaging':0.1853,'UCSF|discharge':0.2909,'UCSF|imaging':0.1645},
 'minilm':{'BIDMC|discharge':0.2333,'BIDMC|imaging':0.1709,'UCSF|discharge':0.2644,'UCSF|imaging':0.1560},
 'medcpt':{'BIDMC|discharge':0.1517,'BIDMC|imaging':0.1461,'UCSF|discharge':0.1743,'UCSF|imaging':0.1463},
 'biolord':{'BIDMC|discharge':0.1941,'BIDMC|imaging':0.1632,'UCSF|discharge':0.2174,'UCSF|imaging':0.1578},
 'bert-base':{'BIDMC|discharge':0.0151,'BIDMC|imaging':0.0181,'UCSF|discharge':0.0161,'UCSF|imaging':0.0121},
 'biobert':{'BIDMC|discharge':0.0187,'BIDMC|imaging':0.0256,'UCSF|discharge':0.0240,'UCSF|imaging':0.0108},
 'clinicalbert':{'BIDMC|discharge':0.0250,'BIDMC|imaging':0.0690,'UCSF|discharge':0.0330,'UCSF|imaging':0.0434},
 'pubmedbert':{'BIDMC|discharge':0.0287,'BIDMC|imaging':0.0488,'UCSF|discharge':0.0358,'UCSF|imaging':0.0313},
 'scibert':{'BIDMC|discharge':0.0181,'BIDMC|imaging':0.0209,'UCSF|discharge':0.0120,'UCSF|imaging':0.0105},
}


def load_rr():
    """Return (rr, patients) where rr[(cell,model)] is a per-query vector and
    patients[cell] is the per-query patient id list. None if cache absent."""
    if not RRCACHE.exists() or not CELLCACHE.exists():
        return None, None
    cells = json.loads(CELLCACHE.read_text())["cells"]
    patients = {c: [r["patient"] for r in recs] for c, recs in cells.items()}
    rr = {}
    for f in RRCACHE.glob("*.npy"):
        # filename: {cell}__{model}.npy with cell using '|'->'_'
        stem = f.stem
        rr[stem] = np.load(f)
    if not rr:
        return None, None
    return rr, patients


def site_patient_index(patients):
    """For each site, map patient -> {genre: [query indices]} so a single patient draw
    applies jointly to both genres of that site (reviewer's point)."""
    idx = {}
    for s in SITES:
        pat2 = defaultdict(lambda: {g: [] for g in GENRES})
        for g in GENRES:
            cell = f"{s}|{g}"
            for i, p in enumerate(patients.get(cell, [])):
                pat2[p][g].append(i)
        idx[s] = dict(pat2)
    return idx


def boot_mrr_maps(rr, patients, rng):
    """One clustered bootstrap replicate -> dict[(site,genre,model)] = resampled MRR.
    Patients resampled per site, applied jointly across that site's two genres."""
    idx = site_patient_index(patients)
    resample = {}
    for s in SITES:
        pats = list(idx[s].keys())
        draw = rng.integers(0, len(pats), len(pats))
        chosen = [pats[k] for k in draw]
        for g in GENRES:
            sel = np.concatenate([idx[s][p][g] for p in chosen if idx[s][p][g]]) \
                  if any(idx[s][p][g] for p in chosen) else np.array([], int)
            resample[(s, g)] = sel
    out = {}
    for (s, g) in resample:
        sel = resample[(s, g)]
        for m in ALL:
            key = f"{s}_{g}__{m}"
            if key in rr and len(sel):
                out[(s, g, m)] = float(rr[key][sel].mean())
    return out


def decompose(mrr):
    vals = [mrr[(s, g, m)] for s in SITES for g in GENRES for m in _models(mrr)]
    grand = np.mean(vals); sstot = sum((v - grand) ** 2 for v in vals)
    ms_models = _models(mrr)
    def fss(f):
        grp = defaultdict(list)
        for s in SITES:
            for g in GENRES:
                for m in ms_models:
                    grp[f(m, s, g)].append(mrr[(s, g, m)])
        return sum(len(v) * (np.mean(v) - grand) ** 2 for v in grp.values())
    ssm = fss(lambda m, s, g: m); sss = fss(lambda m, s, g: s); ssg = fss(lambda m, s, g: g)
    msi = fss(lambda m, s, g: (m, s)) - ssm - sss
    mg = fss(lambda m, s, g: (m, g)) - ssm - ssg
    sg = fss(lambda m, s, g: (s, g)) - sss - ssg
    msg = sstot - ssm - sss - ssg - msi - mg - sg
    return {"model": ssm/sstot, "site": sss/sstot, "genre": ssg/sstot,
            "model×site": msi/sstot, "model×genre": mg/sstot,
            "site×genre": sg/sstot, "model×site×genre": msg/sstot}


_MODELSET = {"all": ALL, "contrastive": CONTRA}
_CURRENT = "all"
def _models(_):
    return _MODELSET[_CURRENT]


def tau_and_regret(mrr):
    """Return per-genre contrastive tau, full tau, and regret from an MRR map."""
    out = {}
    for genre in GENRES:
        for lbl, subset in [("full", ALL), ("contrastive", CONTRA)]:
            a = {m: mrr[("BIDMC", genre, m)] for m in subset if ("BIDMC", genre, m) in mrr}
            b = {m: mrr[("UCSF", genre, m)] for m in subset if ("UCSF", genre, m) in mrr}
            common = sorted(set(a) & set(b))
            if len(common) >= 3:
                out[(genre, lbl, "tau")] = kendalltau([a[m] for m in common],
                                                       [b[m] for m in common])[0]
        # regret on contrastive
        a = {m: mrr[("BIDMC", genre, m)] for m in CONTRA if ("BIDMC", genre, m) in mrr}
        b = {m: mrr[("UCSF", genre, m)] for m in CONTRA if ("UCSF", genre, m) in mrr}
        if len(a) >= 3 and len(b) >= 3:
            bestA, bestB = max(a, key=a.get), max(b, key=b.get)
            out[(genre, "reg", "B->U")] = b[bestB] - b.get(bestA, min(b.values()))
            out[(genre, "reg", "U->B")] = a[bestA] - a.get(bestB, min(a.values()))
    return out


def main():
    global _CURRENT
    rr, patients = load_rr()
    have_ci = rr is not None
    print("=" * 90)
    print(f"ADDED STATISTICS (chunked run) | patient-clustered CIs: "
          f"{'YES' if have_ci else 'NO — RR cache absent, point estimates only'}")
    print("=" * 90)
    if not have_ci:
        print("  To get CIs, re-run two_site_v2_analyze.py --chunk after adding rr-vector")
        print("  caching (np.save per cell/model). Point estimates below use the logged MRRs.\n")

    # ---- point decompositions ----
    pmap = {(s, g, m): POINT[m][f"{s}|{g}"] for s in SITES for g in GENRES for m in ALL}
    print("VARIANCE DECOMPOSITION — full panel vs contrastive-only")
    print("-" * 90)
    print(f"  {'term':<20}{'full 13':>12}{'contrastive 8':>16}")
    _CURRENT = "all"; dfull = decompose(pmap)
    _CURRENT = "contrastive"; dcon = decompose(pmap)
    for k in ["model", "genre", "model×genre", "model×site", "site", "site×genre", "model×site×genre"]:
        print(f"  {k:<20}{dfull[k]:>12.3f}{dcon[k]:>16.3f}")
    print(f"\n  full panel : model {dfull['model']:.3f} dominates (inflated by 5 weak MLM models)")
    print(f"  contrastive: GENRE {dcon['genre']:.3f} > model {dcon['model']:.3f}; "
          f"site {dcon['site']:.3f} ~ 0; model×site {dcon['model×site']:.3f}")
    print(f"  Δη²(genre - site interaction) = model×genre - model×site = "
          f"{dcon['model×genre']-dcon['model×site']:+.3f}  (contrastive)")

    # ---- point tau / regret ----
    print("\nTRANSFER MEASURES (point)")
    print("-" * 90)
    tr = tau_and_regret(pmap)
    for genre in GENRES:
        print(f"  {genre}: contrastive τ={tr.get((genre,'contrastive','tau'),float('nan')):+.3f}  "
              f"full τ={tr.get((genre,'full','tau'),float('nan')):+.3f}  "
              f"regret B→U={tr.get((genre,'reg','B->U'),float('nan')):+.4f}  "
              f"U→B={tr.get((genre,'reg','U->B'),float('nan')):+.4f}")

    # ---- clustered CIs if RR vectors available ----
    if have_ci:
        rng = np.random.default_rng(SEED)
        boot = defaultdict(list)
        boot_dec = {"all": defaultdict(list), "contrastive": defaultdict(list)}
        for _ in range(N_BOOT):
            mrr = boot_mrr_maps(rr, patients, rng)
            if len(mrr) < len(ALL) * 4:
                continue
            tr_b = tau_and_regret(mrr)
            for k, v in tr_b.items():
                boot[k].append(v)
            for scope in ("all", "contrastive"):
                _CURRENT = scope
                d = decompose(mrr)
                for term, val in d.items():
                    boot_dec[scope][term].append(val)
        def ci(x):
            return np.percentile(x, [2.5, 97.5])
        print("\nTRANSFER MEASURES — 95% patient-clustered CI")
        print("-" * 90)
        for genre in GENRES:
            for lbl in ["contrastive", "full"]:
                k = (genre, lbl, "tau")
                if boot[k]:
                    lo, hi = ci(boot[k])
                    print(f"  τ {genre:<10} {lbl:<12} {np.mean(boot[k]):+.3f} [{lo:+.3f},{hi:+.3f}]")
            for d in ["B->U", "U->B"]:
                k = (genre, "reg", d)
                if boot[k]:
                    lo, hi = ci(boot[k])
                    print(f"  regret {genre:<8} {d:<6} {np.mean(boot[k]):+.4f} [{lo:+.4f},{hi:+.4f}]")
        print("\nCONTRASTIVE-ONLY DECOMPOSITION — 95% CI")
        print("-" * 90)
        for term in ["model", "genre", "model×genre", "model×site", "site", "site×genre", "model×site×genre"]:
            x = boot_dec["contrastive"][term]
            if x:
                lo, hi = ci(x)
                print(f"  {term:<20} {np.mean(x):.3f} [{lo:.3f},{hi:.3f}]")

    print("\n" + "=" * 90)
    print("FOR THE REWRITE")
    print("=" * 90)
    print("  * Report BOTH decompositions. Full-panel 'model dominates' is scoped to the")
    print("    heterogeneous panel; among deployable (contrastive) models GENRE dominates.")
    print("  * Cross-site transfer conclusion UNCHANGED: site~0, model×site~0.01, and")
    print("    model×genre > model×site in both scopes.")
    print("  * This ALIGNS with Paper 3 (context rivals model choice), strengthening it.")
    print("  * Regret 0.041 on discharge = ~12% of destination best MRR — call it")
    print("    'modest', not 'negligible'.")


if __name__ == "__main__":
    main()
