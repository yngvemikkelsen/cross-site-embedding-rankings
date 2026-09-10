"""Secondary retrieval metrics for the revision: Hit@1 and Hit@10 (a.k.a. Recall@1 / Recall@10
for known-item retrieval, where each query has exactly ONE relevant target).

WHY THIS IS EXACT, NOT AN APPROXIMATION
---------------------------------------
The pipeline already caches a per-query reciprocal-rank vector rr for every (cell, model):
two_site_v2_analyze.py computes rr[i] = 1/(rank of target i) if the target is within the
top-K (K=10), else 0 (see rr_vector / rr_vector_chunked). Therefore, with one relevant doc
per query:
    Hit@1  = fraction of queries with rr == 1.0     (target ranked first)
    Hit@10 = fraction of queries with rr  > 0.0     (target within the top 10)
No re-embedding is required; this reads the SAME rr cache used for MRR@10.

It reuses the identical patient-clustered bootstrap as two_site_v2_stats.py: patients are
resampled per SITE and the same draw is applied jointly to that site's two genres, because a
patient can appear in both genres.

USAGE
-----
    python two_site_v2_hits.py                # best-chunk (chunked) rr cache  [primary]
    python two_site_v2_hits.py --single       # single-window rr cache
Outputs a table of Hit@1 / Hit@10 with 95% CIs per model and per genre, plus the
contrastive-panel (8-model) means, and writes two_site_v2_hits[_chunk].json.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

RESULTS  = Path.home() / "Projects" / "paper13" / "results"
CELLCACHE = RESULTS / "two_site_v2_cells.json"
SEED, N_BOOT = 42, 2000
CONTRA = ["bge", "gte", "e5", "nomic", "mpnet", "minilm", "medcpt", "biolord"]
MLM    = ["bert-base", "biobert", "clinicalbert", "pubmedbert", "scibert"]
ALL    = CONTRA + MLM
SITES, GENRES = ["BIDMC", "UCSF"], ["discharge", "imaging"]


def load_rr(rrcache):
    if not CELLCACHE.exists() or not rrcache.exists():
        raise SystemExit(f"missing cache: {CELLCACHE} or {rrcache}\n"
                         f"  run two_site_v2_analyze.py first (it writes the rr cache).")
    cells = json.loads(CELLCACHE.read_text())["cells"]
    patients = {c: [r["patient"] for r in recs] for c, recs in cells.items()}
    rr = {}
    for f in rrcache.glob("*.npy"):
        rr[f.stem] = np.load(f)      # stem = "{SITE}_{genre}__{model}"
    if not rr:
        raise SystemExit(f"no rr vectors found in {rrcache}")
    return rr, patients


def site_patient_index(patients):
    """patient -> {genre: [query indices]} per site, so one draw applies to both genres."""
    idx = {}
    for s in SITES:
        pat2 = defaultdict(lambda: {g: [] for g in GENRES})
        for g in GENRES:
            for i, p in enumerate(patients.get(f"{s}|{g}", [])):
                pat2[p][g].append(i)
        idx[s] = dict(pat2)
    return idx


def hit_stats(vec):
    """point Hit@1 and Hit@10 from an rr vector."""
    return float(np.mean(vec == 1.0)), float(np.mean(vec > 0.0))


def clustered_boot(rr, patients, models, statfn):
    """Patient-clustered bootstrap for a scalar statistic averaged over `models`,
    reported per (site, genre). Returns dict[(s,g)] -> (point, lo, hi)."""
    idx = site_patient_index(patients)
    rng = np.random.RandomState(SEED)
    # point estimate
    point = {}
    for s in SITES:
        for g in GENRES:
            vals = [statfn(rr[f"{s}_{g}__{m}"]) for m in models if f"{s}_{g}__{m}" in rr]
            point[(s, g)] = float(np.mean(vals)) if vals else float("nan")
    # bootstrap
    boot = {k: [] for k in point}
    for _ in range(N_BOOT):
        for s in SITES:
            pats = list(idx[s].keys())
            draw = rng.randint(0, len(pats), len(pats))
            chosen = [pats[k] for k in draw]
            for g in GENRES:
                sel = (np.concatenate([idx[s][p][g] for p in chosen if idx[s][p][g]])
                       if any(idx[s][p][g] for p in chosen) else np.array([], int))
                if not len(sel):
                    continue
                vals = [statfn(rr[f"{s}_{g}__{m}"][sel]) for m in models if f"{s}_{g}__{m}" in rr]
                if vals:
                    boot[(s, g)].append(np.mean(vals))
    out = {}
    for k in point:
        b = np.array(boot[k])
        if len(b):
            out[k] = (point[k], float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
        else:
            out[k] = (point[k], float("nan"), float("nan"))
    return out


def per_model_table(rr):
    rows = []
    for m in ALL:
        for s in SITES:
            for g in GENRES:
                key = f"{s}_{g}__{m}"
                if key in rr:
                    h1, h10 = hit_stats(rr[key])
                    rows.append((m, s, g, h1, h10, len(rr[key])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true",
                    help="use single-window rr cache (default: best-chunk/chunked)")
    a = ap.parse_args()
    rrcache = RESULTS / ("two_site_v2_rr" if a.single else "two_site_v2_rr_chunk")
    scoring = "single-window" if a.single else "best-chunk"
    rr, patients = load_rr(rrcache)

    print(f"\n=== Hit@1 / Hit@10  ({scoring} scoring; {N_BOOT} patient-clustered bootstrap) ===\n")

    # Contrastive-panel means with CIs, per (site, genre) and pooled per genre.
    for label, models in [("Contrastive (8)", CONTRA), ("Full panel (13)", ALL)]:
        h1 = clustered_boot(rr, patients, models, lambda v: float(np.mean(v == 1.0)))
        h10 = clustered_boot(rr, patients, models, lambda v: float(np.mean(v > 0.0)))
        print(f"--- {label} ---")
        print(f"  {'cell':<18}{'Hit@1 (95% CI)':<26}{'Hit@10 (95% CI)'}")
        for s in SITES:
            for g in GENRES:
                p1, lo1, hi1 = h1[(s, g)]
                p10, lo10, hi10 = h10[(s, g)]
                print(f"  {s+'/'+g:<18}{p1:.3f} ({lo1:.3f}-{hi1:.3f})     {p10:.3f} ({lo10:.3f}-{hi10:.3f})")
        print()

    # Per-model point values (no CI) for the appendix / supplement
    print("--- per-model point Hit@1 / Hit@10 ---")
    print(f"  {'model':<14}{'site':<7}{'genre':<11}{'Hit@1':<8}{'Hit@10':<8}N")
    rows = per_model_table(rr)
    for m, s, g, h1, h10, n in rows:
        print(f"  {m:<14}{s:<7}{g:<11}{h1:<8.3f}{h10:<8.3f}{n}")

    # Save JSON
    out = {
        "scoring": scoring, "n_boot": N_BOOT, "seed": SEED,
        "contrastive": {f"{s}|{g}": {
            "hit1": clustered_boot(rr, patients, CONTRA, lambda v: float(np.mean(v == 1.0)))[(s, g)],
            "hit10": clustered_boot(rr, patients, CONTRA, lambda v: float(np.mean(v > 0.0)))[(s, g)],
        } for s in SITES for g in GENRES},
        "per_model": [
            {"model": m, "site": s, "genre": g, "hit1": h1, "hit10": h10, "n": n}
            for (m, s, g, h1, h10, n) in rows
        ],
    }
    outfile = RESULTS / ("two_site_v2_hits" + ("" if a.single else "_chunk") + ".json")
    outfile.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outfile}")


if __name__ == "__main__":
    main()
