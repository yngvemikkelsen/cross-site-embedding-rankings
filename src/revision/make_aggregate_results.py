#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Regenerate results/aggregate_results.json from the corrected run.

Reads the JSON products of the analysis, not the manuscript, so the deposited
aggregate file cannot drift from the code:

    $RESULTS_DIR/propagation_pack.json        Tables 1-5 + paired eta-squared contrasts
    $RESULTS_DIR/single_chunk_multiseed.json  20-seed single-random-chunk sensitivity

Values that come from the descriptive robustness scripts (mean-pooled and
mean-centered rescoring, anisotropy) are not emitted as JSON by those scripts and
are supplied on the command line or left at their defaults below; they are
printed for confirmation so they can be checked against the run logs.

Usage:
    python3 make_aggregate_results.py -o results/aggregate_results.json
    RESULTS_DIR=/path python3 make_aggregate_results.py
"""
import argparse
import json
import os
from pathlib import Path

RESULTS = Path(os.environ.get("RESULTS_DIR", "/Users/yngve/projects/paper13/results"))
CONTRA = ["bge", "gte", "e5", "nomic", "mpnet", "minilm", "medcpt", "biolord"]
MLM = ["bert-base", "biobert", "clinicalbert", "pubmedbert", "scibert"]
CELLS = ["BIDMC_discharge", "BIDMC_imaging", "UCSF_discharge", "UCSF_imaging"]
TERMKEY = {"model": "model", "genre": "genre", "model x genre": "model_x_genre",
           "model x site x genre": "model_x_site_x_genre", "model x site": "model_x_site",
           "site x genre": "site_x_genre", "site": "site"}

# From chunk_sensitivity2.py and centering_check.py logs (descriptive rescoring).
MEAN_POOLED = {"genre": 0.213, "model": 0.390, "site": 0.172, "model_x_site": 0.029,
               "tau_discharge": 0.71, "tau_imaging": 0.71}
MEAN_CENTERED = {"genre": 0.356, "model": 0.404, "site": 0.002, "model_x_site": 0.005,
                 "tau_discharge": 0.86, "tau_imaging": 0.57}
# From anisotropy_check.py: mean off-diagonal cosine by model, averaged over cells.
ANISOTROPY = {"e5": 0.835, "gte": 0.793, "medcpt": 0.765, "nomic": 0.689, "bge": 0.668,
              "mpnet": 0.457, "minilm": 0.425, "biolord": 0.302}
CHUNK_MEDIAN = {"BIDMC_discharge": 7, "UCSF_discharge": 10,
                "BIDMC_imaging": 1, "UCSF_imaging": 1}
NULL_SLOPE_RANGE = [0.013, 0.075]


def r3(x):
    return round(float(x), 3)


def r4(x):
    return round(float(x), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="aggregate_results.json")
    a = ap.parse_args()

    pack_p = RESULTS / "propagation_pack.json"
    seed_p = RESULTS / "single_chunk_multiseed.json"
    for p in (pack_p, seed_p):
        if not p.exists():
            raise SystemExit("FAIL: %s not found. Run src/revision/propagation_pack.py and "
                             "src/revision/single_chunk_multiseed.py first." % p)
    pack = json.loads(pack_p.read_text())
    seeds = json.loads(seed_p.read_text())

    def mrr(m, cell):
        s, g = cell.split("_")
        return pack["table1_mrr"]["%s|%s|%s" % (s, g, m)]

    def hit(kind, m, cell):
        s, g = cell.split("_")
        return pack["table2_%s" % kind]["%s|%s|%s" % (s, g, m)]

    t2 = pack["table2_as_printed"]
    tau = pack["table3_tau"]
    tci = pack["table3_ci"]
    dcon, dfull = pack["table5_contrastive"], pack["table5_full"]
    cci = pack["table5_contrastive_ci"]
    seed_dec = seeds["decomposition"]["contrastive"]
    seed_tau = seeds["tau"]["contrastive"]

    out = {
        "_description": ("Aggregate, non-identifying results reproducing every table "
                         "(Table 1 MRR@10; Table 2 Hit@1/Hit@10; Tables 3-6 as in the "
                         "manuscript) in the manuscript. Contains only derived statistics "
                         "(MRR, tau, eta-squared, counts). No clinical note text, no patient "
                         "identifiers. Regenerated for release 2.0.0 after the per-chunk "
                         "instruction-prefix correction; supersedes the values in 1.1.2."),
        "design": {
            "sites": {"BIDMC": "MIMIC-IV-Note v2.2", "UCSF": "ER-Reason v1.0.0"},
            "genres": ["discharge", "imaging"],
            "n_per_cell": 1235,
            "top_k": 10,
            "chunking": {
                "window_tokens": 510,
                "stride_tokens": 384,
                "overlap_tokens": 126,
                "aggregation": "max-over-chunk cosine",
                "document_prefix": ("tokenized separately and prepended to every window; "
                                    "window budget reduced by the prefix length, so content "
                                    "overlap is 124 tokens for E5 and 122 for Nomic-embed"),
            },
            "contrastive_models": CONTRA,
            "mlm_models": MLM,
        },
        "table1_mrr_at_10": {
            "_note": ("MRR@10, query-excluded, chunked. Point values; 95% patient-clustered "
                      "CIs in table1_mrr_at_10_ci."),
            **{m: {c: r3(mrr(m, c)) for c in CELLS} for m in CONTRA + MLM},
        },
        "table1_mrr_at_10_ci": {
            m: {c: [r3(v) for v in pack["table1_ci"]["%s|%s|%s" % (*c.split("_"), m)]]
                for c in CELLS} for m in CONTRA + MLM
        },
        "table2_hit_at_1_and_10": {
            "_note": ("Hit@1 and Hit@10 (=Recall@k for known-item retrieval; one relevant "
                      "target per query). Mean over the 8 contrastive models, 95% "
                      "patient-clustered CI (2000 resamples), best-chunk scoring."),
            "hit_at_1_contrastive_8": {
                c: [r3(t2["%s|%s" % tuple(c.split("_"))]["hit1"]),
                    *[r3(v) for v in t2["%s|%s" % tuple(c.split("_"))]["hit1_ci"]]]
                for c in CELLS},
            "hit_at_10_contrastive_8": {
                c: [r3(t2["%s|%s" % tuple(c.split("_"))]["hit10"]),
                    *[r3(v) for v in t2["%s|%s" % tuple(c.split("_"))]["hit10_ci"]]]
                for c in CELLS},
            "per_model_point": {
                m: {c: {"hit_at_1": r3(hit("hit1", m, c)),
                        "hit_at_10": r3(hit("hit10", m, c))} for c in CELLS}
                for m in CONTRA + MLM},
        },
        "table3_cross_site_tau": {
            "_note": "Kendall tau between site rankings within genre; point + 95% patient-clustered CI.",
            "discharge_contrastive": {"tau": r3(tau["discharge|contrastive"]),
                                      "ci": [r3(v) for v in tci["discharge|contrastive"]]},
            "discharge_full13": {"tau": r3(tau["discharge|full"]),
                                 "ci": [r3(v) for v in tci["discharge|full"]]},
            "imaging_contrastive": {"tau": r3(tau["imaging|contrastive"]),
                                    "ci": [r3(v) for v in tci["imaging|contrastive"]]},
            "imaging_full13": {"tau": r3(tau["imaging|full"]),
                               "ci": [r3(v) for v in tci["imaging|full"]]},
            "top5_overlap": {"discharge": 1.0, "imaging": 0.8},
        },
        "table4_selection_regret": {
            "_note": ("Observed point regret with patient-clustered bootstrap 95% CI. Best "
                      "model reselected within each replicate. Computed from unrounded MRR."),
            "discharge_BIDMC_to_UCSF": {
                "source_best": "nomic", "dest_best": "gte",
                "observed": r3(tau["discharge|regret_BU"]),
                "ci": [r3(v) for v in tci["discharge|regret_BU"]]},
            "discharge_UCSF_to_BIDMC": {
                "source_best": "gte", "dest_best": "nomic",
                "observed": r3(tau["discharge|regret_UB"]),
                "ci": [r3(v) for v in tci["discharge|regret_UB"]]},
            "imaging_BIDMC_to_UCSF": {
                "source_best": "gte", "dest_best": "gte",
                "observed": r3(tau["imaging|regret_BU"]),
                "ci": [r3(v) for v in tci["imaging|regret_BU"]]},
            "imaging_UCSF_to_BIDMC": {
                "source_best": "gte", "dest_best": "gte",
                "observed": r3(tau["imaging|regret_UB"]),
                "ci": [r3(v) for v in tci["imaging|regret_UB"]]},
        },
        "table5_variance_decomposition": {
            "_note": ("eta-squared of cell-level mean MRR@10, not of individual query "
                      "outcomes. Full panel = point estimates; contrastive = point + 95% "
                      "patient-clustered CI, with the full factorial decomposition "
                      "recalculated within each bootstrap replicate."),
            "full_panel_13": {TERMKEY[k]: r3(v) for k, v in dfull.items()},
            "contrastive_8": {TERMKEY[k]: {"eta2": r3(dcon[k]),
                                           "ci": [r3(x) for x in cci[k]]} for k in dcon},
            "paired_contrast_model_x_genre_minus_model_x_site": {
                "contrastive_8": {"delta_eta2": r3(pack["delta_eta2"]["point"]),
                                  "ci": [r3(v) for v in pack["delta_eta2"]["ci"]]},
                "full_panel_13": {"delta_eta2": r3(pack["delta_eta2_full_panel"]["point"]),
                                  "ci": [r3(v) for v in pack["delta_eta2_full_panel"]["ci"]]},
            },
        },
        "table6_scoring_sensitivity": {
            "_note": ("Contrastive-panel eta-squared and cross-site tau across scoring "
                      "choices. Row 1 = primary (patient-clustered, = Table 5). Rows 2-4 are "
                      "descriptive sensitivities re-scored from cached chunk embeddings using "
                      "unweighted cell means and do not carry the bootstrap uncertainty of the "
                      "primary analysis; compare qualitatively rather than inferentially."),
            "best_chunk_primary": {
                "genre": r3(dcon["genre"]), "model": r3(dcon["model"]),
                "site": r3(dcon["site"]), "model_x_site": r3(dcon["model x site"]),
                "tau_discharge": round(tau["discharge|contrastive"], 2),
                "tau_imaging": round(tau["imaging|contrastive"], 2)},
            "mean_pooled": MEAN_POOLED,
            "single_random_chunk_20_seeds": {
                "genre": r3(seed_dec["genre"][0]), "model": r3(seed_dec["model"][0]),
                "site": r3(seed_dec["site"][0]),
                "model_x_site": r3(seed_dec["model x site"][0]),
                "tau_discharge": round(seed_tau["discharge"][0], 2),
                "tau_imaging": round(seed_tau["imaging"][0], 2),
                "sd_across_seeds": {"genre": r3(seed_dec["genre"][1]),
                                    "model": r3(seed_dec["model"][1]),
                                    "site": r3(seed_dec["site"][1]),
                                    "model_x_site": r3(seed_dec["model x site"][1])},
                "n_seeds": seeds["seeds"]},
            "mean_centered_best_chunk": MEAN_CENTERED,
        },
        "robustness_chunk_count": {
            "_note": ("Null order-statistic slope: E[max-sim | mismatched query] vs chunk "
                      "count, measured on real embeddings."),
            "chunk_median_by_cell": CHUNK_MEDIAN,
            "null_slope_range_across_models": NULL_SLOPE_RANGE,
        },
        "robustness_anisotropy": {
            "_note": ("Mean off-diagonal cosine between unrelated document chunks, averaged "
                      "over the four cells (higher = more anisotropic). Describes embedding "
                      "geometry, not retrieval performance."),
            "by_model": ANISOTROPY,
            "mean_centering_removes_floor_to": 0.0,
        },
    }

    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("wrote %s" % a.out)
    print("\nCheck these against the run logs (not derived from JSON products):")
    print("  mean-pooled      :", MEAN_POOLED)
    print("  mean-centered    :", MEAN_CENTERED)
    print("  anisotropy       :", ANISOTROPY)
    print("  chunk medians    :", CHUNK_MEDIAN)
    print("  null slope range :", NULL_SLOPE_RANGE)
    print("\nKey regenerated values:")
    print("  tau imaging contrastive : %.3f" % out["table3_cross_site_tau"]["imaging_contrastive"]["tau"])
    print("  contrastive model eta2  : %.3f" % out["table5_variance_decomposition"]["contrastive_8"]["model"]["eta2"])
    print("  contrastive genre eta2  : %.3f" % out["table5_variance_decomposition"]["contrastive_8"]["genre"]["eta2"])
    print("  model x site eta2       : %.3f" % out["table5_variance_decomposition"]["contrastive_8"]["model_x_site"]["eta2"])


if __name__ == "__main__":
    main()
