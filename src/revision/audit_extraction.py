#!/usr/bin/env python3
"""Paper 19 revision: extraction attrition audit (comments 9 and 15) and
duplicate-merge audit sample (comment 5).

Reuses the project's own functions by importing two_site_v2.py from SRC, so the
dedup rule, pool sampling, query extractor and target-removal logic are the
originals, not reimplementations. Nothing is written back to the cells cache.

Outputs, into RESULTS_DIR:
  extraction_audit.json          per-criterion attrition + sentence-level filter counts
  duplicate_audit_sample.csv     merged duplicate pairs to adjudicate by hand

Usage:
    python3 audit_extraction.py                 # full run (slow: reads MIMIC radiology)
    python3 audit_extraction.py --smoke 20000   # subsample raw rows per cell, quick check
    SRC=/path RESULTS_DIR=/path python3 audit_extraction.py
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(os.environ.get("SRC", "/Users/yngve/projects/paper9"))
RESULTS = Path(os.environ.get("RESULTS_DIR", "/Users/yngve/projects/paper13/results"))
SEED = 42
N_DUP_SAMPLE = 60          # duplicate pairs to write out for adjudication


def load_module():
    p = SRC / "two_site_v2.py"
    if not p.exists():
        sys.exit("FAIL: %s not found (set SRC)" % p)
    if not (SRC / "two_site.py").exists():
        sys.exit("FAIL: two_site.py not found next to two_site_v2.py in %s" % SRC)
    spec = importlib.util.spec_from_file_location("tsv2", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["tsv2"] = m
    spec.loader.exec_module(m)
    return m


def dedup_with_groups(df, norm_key):
    """Same rule as tsv2.dedup (keep first by normalised key), but retain the groups."""
    df = df.copy()
    df["k"] = df["text"].astype(str).map(norm_key)
    groups = defaultdict(list)
    for idx, k in zip(df.index, df["k"]):
        groups[k].append(idx)
    kept = df.drop_duplicates(subset=["k"])
    merged = {k: v for k, v in groups.items() if len(v) > 1}
    return kept.drop(columns=["k"]), len(df) - len(kept), merged, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0,
                    help="subsample this many raw rows per cell for a fast check")
    ap.add_argument("--mimic-note", type=Path, default=None)
    ap.add_argument("--er", type=Path, default=None)
    a = ap.parse_args()

    tsv2 = load_module()
    ts = tsv2.ts
    mimic_note = a.mimic_note or tsv2.MIMIC_NOTE
    er = a.er or tsv2.ER
    print("SRC         = %s" % SRC)
    print("RESULTS_DIR = %s" % RESULTS)
    print("mimic note  = %s" % mimic_note)
    print("er csv      = %s" % er)
    if a.smoke:
        print("SMOKE MODE: %d raw rows per cell - counts are NOT the real attrition" % a.smoke)

    print("\nloading corpora ...")
    raw = {}
    raw[("BIDMC", "discharge")] = tsv2.load_mimic(mimic_note, "discharge", "discharge.csv.gz", "charttime")
    raw[("BIDMC", "imaging")] = tsv2.load_mimic(mimic_note, "imaging", "radiology.csv.gz", "charttime")
    raw[("UCSF", "discharge")] = tsv2.load_er(er, "Discharge_Summary_Text", "Discharge_Summary_Year")
    raw[("UCSF", "imaging")] = tsv2.load_er(er, "Imaging_Text", "Imaging_Year")
    if a.smoke:
        rs = np.random.RandomState(SEED)
        for k in raw:
            df = raw[k]
            if len(df) > a.smoke:
                raw[k] = df.iloc[rs.choice(len(df), a.smoke, replace=False)].reset_index(drop=True)

    # ---------- comment 5: duplicate merge audit ----------
    print("\n" + "=" * 92)
    print("DUPLICATE MERGE AUDIT (comment 5)")
    print("=" * 92)
    dd, dup_rows, dup_stats = {}, [], {}
    rs_dup = np.random.RandomState(SEED)
    for k, df in raw.items():
        kept, dropped, merged, keyed = dedup_with_groups(df, tsv2.norm_key)
        dd[k] = kept
        sizes = Counter(len(v) for v in merged.values())
        dup_stats["%s|%s" % k] = {
            "raw": int(len(df)), "kept": int(len(kept)), "dropped": int(dropped),
            "merged_groups": len(merged),
            "group_size_distribution": {str(s): int(n) for s, n in sorted(sizes.items())},
            "max_group": max((len(v) for v in merged.values()), default=0),
        }
        print("  %-18s raw %8d -> kept %8d  (dropped %6d in %5d groups, max group %d)"
              % ("%s/%s" % k, len(df), len(kept), dropped, len(merged), 
                 max((len(v) for v in merged.values()), default=0)))
        # sample groups for manual adjudication, weighted to this cell
        keys = list(merged)
        if keys:
            take = min(N_DUP_SAMPLE // 4, len(keys))
            for kk in [keys[i] for i in rs_dup.choice(len(keys), take, replace=False)]:
                idxs = merged[kk][:2]
                t1 = str(keyed.loc[idxs[0], "text"])
                t2 = str(keyed.loc[idxs[1], "text"])
                dup_rows.append({
                    "cell": "%s|%s" % k,
                    "group_size": len(merged[kk]),
                    "same_patient": str(keyed.loc[idxs[0], "patient"]) == str(keyed.loc[idxs[1], "patient"]),
                    "len_1": len(t1), "len_2": len(t2),
                    "identical_raw": t1 == t2,
                    "digits_1": sum(c.isdigit() for c in t1),
                    "digits_2": sum(c.isdigit() for c in t2),
                    "text_1": t1[:1500], "text_2": t2[:1500],
                })

    ident = sum(1 for r in dup_rows if r["identical_raw"])
    samepat = sum(1 for r in dup_rows if r["same_patient"])
    print("\n  sampled %d merged pairs: %d identical raw text, %d same patient"
          % (len(dup_rows), ident, samepat))
    print("  pairs that are NOT identical raw text are the ones to read: those are merges")
    print("  driven by digit removal. Adjudicate them in the CSV.")

    # ---------- comments 9 and 15: extraction attrition ----------
    print("\n" + "=" * 92)
    print("EXTRACTION ATTRITION BY CRITERION (comments 9 and 15)")
    print("=" * 92)
    g = np.random.RandomState(SEED)          # same RNG order as build_cells
    pool = min(len(v) for v in dd.values())
    print("  pool = %d per cell\n" % pool)
    attrition = {}
    for k, df in dd.items():
        ix = g.choice(len(df), pool, replace=False) if len(df) > pool else np.arange(len(df))
        sub = df.iloc[ix].reset_index(drop=True)
        dfreq = ts.build_df(sub["text"].tolist())

        c = Counter()
        sent_total = sent_short = sent_boiler = sent_noverb = sent_df = 0
        for _, row in sub.iterrows():
            text = row["text"]
            sents = ts.sentences(text)
            sent_total += len(sents)
            for s in sents:
                if len(s.split()) < 10:
                    sent_short += 1
                elif ts.is_boilerplate(s):
                    sent_boiler += 1
                elif not ts.VERBISH.search(s):
                    sent_noverb += 1
                elif dfreq.get(ts._key(s), 1) > ts.MAX_DF_DOCS:
                    sent_df += 1
            q = ts.narrative_query(text, dfreq)
            if not q:
                c["no_query"] += 1
                if not sents:
                    c["no_query_no_sentences"] += 1
                else:
                    c["no_query_no_prose_run"] += 1
                continue
            target = tsv2.remove_query_from_target(text, q)
            if len(target) < 100:
                c["target_under_100_chars"] += 1
                continue
            qk = tsv2.norm_key(q)[:60]
            if qk and qk in tsv2.norm_key(target):
                c["residual_leak"] += 1
                continue
            c["usable"] += 1

        cell = "%s|%s" % k
        attrition[cell] = {
            "pool": int(pool), "usable": int(c["usable"]),
            "no_query": int(c["no_query"]),
            "no_query_no_sentences": int(c["no_query_no_sentences"]),
            "no_query_no_prose_run": int(c["no_query_no_prose_run"]),
            "target_under_100_chars": int(c["target_under_100_chars"]),
            "residual_leak": int(c["residual_leak"]),
            "sentences_total": int(sent_total),
            "sentences_rejected_short": int(sent_short),
            "sentences_rejected_boilerplate": int(sent_boiler),
            "sentences_rejected_no_verb": int(sent_noverb),
            "sentences_rejected_df_gt_3": int(sent_df),
        }
        print("  %-18s usable %5d/%5d (%5.1f%%)  | no_query %5d  short_target %5d  leak %4d"
              % (cell, c["usable"], pool, 100.0 * c["usable"] / pool,
                 c["no_query"], c["target_under_100_chars"], c["residual_leak"]))
        print("      sentences %7d | rejected: <10w %6d  boilerplate %6d  no-verb %6d  df>3 %6d (%.1f%% of eligible)"
              % (sent_total, sent_short, sent_boiler, sent_noverb, sent_df,
                 100.0 * sent_df / max(sent_total - sent_short, 1)))

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "extraction_audit.json").write_text(
        json.dumps({"smoke": a.smoke, "pool": int(pool),
                    "attrition": attrition, "duplicates": dup_stats}, indent=1),
        encoding="utf-8")
    dest = RESULTS / "duplicate_audit_sample.csv"
    if dup_rows:
        with open(dest, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dup_rows[0]))
            w.writeheader()
            w.writerows(dup_rows)
    print("\nWritten: %s" % (RESULTS / "extraction_audit.json"))
    print("Written: %s  (adjudicate rows where identical_raw is False)" % dest)


if __name__ == "__main__":
    main()
