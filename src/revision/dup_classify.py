#!/usr/bin/env python3
"""Paper 19, editorial comment 5: exhaustive duplicate-merge classification.

Every group merged by the normalised-text key is classified, not sampled. Two
documents sharing a norm_key can differ only in digits, whitespace, punctuation
or case. Only DIGIT-DIFFERING groups can be the failure mode the editor
describes (distinct clinical content collapsed because the key strips digits);
whitespace/punctuation/case differences mean the documents are textually the
same and merging is correct.

So the exhaustive classification gives an exact upper bound on the false-positive
merge rate, and manual adjudication is needed only on the digit-differing subset,
which is sampled here for reading.

Outputs into RESULTS_DIR:
  duplicate_classification.json   exhaustive counts per cell
  duplicate_digit_groups.csv      sampled digit-differing groups to adjudicate

Usage:
    python3 dup_classify.py
    python3 dup_classify.py --sample 200
    SRC=/path RESULTS_DIR=/path python3 dup_classify.py
"""
import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SRC = Path(os.environ.get("SRC", "/Users/yngve/projects/paper9"))
RESULTS = Path(os.environ.get("RESULTS_DIR", "/Users/yngve/projects/paper13/results"))
SEED = 42

WS = re.compile(r"\s+")
DIGIT = re.compile(r"\d")


def load_module():
    p = SRC / "two_site_v2.py"
    if not p.exists():
        sys.exit("FAIL: %s not found (set SRC)" % p)
    spec = importlib.util.spec_from_file_location("tsv2", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["tsv2"] = m
    spec.loader.exec_module(m)
    return m


def digit_string(s):
    return "".join(DIGIT.findall(s))


def classify(texts, scrub):
    """Classify one merged group of raw texts.

    The normalised key strips digits, punctuation and case and collapses
    whitespace, so members of a group can differ only in those.

      identical          byte-identical raw text
      formatting_only    identical digit sequence; differs in whitespace,
                         punctuation or case only - merging is correct
      date_or_id_only    digits differ, but the texts are identical after the
                         pipeline's own scrub() (which removes dates, times,
                         ID numbers and de-identification placeholders before
                         any query is extracted). The difference lies entirely
                         in content the pipeline never sees, so merging cannot
                         have removed a distinguishable target.
      content_differing  digits still differ after scrub() - distinct clinical
                         content was collapsed. THE false-positive class.
    """
    if len(set(texts)) == 1:
        return "identical"
    if len(set(digit_string(t) for t in texts)) == 1:
        return "formatting_only"
    if len(set(scrub(t) for t in texts)) == 1:
        return "date_or_id_only"
    return "content_differing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=150,
                    help="digit-differing groups to write out for adjudication")
    ap.add_argument("--mimic-note", type=Path, default=None)
    ap.add_argument("--er", type=Path, default=None)
    a = ap.parse_args()

    tsv2 = load_module()
    mimic_note = a.mimic_note or tsv2.MIMIC_NOTE
    er = a.er or tsv2.ER
    print("SRC         = %s" % SRC)
    print("RESULTS_DIR = %s" % RESULTS)

    print("\nloading corpora ...")
    raw = {}
    raw[("BIDMC", "discharge")] = tsv2.load_mimic(mimic_note, "discharge", "discharge.csv.gz", "charttime")
    raw[("BIDMC", "imaging")] = tsv2.load_mimic(mimic_note, "imaging", "radiology.csv.gz", "charttime")
    raw[("UCSF", "discharge")] = tsv2.load_er(er, "Discharge_Summary_Text", "Discharge_Summary_Year")
    raw[("UCSF", "imaging")] = tsv2.load_er(er, "Imaging_Text", "Imaging_Year")

    print("\n" + "=" * 100)
    print("EXHAUSTIVE CLASSIFICATION OF EVERY MERGED GROUP (comment 5)")
    print("=" * 100)
    print("\n  %-18s%9s%8s%10s%11s%12s%13s%12s"
          % ("cell", "raw", "groups", "docs_lost", "identical", "formatting",
             "date_or_id", "CONTENT"))
    print("  " + "-" * 98)

    out, rows = {}, []
    rs = np.random.RandomState(SEED)
    for k, df in raw.items():
        texts = df["text"].astype(str).tolist()
        pats = df["patient"].astype(str).tolist()
        groups = defaultdict(list)
        for i, t in enumerate(texts):
            groups[tsv2.norm_key(t)].append(i)
        merged = {kk: v for kk, v in groups.items() if len(v) > 1}

        cls = Counter()
        lost = Counter()
        digit_keys = []
        for kk, idxs in merged.items():
            c = classify([texts[i] for i in idxs], tsv2.ts.scrub)
            cls[c] += 1
            lost[c] += len(idxs) - 1
            if c == "content_differing":
                digit_keys.append(kk)

        cell = "%s|%s" % k
        total_lost = sum(lost.values())
        out[cell] = {
            "raw": len(texts), "groups": len(merged), "docs_lost": int(total_lost),
            "groups_by_class": {c: int(n) for c, n in cls.items()},
            "docs_lost_by_class": {c: int(n) for c, n in lost.items()},
            "false_positive_docs": int(lost["content_differing"]),
            "false_positive_pct_of_corpus":
                100.0 * lost["content_differing"] / max(len(texts), 1),
            "false_positive_pct_of_merges":
                100.0 * lost["content_differing"] / max(total_lost, 1),
            "correct_merge_pct_of_groups":
                100.0 * (cls["identical"] + cls["formatting_only"] + cls["date_or_id_only"])
                / max(len(merged), 1),
        }
        print("  %-18s%9d%8d%10d%11d%12d%13d%12d"
              % (cell, len(texts), len(merged), total_lost, cls["identical"],
                 cls["formatting_only"], cls["date_or_id_only"],
                 cls["content_differing"]))

        if digit_keys:
            take = min(max(a.sample // 4, 1), len(digit_keys))
            for kk in [digit_keys[i] for i in rs.choice(len(digit_keys), take, replace=False)]:
                idxs = merged[kk][:2]
                t1, t2 = texts[idxs[0]], texts[idxs[1]]
                rows.append({
                    "cell": cell, "group_size": len(merged[kk]),
                    "same_patient": pats[idxs[0]] == pats[idxs[1]],
                    "digits_1": sum(c.isdigit() for c in t1),
                    "digits_2": sum(c.isdigit() for c in t2),
                    "digit_string_1": "".join(DIGIT.findall(t1))[:120],
                    "digit_string_2": "".join(DIGIT.findall(t2))[:120],
                    "len_1": len(t1), "len_2": len(t2),
                    "text_1": t1[:1200], "text_2": t2[:1200],
                })

    print("\n  Only content_differing groups are false-positive merges. formatting_only")
    print("  members carry identical digits; date_or_id_only members are identical after")
    print("  the pipeline's own scrub(), which removes dates, times and ID numbers before")
    print("  any query is extracted - so that difference is invisible to the analysis.")
    print("\n  %-18s%18s%18s%20s"
          % ("cell", "false merges", "as % of corpus", "correct merges %"))
    for cell, v in out.items():
        print("  %-18s%18d%17.4f%%%19.1f%%"
              % (cell, v["false_positive_docs"], v["false_positive_pct_of_corpus"],
                 v["correct_merge_pct_of_groups"]))
    tot_fp = sum(v["false_positive_docs"] for v in out.values())
    tot_lost = sum(v["docs_lost"] for v in out.values())
    tot_groups = sum(v["groups"] for v in out.values())
    tot_content = sum(v["groups_by_class"].get("content_differing", 0) for v in out.values())
    print("\n  OVERALL: %d of %d merged groups (%.1f%%) differ in content after scrub;"
          % (tot_content, tot_groups, 100.0 * tot_content / max(tot_groups, 1)))
    print("           %d of %d merged-away documents (%.1f%%) are false-positive merges."
          % (tot_fp, tot_lost, 100.0 * tot_fp / max(tot_lost, 1)))

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "duplicate_classification.json").write_text(json.dumps(out, indent=1),
                                                           encoding="utf-8")
    dest = RESULTS / "duplicate_digit_groups.csv"
    if rows:
        with open(dest, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("\n  %d content-differing groups written for spot-checking" % len(rows))
    else:
        print("\n  no content-differing groups: every merge is identical, formatting-only, or\n  differs solely in dates/IDs that scrub() removes before query extraction")
    print("\nWritten: %s" % (RESULTS / "duplicate_classification.json"))
    print("Written: %s" % dest)


if __name__ == "__main__":
    main()
