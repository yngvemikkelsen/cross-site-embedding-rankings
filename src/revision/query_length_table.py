#!/usr/bin/env python3
"""Paper 19, editorial comment 19: query length distributions across the four cells.

Reports word and character length per cell (mean, SD, median, IQR, min, max) plus
the between-cell spread, to test whether systematically longer queries at one site
or genre could confound the genre main effect.

Usage:
    python3 query_length_table.py
    RESULTS_DIR=/path/to/results python3 query_length_table.py
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(os.environ.get("RESULTS_DIR", "/Users/yngve/projects/paper13/results"))
CELLCACHE = RESULTS / "two_site_v2_cells.json"
WORD = re.compile(r"\S+")


def stats(vals):
    a = np.asarray(vals, dtype=float)
    q1, q3 = np.percentile(a, [25, 75])
    return {"n": len(a), "mean": a.mean(), "sd": a.std(ddof=1), "median": np.median(a),
            "q1": q1, "q3": q3, "min": a.min(), "max": a.max(), "var": a.var(ddof=1)}


def main():
    if not CELLCACHE.exists():
        sys.exit("FAIL: no cells file at %s" % CELLCACHE)
    cells = json.loads(CELLCACHE.read_text())["cells"]

    print("=" * 96)
    print("QUERY LENGTH BY CELL (editorial comment 19)")
    print("RESULTS_DIR = %s" % RESULTS)
    print("=" * 96)

    out = {}
    for unit, fn, label in [("words", lambda t: len(WORD.findall(t)), "WORDS"),
                            ("chars", len, "CHARACTERS")]:
        print("\n%s" % label)
        print("-" * 96)
        print("  %-20s%7s%9s%8s%9s%9s%9s%7s%7s"
              % ("cell", "n", "mean", "SD", "var", "median", "IQR", "min", "max"))
        per_cell = {}
        for cell in sorted(cells):
            vals = [fn(r["query"]) for r in cells[cell]]
            s = stats(vals)
            per_cell[cell] = s
            print("  %-20s%7d%9.1f%8.1f%9.1f%9.1f%5.0f-%-3.0f%7.0f%7.0f"
                  % (cell, s["n"], s["mean"], s["sd"], s["var"], s["median"],
                     s["q1"], s["q3"], s["min"], s["max"]))
        means = {c: per_cell[c]["mean"] for c in per_cell}
        hi, lo = max(means, key=means.get), min(means, key=means.get)
        spread = means[hi] - means[lo]
        pooled_sd = np.mean([per_cell[c]["sd"] for c in per_cell])
        print("\n  between-cell spread: %.1f %s (%s %.1f vs %s %.1f)"
              % (spread, unit, hi, means[hi], lo, means[lo]))
        print("  spread / pooled SD  = %.2f   (small values argue against a length confound)"
              % (spread / pooled_sd if pooled_sd else float("nan")))

        by_genre = {}
        by_site = {}
        for cell, s in per_cell.items():
            site, genre = cell.split("|")
            by_genre.setdefault(genre, []).append(s["mean"])
            by_site.setdefault(site, []).append(s["mean"])
        print("  by genre: %s" % ", ".join("%s %.1f" % (g, np.mean(v)) for g, v in sorted(by_genre.items())))
        print("  by site : %s" % ", ".join("%s %.1f" % (s_, np.mean(v)) for s_, v in sorted(by_site.items())))
        out[unit] = {c: {k: float(v) for k, v in s.items()} for c, s in per_cell.items()}

    print("\nMinimum-length rule: the extractor requires at least ten words.")
    short = {c: sum(1 for r in cells[c] if len(WORD.findall(r["query"])) < 10) for c in sorted(cells)}
    print("  queries under 10 words per cell: %s"
          % ", ".join("%s %d" % (c, n) for c, n in short.items()))

    dest = RESULTS / "query_length_table.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nWritten: %s" % dest)


if __name__ == "__main__":
    main()
