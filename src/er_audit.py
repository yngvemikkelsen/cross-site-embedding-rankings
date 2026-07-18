"""ER-Reason structural audit — check the reviewer's claims before rebuilding anything.

The peer review flagged that the manuscript mischaracterises ER-Reason, and if true this
changes N, the era-confound claim, and the bootstrap unit BEFORE any rerun is worth doing.
Specifically, verify on disk:

  1. NOTE YEARS. The manuscript says UCSF notes are 2022-2024 with no MIMIC overlap. But
     ER-Reason selects a 2022-2024 ED *index encounter* and attaches HISTORICAL notes from
     prior encounters, with dates SHIFTED in de-identification. So Discharge_Summary_Year /
     Imaging_Year may predate 2022 and may overlap MIMIC's 2008-2019. Check the actual
     year columns in the analytical sample.

  2. DUPLICATION. The same historical note may appear in several index-encounter rows, and
     several rows may share a patient. Whole-corpus known-item retrieval with duplicates is
     broken: a duplicated target has >1 correct answer, and near-duplicates inflate every
     model's MRR. Count unique note keys, unique texts, patients, encounters.

  3. BOOTSTRAP UNIT. If one patient contributes multiple documents, per-query bootstrap
     overstates precision. Need patient-level clustering. Quantify docs-per-patient.

  4. The manuscript says ER-Reason "retains real calendar dates and times". The dataset
     doc describes DATE SHIFTING. Confirm dates are shifted, not real, and fix the wording.

No conclusions in the code. Read the numbers, then decide.

Usage:
    python er_audit.py
"""
from __future__ import annotations

import hashlib
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd

ER = Path(os.environ.get("ER_REASON_CSV",
          str(Path.home() / "physionet.org" / "files" / "er-reason" / "1.0.0" / "er_reason.csv")))
MIN_CHARS = 400


def norm_text(s: str) -> str:
    """Normalised key for near-duplicate detection: lowercase, collapse ws, drop digits/punct."""
    s = re.sub(r"\d", "", str(s).lower())
    s = re.sub(r"[^a-z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def audit_genre(df, text_col, year_col, note_key_col):
    print(f"\n{'='*88}\n{text_col}\n{'='*88}")
    sub = df[df[text_col].notna()].copy()
    sub = sub[sub[text_col].astype(str).str.len() >= MIN_CHARS]
    n = len(sub)
    print(f"  rows with note >= {MIN_CHARS} chars      : {n:,}")

    # --- 1. YEARS ---
    if year_col in df.columns:
        yr = pd.to_numeric(sub[year_col], errors="coerce").dropna()
        if len(yr):
            print(f"  {year_col}:")
            print(f"    range   : {int(yr.min())} - {int(yr.max())}")
            print(f"    median  : {int(yr.median())}")
            pre2020 = (yr < 2020).mean()
            overlap = ((yr >= 2008) & (yr <= 2019)).mean()
            print(f"    share < 2020            : {pre2020:.1%}")
            print(f"    share in MIMIC era 08-19: {overlap:.1%}  <-- if >0, 'no temporal overlap' is FALSE")
            print(f"    decade counts: ", dict(yr.astype(int).map(lambda y: f'{y//5*5}s').value_counts().sort_index()))
    else:
        print(f"  {year_col}: COLUMN ABSENT")

    # --- 2. DUPLICATION ---
    if note_key_col in df.columns:
        keys = sub[note_key_col].astype(str)
        print(f"  unique note keys        : {keys.nunique():,} of {n:,}  "
              f"({n - keys.nunique():,} duplicate-key rows)")
    texts = sub[text_col].astype(str)
    exact = texts.nunique()
    print(f"  unique exact texts      : {exact:,} of {n:,}  ({n - exact:,} exact-duplicate rows)")
    normed = texts.map(norm_text)
    nunq = normed.nunique()
    print(f"  unique normalised texts : {nunq:,} of {n:,}  ({n - nunq:,} near-duplicate rows)")
    if n - nunq > 0:
        print(f"    -> {(n-nunq)/n:.1%} of this cell is near-duplicate. Known-item retrieval")
        print(f"       with these present gives duplicated targets MULTIPLE correct answers.")

    # --- 3. PATIENT CLUSTERING ---
    if "patientdurablekey" in df.columns:
        pk = sub["patientdurablekey"].astype(str)
        print(f"  unique patients         : {pk.nunique():,} of {n:,} rows")
        dpp = pk.value_counts()
        print(f"    docs/patient: median {dpp.median():.0f}, max {dpp.max()}, "
              f"share of patients with >1 doc: {(dpp>1).mean():.1%}")
        if (dpp > 1).mean() > 0.05:
            print(f"    -> patient clustering is non-trivial; per-query bootstrap overstates")
            print(f"       precision. Use patient-clustered resampling.")

    return {"n": n, "unique_norm": nunq,
            "unique_patients": sub["patientdurablekey"].nunique() if "patientdurablekey" in df.columns else None}


def main():
    if not ER.exists():
        raise SystemExit(f"missing {ER}")
    df = pd.read_csv(ER, low_memory=False)
    print("=" * 88)
    print(f"ER-REASON AUDIT | {len(df):,} encounter rows")
    print("=" * 88)

    # year columns present?
    yearcols = [c for c in df.columns if "year" in c.lower()]
    print(f"  year columns present: {yearcols}")

    # do the two genres we use in the study
    res = {}
    res["discharge"] = audit_genre(df, "Discharge_Summary_Text", "Discharge_Summary_Year",
                                   "Discharge_Summary_Note_Key")
    res["imaging"] = audit_genre(df, "Imaging_Text", "Imaging_Year", "Imaging_Key")

    # cross-genre patient overlap and encounter structure
    print(f"\n{'='*88}\nENCOUNTER / PATIENT STRUCTURE\n{'='*88}")
    print(f"  encounters (rows)       : {len(df):,}")
    if "patientdurablekey" in df.columns:
        print(f"  unique patients         : {df['patientdurablekey'].nunique():,}")
        epp = df["patientdurablekey"].value_counts()
        print(f"  encounters/patient      : median {epp.median():.0f}, max {epp.max()}, "
              f">1: {(epp>1).mean():.1%}")
        print(f"    -> if a patient has multiple index encounters, the SAME historical note")
        print(f"       can enter the corpus multiple times. That is the duplication above.")

    print(f"\n{'='*88}\nWHAT THIS MEANS FOR THE RERUN\n{'='*88}")
    for g in ["discharge", "imaging"]:
        r = res[g]
        print(f"  {g}: {r['n']:,} raw -> {r['unique_norm']:,} after near-dup removal"
              f"  ({r['n']-r['unique_norm']:,} dropped)")
    smallest = min(res[g]["unique_norm"] for g in res)
    print(f"\n  New matched N per cell after ER dedup <= {smallest:,} (was 1276).")
    print(f"  MIMIC cells must be resampled to this N too.")
    print(f"  Bootstrap unit: PATIENT, not query, wherever docs/patient > 1.")
    print(f"  Fix the era-confound wording per the actual year distributions above.")


if __name__ == "__main__":
    main()
