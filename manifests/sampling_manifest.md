# Sampling manifest

Documents the extraction, deduplication, and matched-sampling procedure that produces the
N=1235-per-cell analysis set. Contains **no clinical note text and no patient identifiers** —
only counts and the deterministic rules. The exact record IDs are regenerated from the source
corpora by `src/two_site_v2.py` under each corpus's data use agreement; they are not
redistributed here.

## Cells (2x2 design)

| Site  | Corpus                  | Genre     | Note type source                          |
|-------|-------------------------|-----------|-------------------------------------------|
| BIDMC | MIMIC-IV-Note v2.2      | discharge | `discharge.csv.gz`                        |
| BIDMC | MIMIC-IV-Note v2.2      | imaging   | `radiology.csv.gz`                        |
| UCSF  | ER-Reason v1.0.0        | discharge | discharge summaries in `er_reason.csv`    |
| UCSF  | ER-Reason v1.0.0        | imaging   | imaging/radiology reports in `er_reason.csv` |

## Pipeline (deterministic; produced by src/two_site_v2.py)

1. **Extract** notes of each genre from each corpus.
2. **Deduplicate** by a deterministic normalized-text key (exact equality after lowercasing,
   whitespace collapse, and digit removal). Duplicate rate observed: 10.8% of ER-Reason
   discharge notes, 12.2% of ER-Reason imaging notes (same historical note attached to
   multiple index encounters). *Action item for submission: state whether duplicate groups
   were manually inspected to confirm digit-removal did not collapse distinct templated notes.*
3. **Assign patient IDs** so bootstrap resampling can cluster by patient.
4. **Build queries** by extracting a query span from each note; **remove the query's own
   sentences from its target** before indexing (query-excluded targets) to prevent verbatim
   overlap. Matcher: coverage-window with a residual-leak guard.
5. **Match sample size** across all four cells to N=1235 (the size of the smallest cell after
   dedup and query exclusion), sampled with a fixed seed.

## Attrition (fill from your run logs)

| Stage                                  | BIDMC/disch | BIDMC/imag | UCSF/disch | UCSF/imag |
|----------------------------------------|-------------|------------|------------|-----------|
| Raw notes of genre                     | FILL        | FILL       | FILL       | FILL      |
| After dedup                            | FILL        | FILL       | FILL       | FILL      |
| After query-span extraction feasible   | FILL        | FILL       | FILL       | FILL      |
| Matched analysis set                   | 1235        | 1235       | 1235       | 1235      |

Run `python src/two_site_v2.py --report-attrition` to emit these counts from the source
corpora, then paste them here. (The counts are non-identifying and safe to commit.)

## Seeds

- Matched-sampling seed: `42` (see `SEED` in `src/two_site_v2.py`).
- Bootstrap seed and replicate count: 2000 patient-clustered resamples, seed `42`
  (`src/two_site_v2_stats.py`).
- Sensitivity re-scoring seed: `42` (`src/chunk_sensitivity2.py`, `src/centering_check.py`).
