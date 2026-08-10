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
   multiple index encounters). The normalized-text key removes digits, so two templated notes that differ only in
   numerical content would be collapsed; this is a deliberately conservative duplicate-removal
   choice (it can only remove candidate targets, never add spurious matches), and the observed
   duplicate rates above are consistent with the same historical note being attached to a
   patient's multiple index encounters rather than with over-collapsing distinct notes.
3. **Assign patient IDs** so bootstrap resampling can cluster by patient.
4. **Build queries** by extracting a query span from each note; **remove the query's own
   sentences from its target** before indexing (query-excluded targets) to prevent verbatim
   overlap. Matcher: coverage-window with a residual-leak guard.
5. **Match sample size** across all four cells to N=1235 (the size of the smallest cell after
   dedup and query exclusion), sampled with a fixed seed.

## Attrition (from the build log; verified)

| Stage                                     | BIDMC/disch | BIDMC/imag | UCSF/disch | UCSF/imag |
|-------------------------------------------|-------------|------------|------------|-----------|
| Raw notes of genre                        | 331,792     | 2,036,503  | 3,872      | 2,526     |
| After dedup                               | 331,790     | 2,020,745  | 3,453      | 2,219     |
| Equalized pool (min dedup cell = 2,219)   | 2,219       | 2,219      | 2,219      | 2,219     |
| Usable after query exclusion              | 2,125       | 1,945      | 2,125      | 1,235     |
| Matched analysis set                      | 1,235       | 1,235      | 1,235      | 1,235     |

After deduplication, each cell is sampled without replacement to a common pool of 2,219
documents (the smallest deduplicated cell, UCSF/imaging) using fixed seed 42, so query
extraction operates on an equal-sized pool per cell. A query is then extracted from each
document and its sentences removed from the target; documents with fewer than 100 residual
characters, or in which the normalized query text still appears in the target, are excluded,
leaving the usable counts above. The four usable cells are matched to the smallest usable cell
(UCSF/imaging, 1,235) with seed 42, giving N=1,235 per cell (4,940 total). These counts are
emitted by `python src/two_site_v2.py --build` (which prints the dedup, pool, usable, and
matched lines) and are non-identifying, hence safe to commit.

## Seeds

- Matched-sampling seed: `42` (see `SEED` in `src/two_site_v2.py`).
- Bootstrap seed and replicate count: 2000 patient-clustered resamples, seed `42`
  (`src/two_site_v2_stats.py`).
- Sensitivity re-scoring seed: `42` (`src/chunk_sensitivity2.py`, `src/centering_check.py`).
