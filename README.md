# Cross-Site Stability of Embedding Model Rankings for Known-Item Retrieval From Clinical Notes

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21434089.svg)](https://doi.org/10.5281/zenodo.21434089)

Analysis code and aggregate results for the two-corpus comparative evaluation study
(BIDMC/MIMIC-IV-Note vs UCSF/ER-Reason). This repository reproduces every table and figure in
the manuscript from the source corpora.

**No clinical note text or patient identifiers are included.** The two corpora are available
only to credentialed users under their respective data use agreements (see below). This
repository contains the code, the exact model and environment specifications, the sampling
rules, and the aggregate (non-identifying) numerical results.

```bash
git clone https://github.com/yngvemikkelsen/cross-site-embedding-rankings.git
```

## Version 2.0.0 — peer-review revision

This release accompanies the revised manuscript (JMIR AI #109305) and supersedes v1.1.2.

**Corrected defect.** In v1.1.2, `_encode_chunks` in `src/two_site_v2_analyze.py` concatenated
a model's document prefix to the full document before token windowing, so only the first
window of a multi-window document carried the prefix. Two of the thirteen models have a
non-empty document prefix (E5-base, `"passage: "`; Nomic-embed, `"search_document: "`); the
other eleven were unaffected. The prefix is now tokenized separately and prepended to every
window, with the window budget reduced by the prefix length so no encoded unit exceeds the
512-token limit. All analyses depending on document embeddings were re-run. Query encoding
was never affected: queries are encoded as a single window and always carried their prefix.

Effect on the reported results: cross-site Kendall tau for imaging fell from 0.714 to 0.643
(contrastive panel) and from 0.846 to 0.821 (full panel); discharge tau was unchanged at
0.857/0.923; selection regret fell from 0.041 to 0.033 (UCSF→BIDMC discharge); the
contrastive-panel decomposition moved from model 0.329 to 0.311, genre 0.535 to 0.563, and
model×site 0.012 to 0.008. The substantive conclusions are unchanged.

**Other corrections in this release.**

- `src/two_site_v2_stats.py` previously computed its point estimates from a hardcoded table of
  MRR values, so it reported stored numbers rather than the current embeddings. It now
  recomputes point estimates from the cached reciprocal-rank vectors whenever they are
  present, and states in its output which source it used. The stored table remains only as a
  fallback when the cache is absent.
- `src/two_site.py`, `src/two_site_v2_hits.py` and `src/gen_fig2.py` were referenced by the
  pipeline or by the manuscript but were missing from v1.1.2. `src/two_site_v2.py` imports
  `two_site.py` by path, so the previous release could not rebuild the analysis set.
- `results/aggregate_results.json` has been regenerated from the corrected run.
- `docs/RUN.md` documented a `RESULTS_DIR` value that was not the one used; the variable is
  now documented as required rather than defaulted.
- `manifests/sampling_manifest.md` now reports the exhaustive duplicate-merge audit rather
  than asserting that observed duplicate rates are benign.
- Five scripts added under `src/revision/` generate the analyses requested during peer review.

## What's here

```
src/                          analysis pipeline (run in order; see docs/RUN.md)
  two_site.py                   query extraction primitives: scrub, reflow, sentences,
                                build_df, narrative_query (imported by two_site_v2.py)
  two_site_v2.py                data layer: extract, dedup, patient IDs, query-excluded
                                targets, matched N=1235
  two_site_v2_analyze.py        model layer: embed (13 models), chunk, score, cache RR
                                vectors  [corrected in v2.0.0]
  two_site_v2_stats.py          statistics: patient-clustered bootstrap CIs; variance
                                decomposition (Tables 3-5)  [corrected in v2.0.0]
  two_site_v2_hits.py           secondary metrics: Hit@1 and Hit@10 (Table 2, Appendix 2)
  chunk_sensitivity2.py         robustness: chunk-count / order-statistic sensitivity
                                (Table 6, part)
  anisotropy_check.py           robustness: embedding anisotropy diagnostic
  centering_check.py            robustness: mean-centering sensitivity (Table 6, part)
  er_audit.py                   ER-Reason extraction audit
  gen_fig2.py                   Figure 2 (cross-site transfer and variance decomposition)
src/revision/                 analyses added in response to peer review
  propagation_pack.py           recomputes Tables 1-5, the paired eta-squared contrasts and
                                all bootstrap intervals from the cached RR vectors
  single_chunk_multiseed.py     single-random-chunk scoring across 20 seeds (comment 12)
  dup_classify.py               exhaustive classification of every duplicate-merge group
                                (comment 5)
  audit_extraction.py           per-criterion extraction attrition and sentence-level filter
                                counts (comments 9 and 15)
  query_length_table.py         query length distributions by cell (comment 19, Appendix 3)
manifests/
  models.csv                    exact HF repo, pooling, prefixes for all 13 models
  models_resolved.csv           (generated) exact commit hash per model — run
                                env/resolve_revisions.py
  sampling_manifest.md          extraction, dedup, and matched-sampling rules, attrition
                                table, and duplicate-merge audit
results/
  aggregate_results.json        every number behind Tables 1-6 and the robustness analyses
                                (no note text)  [regenerated in v2.0.0]
env/
  requirements.txt              exact pinned analysis-library versions
  capture_env.sh                records python/platform/GPU/pip-freeze into environment.lock
  resolve_revisions.py          resolves exact HF commit hashes into the model manifest
docs/
  RUN.md                        step-by-step reproduction commands
CITATION.cff                  machine-readable citation metadata
LICENSE                       MIT (code only)
README.md                     this file
```

## Data access (not redistributed here)

- **MIMIC-IV-Note v2.2** — PhysioNet, credentialed access + signed DUA.
  doi:10.13026/1n74-ne17
- **ER-Reason v1.0.0** — PhysioNet, credentialed access + signed DUA.
  doi:10.13026/55s7-3c27

Place the downloaded corpora where the scripts expect them and follow `docs/RUN.md`. Corpus
locations and the results directory are set by environment variables or command-line flags;
see the top of `src/two_site_v2.py`.

## Reproducing the results

`RESULTS_DIR` must be set: it is where the analysis set, embedding caches, reciprocal-rank
vectors and output tables are written and read. All scripts honour it.

```bash
# 0. environment
python -m venv .venv && source .venv/bin/activate
pip install -r env/requirements.txt
bash env/capture_env.sh                   # writes env/environment.lock
python env/resolve_revisions.py           # writes manifests/models_resolved.csv

export RESULTS_DIR=/path/to/results

# 1-4. pipeline (see docs/RUN.md for full commands and expected outputs)
python src/two_site_v2.py --build                 # build matched analysis set
python src/two_site_v2_analyze.py --run --chunk   # embed + score (GPU/MPS recommended)
python src/two_site_v2_stats.py                   # Tables 3-5
python src/two_site_v2_hits.py                    # Table 2, Appendix 2
python src/chunk_sensitivity2.py                  # chunk-count robustness (Table 6)
python src/anisotropy_check.py                    # anisotropy diagnostic
python src/centering_check.py                     # centering robustness (Table 6)
python src/gen_fig2.py -o figure2.png             # Figure 2

# 5. analyses added during peer review
python src/revision/propagation_pack.py           # all corrected table values, one pass
python src/revision/single_chunk_multiseed.py     # 20-seed single-chunk sensitivity
python src/revision/dup_classify.py               # duplicate-merge classification
python src/revision/audit_extraction.py           # extraction attrition by criterion
python src/revision/query_length_table.py         # query length distributions
```

Every headline number in `results/aggregate_results.json` should reproduce from these steps.
`src/revision/propagation_pack.py` regenerates Tables 1-5 and both paired eta-squared
contrasts in a single run and is the quickest way to verify the reported values.

## Determinism

- Fixed seed `42` throughout (sampling, bootstrap, sensitivity re-scoring); seed `17` for the
  scripts that state it.
- Bootstrap: 2000 patient-clustered resamples, drawn once per site and applied jointly to that
  site's two genre cells.
- The single-random-chunk sensitivity is reported as the mean and standard deviation across 20
  independent seeds rather than a single realization.
- Model revisions pinned via `manifests/models_resolved.csv` (commit hashes), so "BGE-base"
  etc. resolve to exact checkpoints rather than moving family pointers.

## Citation

If you use this code, please cite the manuscript (details on publication) and the two source
corpora (DOIs above). A `CITATION.cff` is included so GitHub's "Cite this repository" button
produces a ready-made reference. Archived release: doi:10.5281/zenodo.21434089
(https://doi.org/10.5281/zenodo.21434089).

Repository: https://github.com/yngvemikkelsen/cross-site-embedding-rankings

## License

Code: see `LICENSE` (MIT). The clinical corpora are governed by their own PhysioNet DUAs and
are **not** covered by this license.
