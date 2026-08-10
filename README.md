# Cross-Site Stability of Embedding Model Rankings for Known-Item Retrieval From Clinical Notes

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21434089.svg)](https://doi.org/10.5281/zenodo.21434089)

Analysis code and aggregate results for the two-corpus comparative evaluation study
(BIDMC/MIMIC-IV-Note vs UCSF/ER-Reason). This repository reproduces every table in the
manuscript from the source corpora.

**No clinical note text or patient identifiers are included.** The two corpora are available
only to credentialed users under their respective data use agreements (see below). This
repository contains the code, the exact model and environment specifications, the sampling
rules, and the aggregate (non-identifying) numerical results.

```bash
git clone https://github.com/yngvemikkelsen/cross-site-embedding-rankings.git
```

## What's here

```
src/                      analysis pipeline (run in order; see docs/RUN.md)
  two_site_v2.py            data layer: extract, dedup, patient IDs, query-excluded targets, matched N=1235
  two_site_v2_analyze.py    model layer: embed (13 models), chunk, score, cache RR vectors
  two_site_v2_stats.py      statistics: patient-clustered bootstrap CIs; variance decomposition (Tables 2-4)
  chunk_sensitivity2.py     robustness: chunk-count / order-statistic sensitivity (Table 5, part)
  anisotropy_check.py       robustness: embedding anisotropy diagnostic
  centering_check.py        robustness: mean-centering sensitivity (Table 5, part)
  er_audit.py               ER-Reason extraction audit
manifests/
  models.csv                exact HF repo, pooling, prefixes for all 13 models
  models_resolved.csv       (generated) exact commit hash per model — run env/resolve_revisions.py
  sampling_manifest.md      extraction, dedup, and matched-sampling rules + attrition table
results/
  aggregate_results.json    every number behind Tables 1-5 and the robustness analyses (no note text)
env/
  requirements.txt          library versions (replace PIN_VERSION placeholders before pinning; see notes)
  capture_env.sh            records python/platform/GPU/pip-freeze into environment.lock
  resolve_revisions.py      resolves exact HF commit hashes into the model manifest
docs/
  RUN.md                    step-by-step reproduction commands
```

## Data access (not redistributed here)

- **MIMIC-IV-Note v2.2** — PhysioNet, credentialed access + signed DUA.
  doi:10.13026/1n74-ne17
- **ER-Reason v1.0.0** — PhysioNet, credentialed access + signed DUA.
  doi:10.13026/55s7-3c27

Place the downloaded corpora where the scripts expect them (paths are set at the top of
`src/two_site_v2.py`) and follow `docs/RUN.md`.

## Reproducing the results

```bash
# 0. environment
python -m venv .venv && source .venv/bin/activate
pip install -r env/requirements.txt      # after pinning versions
bash env/capture_env.sh                   # writes env/environment.lock
python env/resolve_revisions.py           # writes manifests/models_resolved.csv

# 1-4. pipeline (see docs/RUN.md for full commands and expected outputs)
python src/two_site_v2.py                 # build matched analysis set
python src/two_site_v2_analyze.py --run --chunk   # embed + score (needs GPU for speed)
python src/two_site_v2_stats.py           # Tables 2-4
python src/chunk_sensitivity2.py          # chunk-count robustness (Table 5)
python src/anisotropy_check.py            # anisotropy diagnostic
python src/centering_check.py             # centering robustness (Table 5)
```

Every headline number in `results/aggregate_results.json` should reproduce from these steps.

## Determinism

- Fixed seed `42` throughout (sampling, bootstrap, sensitivity re-scoring).
- Bootstrap: 2000 patient-clustered resamples.
- Model revisions pinned via `manifests/models_resolved.csv` (commit hashes), so "BGE-base"
  etc. resolve to exact checkpoints rather than moving family pointers.

## Citation

If you use this code, please cite the manuscript (details on publication) and the two source
corpora (DOIs above). A `CITATION.cff` is included so GitHub's "Cite this repository" button
produces a ready-made reference. Archived release: doi:10.5281/zenodo.21434089 (https://doi.org/10.5281/zenodo.21434089).

Repository: https://github.com/yngvemikkelsen/cross-site-embedding-rankings

## License

Code: see `LICENSE` (MIT). The clinical corpora are governed by their own PhysioNet DUAs and
are **not** covered by this license.
