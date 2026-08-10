# Reproduction guide

Every command assumes you are in the repository root with the virtual environment active and
the two corpora downloaded to their credentialed locations.

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate
# install the exact analysis environment
pip install -r env/requirements.txt
bash env/capture_env.sh            # -> env/environment.lock
python env/resolve_revisions.py    # -> manifests/models_resolved.csv (exact commit hashes)
```

## 1. Corpus paths

Point the scripts at your credentialed downloads via environment variables (defaults shown):

```bash
export ER_REASON_CSV="$HOME/physionet.org/files/er-reason/1.0.0/er_reason.csv"
export RESULTS_DIR="$HOME/paper19_results"          # where caches + outputs are written
# MIMIC-IV-Note note dir defaults to $HOME/physionet.org/files/mimic-iv-note/2.2/note
# or pass --mimic-note /path/to/note to src/two_site_v2.py
```

Corpora needed:
- MIMIC-IV-Note v2.2 (`discharge.csv.gz`, `radiology.csv.gz`)
- ER-Reason v1.0.0 (`er_reason.csv`)

No absolute user paths are hardcoded; everything resolves from these variables.

## 2. Build the matched analysis set

```bash
python src/two_site_v2.py
# Produces the deduplicated, query-excluded, patient-tagged, matched (N=1235/cell) cell file.
# Emits the attrition counts reported in manifests/sampling_manifest.md.
```

## 3. Embed and score (13 models, chunked)

```bash
python src/two_site_v2_analyze.py --run --chunk
# Embeds all 13 models with their manifest-specified pooling/prefixes, chunks long docs
# (510-token window, 384 stride, 126 overlap), scores by max-over-chunk cosine, caches
# reciprocal-rank vectors. GPU strongly recommended.
# Quick subset check: python src/two_site_v2_analyze.py --run --chunk --models bge gte e5 nomic
```

## 4. Primary statistics (Tables 3-5)

```bash
python src/two_site_v2_stats.py
# Cross-site Kendall tau (Table 3), selection regret (Table 4), variance decomposition
# full-panel + contrastive with patient-clustered bootstrap CIs (Table 5).
```

## 5. Robustness (Table 6 + diagnostics)

```bash
python src/chunk_sensitivity2.py   # chunk-count / order-statistic sensitivity
python src/anisotropy_check.py     # off-diagonal cosine per model; null floor vs slope
python src/centering_check.py      # mean-centering sensitivity (genre survives; tau stays positive)
```

## Expected numbers

Compare against `results/aggregate_results.json`. Headline checks:
- Discharge contrastive cross-site tau ~0.86; imaging ~0.71.
- model x site eta-squared 0.010-0.028 across all scoring choices.
- Site main effect: ~0.005 best-chunk, ~0.160 mean-pooled (scoring-dependent).
- Genre survives mean-centering (~0.36) and the chunk-count null slope is small (0.01-0.08).

## Notes

- All seeds are `42`; bootstrap is 2000 patient-clustered resamples.
- The `er_audit.py` script is a standalone check on ER-Reason extraction and is not required
  for the main pipeline.
