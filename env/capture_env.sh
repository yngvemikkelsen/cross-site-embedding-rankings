#!/usr/bin/env bash
# Run this ONCE on the analysis machine to record the exact environment.
# Commit the resulting env/environment.lock so the run is fully reproducible.
set -euo pipefail
OUT="env/environment.lock"
{
  echo "# Environment lock — generated $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "## Python"
  python --version 2>&1
  echo
  echo "## Platform"
  python - <<'PY'
import platform
print("system:", platform.system(), platform.release())
print("machine:", platform.machine())
print("processor:", platform.processor())
PY
  echo
  echo "## Pinned package versions (pip freeze, relevant subset)"
  pip freeze | grep -Ei '^(torch|transformers|sentence-transformers|einops|numpy|scipy|pandas|tokenizers|huggingface-hub)==' || true
  echo
  echo "## Full pip freeze"
  pip freeze
  echo
  echo "## GPU (if any)"
  (nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null) || echo "no GPU / nvidia-smi unavailable"
  echo
  echo "## Resolved HuggingFace model revisions"
  echo "# For each repo in manifests/models.csv, record the commit hash actually downloaded:"
  echo "#   python env/resolve_revisions.py   # writes manifests/models_resolved.csv"
} > "$OUT"
echo "wrote $OUT"
