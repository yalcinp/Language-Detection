#!/usr/bin/env bash
set -euo pipefail

# Run from repo root (where src/ exists)
if [[ ! -d "src" ]]; then
  echo "ERROR: src/ directory not found."
  echo "Please run this script from the repository root."
  exit 1
fi

# Prefer python3 if available
PYTHON_BIN="python"
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

# Basic sanity checks
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python interpreter not found (python/python3)."
  exit 1
fi

mkdir -p results

echo "Python version:"
"$PYTHON_BIN" --version
echo

echo "=== [1/5] Mini data summary ==="
"$PYTHON_BIN" src/data_summary.py

echo "=== [2/5] Main evaluations & ablations ==="
"$PYTHON_BIN" src/run_all_ablations.py --all

echo "=== [3/5] Short-text robustness ==="
"$PYTHON_BIN" src/eval_short_text_robustness_all.py

echo "=== [4/5] Coverage–risk analysis ==="
"$PYTHON_BIN" src/eval_coverage_risk_all.py

echo "=== [5/5] NB interpretability example ==="
"$PYTHON_BIN" src/inspect_nb_ngrams.py --topk 5

echo
echo "=== DONE ==="
echo "All results written to: results/"
