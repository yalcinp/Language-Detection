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
# -------------------------
# 1) Data sanity check
# -------------------------
echo "=== [1/9] Dataset summary ==="
$PYTHON_BIN src/data_summary.py
echo

# -------------------------
# 2) Main NB evaluation (golden reference)
# -------------------------
echo "=== [2/9] Main NB evaluation ==="
$PYTHON_BIN src/evaluate.py
echo

# -------------------------
# 3) Baselines
# -------------------------
echo "=== [3/9] Baseline: langid.py ==="
$PYTHON_BIN src/baseline_langid.py
echo

echo "=== [4/9] Baseline: fastText ==="
$PYTHON_BIN src/baseline_fasttext.py
echo

# -------------------------
# 4) Ablations (single entrypoint)
# -------------------------
echo "=== [5/9] All ablations ==="
$PYTHON_BIN src/run_all_ablations.py --all
echo

# -------------------------
# 5) Short-text robustness
# -------------------------
echo "=== [6/9] Short-text robustness ==="
$PYTHON_BIN src/eval_short_text_robustness_all.py
echo

# -------------------------
# 6) Selective prediction & coverage–risk
# -------------------------
echo "=== [7/9] Abstention and coverage–risk analysis ==="
$PYTHON_BIN src/eval_abstain_nb.py
$PYTHON_BIN src/eval_coverage_risk_all.py
$PYTHON_BIN src/plot_abstain_pdf.py
echo

# -------------------------
# 7) Interpretability
# -------------------------
echo "=== [8/9] NB interpretability example ==="
$PYTHON_BIN src/inspect_nb_ngrams.py --topk 5
echo

echo "=== DONE ==="
echo "All results written to: results/"
echo "Tip: run './run.sh --prepare' to regenerate the dataset."

echo
echo "=== DONE ==="
echo "All results written to: results/"