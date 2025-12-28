#!/usr/bin/env bash
set -e

echo "=== [1/5] Mini data summary ==="
python src/data_summary.py

echo "=== [2/5] Main evaluations & ablations ==="
python src/run_all_ablations.py --all

echo "=== [3/5] Short-text robustness ==="
python src/eval_short_text_robustness_all.py

echo "=== [4/5] Coverage–risk analysis ==="
python src/eval_coverage_risk_all.py

echo "=== [5/5] NB interpretability example ==="
python src/inspect_nb_ngrams.py --topk 5

echo "=== DONE: all results written to results/ ==="

