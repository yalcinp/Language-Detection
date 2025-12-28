# Language Detection with Character n-gram Naive Bayes

This repository contains a fully transparent and reproducible implementation of a
character n-gram Naive Bayes language identification system, together with
systematic evaluation, ablation, robustness, abstention, and interpretability analyses.

The focus is not only on accuracy, but on understanding robustness and uncertainty
under controlled experimental settings.

---

## Supported Languages

Experiments are conducted on a fixed set of 12 languages:

en, de, fr, es, it, tr, nl, sv, pl, ru, ar, zh

All data splits are balanced across languages.

---

## Repository Structure

.
├── src/
│ ├── own_char_ngram_nb.py
│ ├── prepare_data.py
│ ├── data_summary.py
│ ├── evaluate.py
│ ├── run_all_ablations.py
│ ├── eval_preproc_ablation.py
│ ├── eval_space_ablation.py
│ ├── eval_short_text_robustness_all.py
│ ├── eval_coverage_risk_all.py
│ ├── eval_abstain_nb.py
│ ├── inspect_nb_ngrams.py
│ ├── baseline_langid.py
│ └── baseline_fasttext.py
│
├── results/
├── run.sh
├── requirements.txt
└── README.md



---

## Installation

It is recommended to use a virtual environment.

```bash
git clone https://github.com/yalcinp/Language-Detection.git
cd Language-Detection
pip install -r requirements.txt

## Implemented analyses:

Character n-gram Naive Bayes with 1–5 grams
Baselines: langid.py and fastText
Preprocessing ablation
Whitespace ablation
Short-text robustness evaluation
Selective prediction via margin-based abstention
Coverage–risk curves
N-gram-level interpretability analysis
