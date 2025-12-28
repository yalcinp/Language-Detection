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

## Installation

It is recommended to use a virtual environment.

```bash
git clone https://github.com/yalcinp/Language-Detection.git
cd Language-Detection
pip install -r requirements.txt
```

---
## Reproducing All Experiments

All experiments and analyses can be reproduced with a single command: 
```bash
bash run.sh
```

## Implemented Analyses

Character n-gram Naive Bayes (1–5 grams)  
Baselines: langid.py and fastText  
Preprocessing ablation  
Whitespace ablation  
Short-text robustness evaluation  
Selective prediction via margin-based abstention  
Coverage–risk curves  
N-gram-level interpretability analysis

## FastText Model (`lid.176.bin`)

A pre-trained FastText language identification model is included as `lid.176.bin`. To use it:

```python
import fasttext

model = fasttext.load_model('lid.176.bin')
prediction = model.predict("Your text here")
print(prediction)
```