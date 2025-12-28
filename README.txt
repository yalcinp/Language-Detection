 _          _ _
| |__   ___| | | ___
| '_ \ / _ \ | |/ _ \
| | | |  __/ | | (_) |
|_| |_|\___|_|_|\___/


A short technical report describing the implemented language identification method,
experimental setup, and results is included in this submission:

- Language_Identification_Report.pdf

The report summarizes:
- the character n-gram Naive Bayes approach,
- implementation details,
- evaluation results and comparisons with baselines,
- robustness and error analysis.


The complete source code is also included in this directory. It contains the full
implementation of the character n-gram Naive Bayes model, together with evaluation,
ablation, robustness, and interpretability scripts used to generate the reported
results.

[!] Note. The lid.176.bin model is required for the baseline experiments. It has been excluded from the submission to avoid memory-related issues. !! 