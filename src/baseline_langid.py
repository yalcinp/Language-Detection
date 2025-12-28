from __future__ import annotations
from pathlib import Path
from typing import List

from py3langid import langid

from evaluate import evaluate

# Make langid deterministic / consistent
langid.set_languages(["en", "de", "fr", "es", "it", "tr", "nl", "sv", "pl", "ru", "ar", "zh"])

def predict_langid(texts: List[str]) -> List[str]:
    preds = []
    for t in texts:
        lang, _ = langid.classify(t)
        preds.append(lang)
    return preds

if __name__ == "__main__":
    test_path = Path("data/test.jsonl")
    labels = ["en", "de", "fr", "es", "it", "tr", "nl", "sv", "pl", "ru", "ar", "zh"]
    evaluate(
        predictor=predict_langid,
        test_path=test_path,
        name="langid.py baseline",
        labels_order=labels,
        out_path=Path("results/langid_baseline.json"),
    )



