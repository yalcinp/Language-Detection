from __future__ import annotations
from pathlib import Path
from typing import List
import langid
from evaluate import evaluate


LABELS = ["en", "de", "fr", "es", "it", "tr", "nl", "sv", "pl", "ru", "ar", "zh"]
# Restrict the model to our specific language set 
langid.set_languages(LABELS)

def predict_langid(texts: List[str]) -> List[str]:
    """py3langid language identification."""
    preds = []
    for t in texts:
        lang, _ = langid.classify(t)
        preds.append(lang)
    return preds

# Evaluate on test split
if __name__ == "__main__":
    test_path = Path("data/test.jsonl")
    
    evaluate(
        predictor=predict_langid,
        test_path=test_path,
        name="langid.py baseline",
        labels_order=LABELS,
        out_path=Path("results/langid_baseline.json"),
    )



