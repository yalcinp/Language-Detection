from __future__ import annotations
from pathlib import Path
from typing import List

import fasttext
from evaluate import evaluate

# Load pre-trained fastText model
MODEL_PATH = Path("models/lid.176.bin")
model = fasttext.load_model(str(MODEL_PATH))

LABELS = {"en", "de", "fr", "es", "it", "tr", "nl", "sv", "pl", "ru", "ar", "zh"}

def predict_fasttext(texts: List[str]) -> List[str]:
    """fastText LID predictions for a batch of texts."""
    preds = []
    for t in texts:
        labels, probs = model.predict(t, k=1)
        clean_label = labels[0].replace("__label__", "") # strip "__label__" prefix
        # Keep out-of-set labels unchanged
        preds.append(clean_label if clean_label in LABELS else clean_label)
    return preds

# Evaluate on test split
if __name__ == "__main__":
    test_path = Path("data/test.jsonl")
    labels = ["en","de","fr","es","it","tr","nl","sv","pl","ru","ar","zh"]
    evaluate(
        predictor=predict_fasttext,
        test_path=test_path,
        name="fastText lid.176 baseline",
        labels_order=labels,
        out_path=Path("results/fasttext_baseline.json"),
    )

