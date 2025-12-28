from __future__ import annotations
from pathlib import Path
from typing import List

import fasttext

from evaluate import evaluate

# Load model once
MODEL_PATH = Path("models/lid.176.bin")
model = fasttext.load_model(str(MODEL_PATH))

# Map fastText labels to our ISO-2 set
# fastText returns labels like "__label__en"
KEEP = {"en","de","fr","es","it","tr","nl","sv","pl","ru","ar","zh"}

def predict_fasttext(texts: List[str]) -> List[str]:
    preds = []
    for t in texts:
        labels, probs = model.predict(t, k=1)
        lab = labels[0].replace("__label__", "")
        # if model predicts outside our set, keep as-is (will count as error)
        preds.append(lab if lab in KEEP else lab)
    return preds

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

