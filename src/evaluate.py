from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, List, Dict, Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def load_jsonl(path: Path):
    texts, labels = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(obj["lang"])
    return texts, labels


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def evaluate(
    predictor: Callable[[List[str]], List[str]],
    test_path: Path,
    name: str,
    labels_order: List[str] | None = None,
    out_path: Path | None = None,
):
    texts, gold = load_jsonl(test_path)
    pred = predictor(texts)

    acc = accuracy_score(gold, pred)
    f1 = f1_score(gold, pred, average="macro")

    if labels_order is None:
        labels_order = sorted(list(set(gold)))

    cm = confusion_matrix(gold, pred, labels=labels_order)

    result = {
        "name": name,
        "n_test": len(gold),
        "labels": labels_order,
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }

    print(f"=== {name} ===")
    print(f"Test size : {len(gold)}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Macro-F1  : {f1:.4f}")

    if out_path is not None:
        save_json(out_path, result)
        print(f"Saved results to: {out_path}")

    return result

