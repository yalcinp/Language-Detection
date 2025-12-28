from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from own_char_ngram_nb import CharNgramNB, load_train


def main() -> None:
    train_path = Path("data/train.jsonl")
    test_path = Path("data/test.jsonl")
    out_path = Path("results/abstain_nb_curve.json")

    # Train
    train_data = load_train(train_path)
    model = CharNgramNB(n_min=1, n_max=5, alpha=0.1)
    model.fit(train_data)

    # Test (manual JSONL read, no imports)
    test_rows = []
    with test_path.open("r", encoding="utf-8") as f:
        for line in f:
            test_rows.append(json.loads(line))

    texts = [r["text"] for r in test_rows]
    gold = [r["lang"] for r in test_rows]

    preds, margins = model.predict_with_scores(texts)

    # length-normalize margin (char length)
    m = np.array(margins, dtype=float)
    def ngrams_count(s: str, n_min: int, n_max: int) -> int:
        L = len(s)
        total = 0
        for n in range(n_min, n_max + 1):
            total += max(0, L - n + 1)
        return max(1, total)

    ng_counts = np.array([ngrams_count(t, 1, 5) for t in texts], dtype=float)
    m_norm = m / ng_counts


    # percentile-based tau sweep (on normalized margins)
    taus = np.percentile(m_norm, [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]).tolist()


    curve = []
    N = len(gold)

    for tau in taus:
        kept = [i for i, mi in enumerate(m_norm) if mi >= tau]
        n_kept = len(kept)
        coverage = n_kept / N

        if n_kept == 0:
            acc = None
        else:
            correct = sum(1 for i in kept if preds[i] == gold[i])
            acc = correct / n_kept

        curve.append(
            {
                "tau": float(tau),
                "coverage": float(coverage),
                "accuracy": (None if acc is None else float(acc)),
                "n_kept": int(n_kept),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "name": "Char n-gram NB selective prediction (margin = top1 - top2 log-posterior)",
                "curve": curve,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Saved:", out_path)
    print("tau\tcoverage\taccuracy\tn_kept")
    for r in curve:
        acc_str = "None" if r["accuracy"] is None else f"{r['accuracy']:.4f}"
        print(f"{r['tau']:.4f}\t{r['coverage']:.3f}\t\t{acc_str}\t{r['n_kept']}")


if __name__ == "__main__":
    main()

