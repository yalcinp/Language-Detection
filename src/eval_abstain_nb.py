from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from own_char_ngram_nb import CharNgramNB, load_train

# Experiment settings
N_MIN, N_MAX = 1, 5
ALPHA = 0.1
PERCENTILES = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]

def normalize(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return " ".join(text.lower().split())

def shortk(t: str, k: int) -> str:
    return normalize(t)[:k]

def ngrams_count(s: str, n_min: int, n_max: int) -> int:
    L = len(s)
    total = 0
    for n in range(n_min, n_max + 1):
        total += max(0, L - n + 1)
    return max(1, total)

def main() -> None:
    train_path = Path("data/train.jsonl")
    test_path = Path("data/test.jsonl")

    # Train model
    train_data = load_train(train_path)
    model = CharNgramNB(n_min=N_MIN, n_max=N_MAX, alpha=ALPHA)
    model.fit(train_data)

    # Load test split
    test_rows = []
    with test_path.open("r", encoding="utf-8") as f:
        for line in f:
            test_rows.append(json.loads(line))

    texts_full = [normalize(r["text"]) for r in test_rows]
    gold = [r["lang"] for r in test_rows]

    settings = [
        ("full", texts_full),
        ("short50", [shortk(t, 50) for t in texts_full]),
        ("short20", [shortk(t, 20) for t in texts_full]),
    ]

    for tag, texts in settings:
        out_path = Path(f"results/abstain_nb_{tag}.json")
        preds, margins = model.predict_with_scores(texts)

        # Normalize margins by n-gram count
        m = np.array(margins, dtype=float)
        ng_counts = np.array([ngrams_count(t, N_MIN, N_MAX) for t in texts], dtype=float)
        m_norm = m / ng_counts

        # Percentile sweep over tau
        taus = np.percentile(m_norm, PERCENTILES).tolist()

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
                    "name": f"NB ({tag})",
                    "setting": tag,
                    "confidence": f"margin(top1-top2), ngram-normalized (# {N_MIN}-{N_MAX} char-ngrams)",
                    "percentiles": PERCENTILES,
                    "curve": curve,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print("Saved:", out_path)

if __name__ == "__main__":
    main()
