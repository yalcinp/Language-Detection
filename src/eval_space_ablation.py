from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple, Dict
from own_char_ngram_nb import CharNgramNB, load_train
from evaluate import evaluate

LABELS = ["en","de","fr","es","it","tr","nl","sv","pl","ru","ar","zh"]
LS = [None, 20, 30]  # None = full


def truncate(s: str, L: int | None) -> str:
    return s if L is None else s[:L]


if __name__ == "__main__":
    train_path = Path("data/train.jsonl")
    test_path = Path("data/test.jsonl")
    out_dir = Path("results/space_short_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_train(train_path)

    # Initialize two models with the same config
    m_keep = CharNgramNB(n_min=1, n_max=5, alpha=0.1)
    m_drop = CharNgramNB(n_min=1, n_max=5, alpha=0.1)

    # Train keep-spaces model (on normalized text)
    train_keep = [(m_keep.normalize(t), y) for (t, y) in train_data]
    m_keep.fit(train_keep)

    # Train drop-spaces model (normalized and spaces removed)
    train_drop = [(m_drop.normalize(t), y) for (t, y) in train_data]
    m_drop.fit(train_drop)

    rows: List[Dict] = []

    for L in LS:
        tag = "full" if L is None else f"L{L}"

        def predict_keep(texts, _L=L):
            return [m_keep.predict_one(truncate(m_keep.normalize(t), _L)) for t in texts]

        def predict_drop(texts, _L=L):
            return [m_drop.predict_one(truncate(m_drop.normalize(t), _L)) for t in texts]

        res_keep = evaluate(
            predictor=predict_keep,
            test_path=test_path,
            name=f"NB keep spaces | {tag}",
            labels_order=LABELS,
            out_path=out_dir / f"nb_keep_{tag}.json",
        )

        res_drop = evaluate(
            predictor=predict_drop,
            test_path=test_path,
            name=f"NB drop spaces | {tag}",
            labels_order=LABELS,
            out_path=out_dir / f"nb_drop_{tag}.json",
        )

        rows.append(
            {
                "setting": tag,
                "L": L,
                "keep_acc": res_keep["accuracy"],
                "keep_f1": res_keep["macro_f1"],
                "drop_acc": res_drop["accuracy"],
                "drop_f1": res_drop["macro_f1"],
                "dacc_drop_minus_keep": res_drop["accuracy"] - res_keep["accuracy"],
                "df1_drop_minus_keep": res_drop["macro_f1"] - res_keep["macro_f1"],
            }
        )

    summary_path = out_dir / "summary_space_short_ablation.json"
    summary_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print("Saved:", summary_path)

    print("\nWhitespace ablation (NB): keep vs drop spaces, with optional truncation")
    print("setting\tkeep_acc\tkeep_f1\tdrop_acc\tdrop_f1\tdAcc\t\tdF1")
    for r in rows:
        print(
            f"{r['setting']}\t"
            f"{r['keep_acc']:.4f}\t{r['keep_f1']:.4f}\t"
            f"{r['drop_acc']:.4f}\t{r['drop_f1']:.4f}\t"
            f"{r['dacc_drop_minus_keep']:+.4f}\t{r['df1_drop_minus_keep']:+.4f}"
        )
