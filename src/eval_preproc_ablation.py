from __future__ import annotations
import json
import unicodedata
import numpy as np
import matplotlib.pyplot as plt
import fasttext
from pathlib import Path
from typing import Dict, List, Tuple
from py3langid import langid
from evaluate import evaluate
from own_char_ngram_nb import CharNgramNB, load_train


LABELS = ["en", "de", "fr", "es", "it", "tr", "nl", "sv", "pl", "ru", "ar", "zh"]


def drop_punct(s: str) -> str:
    """Strip Unicode punctuation."""
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))


def preprocess(text: str, *, lower: bool, nfkc: bool, drop_punctuation: bool) -> str:
    """Configurable pipeline for preprocessing ablation."""
    t = text
    if nfkc:
        t = unicodedata.normalize("NFKC", t)
    if lower:
        t = t.lower()
    if drop_punctuation:
        t = drop_punct(t)

    # Standardize whitespace
    t = " ".join(t.split())
    return t

def cfg_name(lower: bool, nfkc: bool, drop_punctuation: bool) -> str:
    return f"lower={int(lower)} nfkc={int(nfkc)} punctdrop={int(drop_punctuation)}"


def short_cfg_label(cfg: str) -> str:
    """Map config string to short label."""
    parts = cfg.split()
    l = parts[0].split("=")[1]
    n = parts[1].split("=")[1]
    p = parts[2].split("=")[1]
    return f"l{l} n{n} p{p}"


def read_metrics(path: Path) -> Tuple[float, float]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return float(obj["accuracy"]), float(obj["macro_f1"])


def main() -> None:
    train_path = Path("data/train.jsonl")
    test_path = Path("data/test.jsonl")
    ft_model_path = Path("models/lid.176.bin")

    if not train_path.exists():
        raise FileNotFoundError(f"Missing: {train_path.resolve()} (run prepare_data.py first)")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing: {test_path.resolve()} (run prepare_data.py first)")
    if not ft_model_path.exists():
        raise FileNotFoundError(f"Missing fastText model: {ft_model_path.resolve()}")

    # fastText + langid init once
    ft = fasttext.load_model(str(ft_model_path))
    langid.set_languages(LABELS)

    # Load train 
    train_raw = load_train(train_path)  

    out_dir = Path("results/preproc_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []

    # 8 configs in deterministic order
    configs = [(lower, nfkc, punctdrop) for lower in (False, True) for nfkc in (False, True) for punctdrop in (False, True)]

    for lower, nfkc, punctdrop in configs:
        tag = cfg_name(lower, nfkc, punctdrop)

        # NB: retrain with this preprocessing 
        train_proc = [(preprocess(t, lower=lower, nfkc=nfkc, drop_punctuation=punctdrop), y) for (t, y) in train_raw]
        nb = CharNgramNB(n_min=1, n_max=5, alpha=0.1)
        nb.fit(train_proc)

        nb_out = out_dir / f"nb_{tag}.json"
        evaluate(
            predictor=lambda xs, _nb=nb: [
                _nb.predict_one(preprocess(x, lower=lower, nfkc=nfkc, drop_punctuation=punctdrop)) for x in xs
            ],
            test_path=test_path,
            name=f"NB | {tag}",
            labels_order=LABELS,
            out_path=nb_out,
        )
        nb_acc, nb_f1 = read_metrics(nb_out)

        # langid 
        li_out = out_dir / f"langid_{tag}.json"
        evaluate(
            predictor=lambda xs: [
                langid.classify(preprocess(x, lower=lower, nfkc=nfkc, drop_punctuation=punctdrop))[0] for x in xs
            ],
            test_path=test_path,
            name=f"langid | {tag}",
            labels_order=LABELS,
            out_path=li_out,
        )
        li_acc, li_f1 = read_metrics(li_out)

        # fastText 
        ft_out = out_dir / f"fasttext_{tag}.json"
        evaluate(
            predictor=lambda xs: [
                ft.predict(preprocess(x, lower=lower, nfkc=nfkc, drop_punctuation=punctdrop), k=1)[0][0].replace("__label__", "")
                for x in xs
            ],
            test_path=test_path,
            name=f"fastText | {tag}",
            labels_order=LABELS,
            out_path=ft_out,
        )
        ft_acc, ft_f1 = read_metrics(ft_out)

        rows.append(
            {
                "config": tag,
                "nb_acc": nb_acc,
                "nb_macro_f1": nb_f1,
                "langid_acc": li_acc,
                "langid_macro_f1": li_f1,
                "fasttext_acc": ft_acc,
                "fasttext_macro_f1": ft_f1,
            }
        )

    # Save combined
    combined = out_dir / "preproc_ablation_summary.json"
    combined.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", combined)

    # Print compact table (sorted by NB accuracy descending)
    rows_sorted = sorted(rows, key=lambda r: r["nb_acc"], reverse=True)

    # Plot results
    cfgs = [r["config"] for r in rows]
    nb_acc = [r["nb_acc"] for r in rows]

    x = np.arange(len(cfgs))

    plt.figure(figsize=(10, 3.5))
    plt.bar(x, nb_acc, color="#1f77b4")  

    plt.xticks(x, [short_cfg_label(c) for c in cfgs], rotation=45, ha="right")
    plt.xlabel("Preprocessing config (l=lower, n=nfkc, p=punctdrop)")
    plt.ylabel("Accuracy")
    plt.title("Preprocessing ablation (NB)")
    plt.ylim(0.85, 1.0)   # manuel
    plt.grid(True, axis="y", alpha=0.3)

    fig_path = out_dir / "fig_preprocessing_ablation_nb.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print("Saved figure:", fig_path)


if __name__ == "__main__":
    main()
