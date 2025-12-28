from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import fasttext
from py3langid import langid

from evaluate import evaluate
from own_char_ngram_nb import CharNgramNB, load_train

LABELS = ["en","de","fr","es","it","tr","nl","sv","pl","ru","ar","zh"]


# -------------------------
# preprocessing
# -------------------------
def drop_punct(s: str) -> str:
    # remove Unicode punctuation (category starts with "P")
    out = []
    for ch in s:
        if not unicodedata.category(ch).startswith("P"):
            out.append(ch)
    return "".join(out)


def preprocess(text: str, *, lower: bool, nfkc: bool, drop_punctuation: bool) -> str:
    t = text
    if nfkc:
        t = unicodedata.normalize("NFKC", t)
    if lower:
        t = t.lower()
    if drop_punctuation:
        t = drop_punct(t)
    # keep whitespace normalization fixed across settings
    t = " ".join(t.split())
    return t


def cfg_name(lower: bool, nfkc: bool, drop_punctuation: bool) -> str:
    return f"lower={int(lower)} nfkc={int(nfkc)} punctdrop={int(drop_punctuation)}"


# -------------------------
# helpers to read back metrics from evaluate() json
# -------------------------
def read_metrics(path: Path) -> Tuple[float, float]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    # evaluate.py outputs at least: accuracy, macro_f1 (based on your logs)
    return float(obj["accuracy"]), float(obj["macro_f1"])


# -------------------------
# main
# -------------------------
def main() -> None:
    train_path = Path("data/train.jsonl")
    test_path = Path("data/test.jsonl")

    # fastText + langid init once
    ft = fasttext.load_model(str(Path("models/lid.176.bin")))
    langid.set_languages(LABELS)

    # load train once (we will preprocess copies per config)
    train_raw = load_train(train_path)  # List[Tuple[text, lang]]

    out_dir = Path("results/preproc_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []

    # 8 configs
    configs = []
    for lower in (False, True):
        for nfkc in (False, True):
            for punctdrop in (False, True):
                configs.append((lower, nfkc, punctdrop))

    for lower, nfkc, punctdrop in configs:
        tag = cfg_name(lower, nfkc, punctdrop)

        # ---------- NB: retrain with this preprocessing ----------
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

        # ---------- langid ----------
        li_out = out_dir / f"langid_{tag}.json"
        evaluate(
            predictor=lambda xs: [langid.classify(preprocess(x, lower=lower, nfkc=nfkc, drop_punctuation=punctdrop))[0] for x in xs],
            test_path=test_path,
            name=f"langid | {tag}",
            labels_order=LABELS,
            out_path=li_out,
        )
        li_acc, li_f1 = read_metrics(li_out)

        # ---------- fastText ----------
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

    # save combined
    combined = out_dir / "preproc_ablation_summary.json"
    combined.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print("Saved:", combined)

    # print compact table (sorted by NB accuracy descending)
    rows_sorted = sorted(rows, key=lambda r: r["nb_acc"], reverse=True)

    print("\nPreprocessing ablation (8 configs):")
    print("config\tNB_acc\tNB_F1\tlangid_acc\tlangid_F1\tfastText_acc\tfastText_F1")
    for r in rows_sorted:
        print(
            f"{r['config']}\t"
            f"{r['nb_acc']:.4f}\t{r['nb_macro_f1']:.4f}\t"
            f"{r['langid_acc']:.4f}\t{r['langid_macro_f1']:.4f}\t"
            f"{r['fasttext_acc']:.4f}\t{r['fasttext_macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()

