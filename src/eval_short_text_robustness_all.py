from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from own_char_ngram_nb import CharNgramNB, load_train
import langid
import fasttext


LABELS = ["en","de","fr","es","it","tr","nl","sv","pl","ru","ar","zh"]
LENGTHS = [5, 10, 20, 30, 50, 100, 200]


def load_test(path: Path) -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            texts.append(o["text"])
            labels.append(o["lang"])
    return texts, labels

def norm(t: str) -> str:
    return " ".join(t.lower().split())

def trunc(t: str, L: int) -> str:
    return norm(t)[:L]

def accuracy(preds: List[str], gold: List[str]) -> float:
    return sum(p == g for p, g in zip(preds, gold)) / len(gold)


def main() -> None:
    train_path = Path("data/train.jsonl")
    test_path = Path("data/test.jsonl")
    ft_path = Path("models/lid.176.bin")

    # data
    train = load_train(train_path)
    texts_full, gold = load_test(test_path)

    # NB
    nb = CharNgramNB(n_min=1, n_max=5, alpha=0.1)
    nb.fit(train)

    # langid + fastText
    langid.set_languages(LABELS)
    ft = fasttext.load_model(str(ft_path))

    def pred_nb(xs: List[str]) -> List[str]:
        return [nb.predict_one(x) for x in xs]

    def pred_langid(xs: List[str]) -> List[str]:
        return [langid.classify(x)[0] for x in xs]

    def pred_fasttext(xs: List[str]) -> List[str]:
        return [ft.predict(x, k=1)[0][0].replace("__label__", "") for x in xs]

    # Full - no truncation
    texts_full_norm = [norm(x) for x in texts_full]
    full_nb = accuracy(pred_nb(texts_full_norm), gold)
    full_li = accuracy(pred_langid(texts_full_norm), gold)
    full_ft = accuracy(pred_fasttext(texts_full_norm), gold)

    print(f"FULL (no trunc) | NB={full_nb:.4f} | langid={full_li:.4f} | fastText={full_ft:.4f}")

    rows: List[Dict] = []
    ys_nb, ys_li, ys_ft = [], [], []

    for L in LENGTHS:
        xs = [trunc(x, L) for x in texts_full]

        a_nb = accuracy(pred_nb(xs), gold)
        a_li = accuracy(pred_langid(xs), gold)
        a_ft = accuracy(pred_fasttext(xs), gold)

        rows.append({"L": L, "NB": a_nb, "langid": a_li, "fastText": a_ft})
        ys_nb.append(a_nb)
        ys_li.append(a_li)
        ys_ft.append(a_ft)

        print(f"L={L:3d} | NB={a_nb:.4f} | langid={a_li:.4f} | fastText={a_ft:.4f}")

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "short_text_robustness_all.json"
    out_json.write_text(
        json.dumps(
            {
                "note": "Accuracy vs input length. Texts are lowercased + whitespace-normalized; truncation uses first L characters.",
                "full_no_trunc": {"NB": full_nb, "langid": full_li, "fastText": full_ft},
                "lengths": LENGTHS,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("Saved:", out_json)

    # plot
    plt.figure(figsize=(6.5, 4))
    plt.plot(LENGTHS, ys_nb, marker="o", label="NB (char n-gram)")
    plt.plot(LENGTHS, ys_li, marker="o", label="langid")
    plt.plot(LENGTHS, ys_ft, marker="o", label="fastText")

    plt.xlabel("Input length (characters)")
    plt.ylabel("Accuracy")
    plt.title("Short-text robustness")

    plt.xticks(LENGTHS, [str(L) for L in LENGTHS])
    plt.ylim(0.5, 1.01)          
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    out_png = out_dir / "short_text_robustness_all.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    print("Saved:", out_png, "and", out_png.with_suffix(".pdf"))



if __name__ == "__main__":
    main()

