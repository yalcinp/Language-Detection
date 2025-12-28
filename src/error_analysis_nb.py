from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from own_char_ngram_nb import CharNgramNB, load_train
from matplotlib.colors import LogNorm

LABELS = ["en","de","fr","es","it","tr","nl","sv","pl","ru","ar","zh"]
IDX = {l:i for i,l in enumerate(LABELS)}


def load_test(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            rows.append((o["text"], o["lang"]))
    return rows

def plot_confusion_matrix(C: np.ndarray, labels, out_pdf: Path) -> None:

    """Generate and save a normalized confusion matrix plot."""
    M = C.astype(float).copy()

    # mask diagonal + zeros (so the plot shows only actual confusions)
    np.fill_diagonal(M, np.nan)
    M[M == 0] = np.nan

    finite = M[np.isfinite(M)]
    if finite.size == 0:
        print("No off-diagonal errors; skipping confusion plot.")
        return

    vmin = max(1.0, float(np.nanmin(M)))
    vmax = float(np.nanmax(M))

    fig, ax = plt.subplots(figsize=(5.8, 4.8), constrained_layout=True)

    im = ax.imshow(
        M,
        interpolation="nearest",
        aspect="equal",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis",
    )
    im.cmap.set_bad(color="white")

    ax.set_title("Confusion matrix (NB)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Error count (log scale)", rotation=90)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("Saved confusion matrix:", out_pdf)



def main():
    train = load_train(Path("data/train.jsonl"))
    test = load_test(Path("data/test.jsonl"))

    texts = [t for t,_ in test]
    gold  = [y for _,y in test]

    # train NB
    m = CharNgramNB(n_min=1, n_max=5, alpha=0.1)
    m.fit(train)

    preds = [m.predict_one(t) for t in texts]

    # calculate confusion counts
    C = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for g, p in zip(gold, preds):
        C[IDX[g], IDX[p]] += 1

    # per-language stats 
    metrics = {}
    for l in LABELS:
        i = IDX[l]
        tp = C[i, i]
        fp = C[:, i].sum() - tp
        fn = C[i, :].sum() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec / (prec + rec) if (prec + rec) > 0 else 0.0

        metrics[l] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(C[i, :].sum()),
        }

    Path("results").mkdir(exist_ok=True)
    Path("results/per_language_metrics_nb.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # most confused language pairs
    pairs = []
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if i != j and C[i, j] > 0:
                pairs.append((LABELS[i], LABELS[j], int(C[i, j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = pairs[:10]

    Path("results/top_confusions_nb.json").write_text(
        json.dumps(top_pairs, indent=2), encoding="utf-8"
    )

    # export a few sample errors for manual review
    error_cases = []
    for t, g, p in zip(texts, gold, preds):
        if g != p:
            error_cases.append({
                "text": t[:120],
                "gold": g,
                "pred": p,
            })
            if len(error_cases) >= 50:
                break

    Path("results/error_examples_nb.json").write_text(
        json.dumps(error_cases, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("Saved:")
    print("- results/per_language_metrics_nb.json")
    print("- results/top_confusions_nb.json")
    print("- results/error_examples_nb.json")

    # create pdf
    plot_confusion_matrix(
    C=C,
    labels=LABELS,
    out_pdf=Path("results/fig_confusion_matrix_nb.pdf"),
)


    
if __name__ == "__main__":
    main()

