from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from own_char_ngram_nb import CharNgramNB
import langid
import fasttext


LABELS = ["en", "de", "fr", "es", "it", "tr", "nl", "sv", "pl", "ru", "ar", "zh"]
LABEL_SET = set(LABELS)


def load_jsonl(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            rows.append((o["text"], o["lang"]))
    return rows


def norm(t: str) -> str:
    return " ".join(t.lower().split())


def shortk(t: str, k: int) -> str:
    return norm(t)[:k]


def ngrams_count(s: str, n_min: int, n_max: int) -> int:
    """Calculate possible n-grams for length normalization."""
    L = len(s)
    total = 0
    for n in range(n_min, n_max + 1):
        total += max(0, L - n + 1)
    return max(1, total)


def build_curve(scores: np.ndarray, preds: List[str], gold: List[str], percentiles=None) -> Dict:
    """Calculates coverage/risk trade-off on confidence percentiles."""
    if percentiles is None:
        percentiles = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]

    taus = np.percentile(scores, percentiles).tolist()
    N = len(gold)

    curve = []
    for tau in taus:
        # keep only samples in confidence threshold
        kept = [i for i, si in enumerate(scores) if si >= tau]
        n_kept = len(kept)
        coverage = n_kept / N if N > 0 else 0.0

        if n_kept == 0:
            acc = None
            risk = None
        else:
            correct = sum(1 for i in kept if preds[i] == gold[i])
            acc = correct / n_kept
            risk = 1.0 - acc

        curve.append(
            {
                "tau": float(tau),
                "coverage": float(coverage),
                "accuracy": None if acc is None else float(acc),
                "risk": None if risk is None else float(risk),
                "n_kept": int(n_kept),
            }
        )

    return {"curve": curve, "percentiles": percentiles}


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved JSON:", path)


def plot_risk_curves(
    json_paths: List[Path],
    title: str,
    out_path: Path,
    ylim=(0.0, 0.2),
) -> None:
    plt.figure()
    for p in json_paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        name = data.get("name", p.stem)
        xs = [r["coverage"] for r in data["curve"] if r["risk"] is not None]
        ys = [r["risk"] for r in data["curve"] if r["risk"] is not None]
        plt.plot(xs, ys, marker="o", label=name)

    plt.xlabel("Coverage")
    plt.ylabel("Risk (1 - accuracy on kept samples)")
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.ylim(*ylim)
    plt.grid(True)
    plt.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print("Saved plot:", out_path, "and", out_path.with_suffix(".pdf"))


def plot_risk_panels(
    panels: List[Tuple[str, List[Path], Tuple[float, float]]],
    out_pdf: Path,
) -> None:
    
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), sharex=True)

    if n == 1:
        axes = [axes]

    legend_handles = None
    legend_labels = None

    for ax, (title, json_paths, ylim) in zip(axes, panels):
        for p in json_paths:
            data = json.loads(p.read_text(encoding="utf-8"))
            name = data.get("name", p.stem)
            xs = [r["coverage"] for r in data["curve"] if r["risk"] is not None]
            ys = [r["risk"] for r in data["curve"] if r["risk"] is not None]
            ax.plot(xs, ys, marker="o", label=name)

        ax.set_title(title)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Risk (1 - accuracy on kept samples)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=3, frameon=False)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")  # PDF = vector, dpi yok
    plt.close(fig)
    print("Saved combined panels:", out_pdf)



def nb_scores(train: List[Tuple[str, str]], texts: List[str]) -> Tuple[List[str], np.ndarray]:
    nb = CharNgramNB(n_min=1, n_max=5, alpha=0.1)
    nb.fit(train)
    preds, margins = nb.predict_with_scores(texts)

    m = np.array(margins, dtype=float)
    ng = np.array([ngrams_count(t, 1, 5) for t in texts], dtype=float)  # ngram-normalize
    scores = m / ng
    return preds, scores


def langid_scores(texts: List[str]) -> Tuple[List[str], np.ndarray]:
    langid.set_languages(LABELS)
    preds: List[str] = []
    scores: List[float] = []
    for t in texts:
        ranked = langid.rank(t)  # list[(lang, score)]
        ranked = [(l, s) for (l, s) in ranked if l in LABEL_SET]
        ranked.sort(key=lambda x: x[1], reverse=True)

        if len(ranked) == 0:
            preds.append("unk")
            scores.append(0.0)
            continue

        if len(ranked) == 1:
            l1, s1 = ranked[0]
            preds.append(l1)
            scores.append(float(s1) / max(1, len(t)))
            continue

        (l1, s1), (l2, s2) = ranked[0], ranked[1]
        preds.append(l1)
        margin = float(s1) - float(s2)
        scores.append(margin / max(1, len(t)))  # length-normalize
    return preds, np.array(scores, dtype=float)


def fasttext_scores(ft_model, texts: List[str]) -> Tuple[List[str], np.ndarray]:
    """fastText: probability margin"""
    preds: List[str] = []
    scores: List[float] = []
    for t in texts:
        labs, probs = ft_model.predict(t, k=2)
        if len(labs) == 0:
            preds.append("unk")
            scores.append(0.0)
            continue

        l1 = labs[0].replace("__label__", "")
        p1 = float(probs[0])
        p2 = float(probs[1]) if len(probs) > 1 else 0.0

        preds.append(l1)
        scores.append(p1 - p2)  # already in [0,1]
    return preds, np.array(scores, dtype=float)


def run_setting(
    tag: str,
    train: List[Tuple[str, str]],
    texts: List[str],
    gold: List[str],
    ft_model,
) -> List[Path]:
    out_jsons: List[Path] = []

    # NB
    nb_p, nb_s = nb_scores(train, texts)
    obj = build_curve(nb_s, nb_p, gold)
    obj.update(
        {
            "name": f"NB ({tag})",
            "setting": tag,
            "confidence": "margin(top1-top2), ngram-normalized (#1-5 char-ngrams)",
        }
    )
    p = Path(f"results/risk_nb_{tag}.json")
    save_json(obj, p)
    out_jsons.append(p)

    # langid
    li_p, li_s = langid_scores(texts)
    obj = build_curve(li_s, li_p, gold)
    obj.update(
        {
            "name": f"langid ({tag})",
            "setting": tag,
            "confidence": "margin(top1-top2), length-normalized",
        }
    )
    p = Path(f"results/risk_langid_{tag}.json")
    save_json(obj, p)
    out_jsons.append(p)

    # fastText (reuse loaded model)
    ft_p, ft_s = fasttext_scores(ft_model, texts)
    obj = build_curve(ft_s, ft_p, gold)
    obj.update(
        {
            "name": f"fastText ({tag})",
            "setting": tag,
            "confidence": "prob margin (p1-p2)",
        }
    )
    p = Path(f"results/risk_fasttext_{tag}.json")
    save_json(obj, p)
    out_jsons.append(p)

    return out_jsons


def main() -> None:
    train_path = Path("data/train.jsonl")
    test_path = Path("data/test.jsonl")
    ft_model_path = Path("models/lid.176.bin")  # adjust if different

    if not train_path.exists():
        raise FileNotFoundError(f"Missing: {train_path.resolve()} (run prepare_data.py first)")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing: {test_path.resolve()} (run prepare_data.py first)")
    if not ft_model_path.exists():
        raise FileNotFoundError(f"Missing fastText model: {ft_model_path.resolve()}")

    # Load fastText 
    ft_model = fasttext.load_model(str(ft_model_path))

    train = load_jsonl(train_path)
    test = load_jsonl(test_path)
    texts_full = [t for t, _ in test]
    gold = [y for _, y in test]

    # Full
    jsons_full = run_setting("full", train, texts_full, gold, ft_model)
    plot_risk_curves(
        jsons_full,
        title="Coverage–Risk (full text)",
        out_path=Path("results/coverage_risk_full.png"),
        ylim=(0.0, 0.05),
    )

    # SHORT50
    texts_50 = [shortk(x, 50) for x in texts_full]
    jsons_50 = run_setting("short50", train, texts_50, gold, ft_model)
    plot_risk_curves(
        jsons_50,
        title="Coverage–Risk (short50)",
        out_path=Path("results/coverage_risk_short50.png"),
        ylim=(0.0, 0.15),
    )

    # SHORT20
    texts_20 = [shortk(x, 20) for x in texts_full]
    jsons_20 = run_setting("short20", train, texts_20, gold, ft_model)
    plot_risk_curves(
        jsons_20,
        title="Coverage–Risk (short20)",
        out_path=Path("results/coverage_risk_short20.png"),
        ylim=(0.0, 0.25),
    )

    # all combined figure 
    plot_risk_panels(
        panels=[
            ("(a) Full",    jsons_full, (0.0, 0.05)),
            ("(b) Short50", jsons_50,   (0.0, 0.15)),
            ("(c) Short20", jsons_20,   (0.0, 0.25)),
        ],
        out_pdf=Path("results/coverage_risk_panels.pdf"),
    )



if __name__ == "__main__":
    main()
