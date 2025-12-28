from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

from own_char_ngram_nb import CharNgramNB, load_train


LABELS = ["en","de","fr","es","it","tr","nl","sv","pl","ru","ar","zh"]


def load_jsonl(path: Path) -> List[Tuple[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            rows.append((o["text"], o["lang"]))
    return rows


def norm(t: str) -> str:
    # match model normalization
    return " ".join(t.lower().split())


def ngrams_count(s: str, n_min: int, n_max: int) -> int:
    L = len(s)
    total = 0
    for n in range(n_min, n_max + 1):
        total += max(0, L - n + 1)
    return max(1, total)


def log_posteriors(model: CharNgramNB, text: str) -> Dict[str, float]:
    """
    Recompute log-posterior scores exactly as in predict_one/predict_with_scores.
    """
    ngrams = model._extract_ngrams(text)  # uses model normalization internally
    V = len(model.vocab)
    total_docs = sum(model.class_counts.values())
    scores: Dict[str, float] = {}

    for c in model.labels:
        logp = math.log(model.class_counts[c] / total_docs)  # log prior
        denom = model.total_ngrams[c] + model.alpha * V
        counts_c = model.ngram_counts[c]
        for ng in ngrams:
            cnt = counts_c.get(ng, 0)
            logp += math.log((cnt + model.alpha) / denom)
        scores[c] = logp

    return scores


def top_ngram_contributions(
    model: CharNgramNB,
    text: str,
    c1: str,
    c2: str,
    topk: int = 10,
) -> Dict:
    """
    Contribution of each n-gram to margin = score(c1) - score(c2).

    margin = (log prior c1 - log prior c2) + sum_ng count(ng in doc) * [
        log((cnt_c1+alpha)/denom_c1) - log((cnt_c2+alpha)/denom_c2)
    ]
    """
    # get doc ngrams + counts
    ngrams = model._extract_ngrams(text)
    doc_counts: Dict[str, int] = {}
    for ng in ngrams:
        doc_counts[ng] = doc_counts.get(ng, 0) + 1

    V = len(model.vocab)
    total_docs = sum(model.class_counts.values())

    # prior contribution
    prior_c1 = math.log(model.class_counts[c1] / total_docs)
    prior_c2 = math.log(model.class_counts[c2] / total_docs)
    prior_delta = prior_c1 - prior_c2

    denom1 = model.total_ngrams[c1] + model.alpha * V
    denom2 = model.total_ngrams[c2] + model.alpha * V
    counts1 = model.ngram_counts[c1]
    counts2 = model.ngram_counts[c2]

    contribs = []
    for ng, k in doc_counts.items():
        cnt1 = counts1.get(ng, 0)
        cnt2 = counts2.get(ng, 0)

        w1 = math.log((cnt1 + model.alpha) / denom1)
        w2 = math.log((cnt2 + model.alpha) / denom2)

        per_occ = w1 - w2
        total = k * per_occ

        contribs.append(
            {
                "ngram": ng,
                "count_in_doc": int(k),
                "per_occ_delta": float(per_occ),
                "total_delta": float(total),
                "cnt_in_c1": int(cnt1),
                "cnt_in_c2": int(cnt2),
            }
        )

    # sort by absolute effect on margin (most influential)
    contribs.sort(key=lambda d: abs(d["total_delta"]), reverse=True)
    top = contribs[:topk]

    return {
        "prior_delta": float(prior_delta),
        "top_contributions": top,
    }


def main() -> None:
    train_path = Path("data/train.jsonl")
    test_path = Path("data/test.jsonl")
    out_path = Path("results/nb_interpretability_example.json")

    # train NB
    train = load_train(train_path)
    model = CharNgramNB(n_min=1, n_max=5, alpha=0.1)  # <- your stated model: 3–5 grams
    model.fit(train)

    # load test
    test = load_jsonl(test_path)
    texts = [t for t, _ in test]
    gold = [y for _, y in test]

    # find "most confident wrong" by ngram-normalized margin
    preds, margins = model.predict_with_scores(texts)

    best_i = None
    best_norm_margin = -1e18

    for i, (p, g, m) in enumerate(zip(preds, gold, margins)):
        if p == g:
            continue
        txt_norm = norm(texts[i])
        ng = ngrams_count(txt_norm, model.n_min, model.n_max)
        m_norm = float(m) / float(ng)
        if m_norm > best_norm_margin:
            best_norm_margin = m_norm
            best_i = i

    if best_i is None:
        print("No misclassifications found on test set. (Model may be perfect on this split.)")
        return

    text = texts[best_i]
    text_n = norm(text)
    scores = log_posteriors(model, text)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    (c1, s1), (c2, s2) = ranked[0], ranked[1]

    margin_raw = float(s1 - s2)
    margin_norm = margin_raw / float(ngrams_count(text_n, model.n_min, model.n_max))

    contrib = top_ngram_contributions(model, text, c1=c1, c2=c2, topk=10)

    example = {
        "index": int(best_i),
        "gold": gold[best_i],
        "pred": c1,
        "runner_up": c2,
        "margin_raw": float(margin_raw),
        "margin_norm_ngram": float(margin_norm),
        "text_snippet": text_n[:300],
        "prior_delta": contrib["prior_delta"],
        "top_contributions": contrib["top_contributions"],
        "note": "top_contributions sorted by |total_delta| for margin=score(pred)-score(runner_up)",
        "model": {"n_min": model.n_min, "n_max": model.n_max, "alpha": model.alpha},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(example, indent=2, ensure_ascii=False), encoding="utf-8")

    # pretty print
    print("\n=== NB interpretability ===")
    print("Selection : most confident wrong (ngram-normalized margin)")
    print(f"Index     : {example['index']}")
    print(f"Gold      : {example['gold']}")
    print(f"Pred      : {example['pred']} (runner-up: {example['runner_up']})")
    print(f"Margin    : raw={example['margin_raw']:.4f} | norm(ngrams)={example['margin_norm_ngram']:.6f}")
    print(f"Text      : {example['text_snippet']!r}")
    print(f"Prior Δ   : {example['prior_delta']:.4f}")
    print("\nTop n-gram contributions (toward pred vs runner-up):")
    for j, d in enumerate(example["top_contributions"], 1):
        print(
            f"{j:02d}. {d['ngram']!r}  "
            f"count={d['count_in_doc']}  "
            f"Δ/occ={d['per_occ_delta']:+.4f}  "
            f"Δtotal={d['total_delta']:+.4f}  "
            f"cnt_pred={d['cnt_in_c1']} cnt_2nd={d['cnt_in_c2']}"
        )

    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()

