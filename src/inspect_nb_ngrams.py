from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

from own_char_ngram_nb import CharNgramNB


LABELS = ["en","de","fr","es","it","tr","nl","sv","pl","ru","ar","zh"]


def load_jsonl(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append((obj["text"], obj["lang"]))
    return rows


def ngrams_count(s: str, n_min: int, n_max: int) -> int:
    L = len(s)
    total = 0
    for n in range(n_min, n_max + 1):
        total += max(0, L - n + 1)
    return max(1, total)


def log_posteriors(model: CharNgramNB, text: str) -> Dict[str, float]:
    """Manually re-calculate scores to inspect the process."""
    text_n = model.normalize(text)
    ngrams = model.extract_ngrams(text_n)
    V = len(model.vocab)
    total_docs = sum(model.class_counts.values())
    scores: Dict[str, float] = {}

    for c in model.labels:
        # log P(c)
        logp = math.log(model.class_counts[c] / total_docs)  
        denom = model.total_ngrams[c] + model.alpha * V
        counts_c = model.ngram_counts[c]
        # Likelihoods
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
    """Find which n-grams push the margin between top two classes."""
    text_n = model.normalize(text)
    ngrams = model.extract_ngrams(text_n)

    doc_counts: Dict[str, int] = {}
    for ng in ngrams:
        doc_counts[ng] = doc_counts.get(ng, 0) + 1

    V = len(model.vocab)
    total_docs = sum(model.class_counts.values())

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

    contribs.sort(key=lambda d: abs(d["total_delta"]), reverse=True)
    top = contribs[:topk]

    return {
        "prior_delta": float(prior_delta),
        "top_contributions": top,
    }


def export_ngram_inspection_json(
    model: CharNgramNB,
    n: int = 3,
    topk: int = 20,
    out_path: Path = Path("results/ngram_inspection.json"),
) -> None:
    """Check which n-grams are most unique to each class."""
    V = len(model.vocab)
    total_all = sum(model.total_ngrams[c] for c in model.labels)

    # global counts for specific n-gram length
    counts_all: Dict[str, int] = {}
    for c in model.labels:
        for ng, cnt in model.ngram_counts[c].items():
            if len(ng) == n:
                counts_all[ng] = counts_all.get(ng, 0) + cnt

    result: Dict[str, object] = {
        "name": f"NB global n-gram inspection (n={n})",
        "n": int(n),
        "topk": int(topk),
        "alpha": float(model.alpha),
        "labels": list(model.labels),
        "method": "log_odds(log P(ng|c) - log P(ng|not c))",
        "top_ngrams": {},
    }

    top_ngrams: Dict[str, List[Dict[str, object]]] = {}

    for c in model.labels:
        denom_c = model.total_ngrams[c] + model.alpha * V
        denom_notc = (total_all - model.total_ngrams[c]) + model.alpha * V
        counts_c = model.ngram_counts[c]

        scored: List[Dict[str, object]] = []
        for ng, cnt_all in counts_all.items():
            cnt_c = counts_c.get(ng, 0)
            cnt_notc = cnt_all - cnt_c

            logp_c = math.log((cnt_c + model.alpha) / denom_c)
            logp_notc = math.log((cnt_notc + model.alpha) / denom_notc)

            scored.append(
                {
                    "ngram": ng,
                    "log_odds": float(logp_c - logp_notc),
                    "count_in_class": int(cnt_c),
                    "count_outside": int(cnt_notc),
                }
            )

        scored.sort(key=lambda d: d["log_odds"], reverse=True)
        top_ngrams[c] = scored[:topk]

    result["top_ngrams"] = top_ngrams

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    train_path = Path("data/train.jsonl")
    test_path = Path("data/test.jsonl")

    out_example = Path("results/nb_interpretability_example.json")
    out_inspect = Path("results/ngram_inspection.json")

    # Train NB
    train = load_jsonl(train_path)
    model = CharNgramNB(n_min=1, n_max=5, alpha=0.1)
    model.fit(train)

    # n-gram inspection 
    export_ngram_inspection_json(model, n=3, topk=20, out_path=out_inspect)

    # Load test
    test = load_jsonl(test_path)
    texts = [t for t, _ in test]
    gold = [y for _, y in test]

    # Find "most confident wrong" by ngram-normalized margin
    preds, margins = model.predict_with_scores(texts)

    best_i = None
    best_norm_margin = -1e18

    for i, (p, g, m) in enumerate(zip(preds, gold, margins)):
        if p == g:
            continue
        txt_norm = model.normalize(texts[i])
        ng = ngrams_count(txt_norm, model.n_min, model.n_max)
        m_norm = float(m) / float(ng)
        if m_norm > best_norm_margin:
            best_norm_margin = m_norm
            best_i = i

    if best_i is None:
        print("All predictions correct.")
        return

    text = texts[best_i]
    text_n = model.normalize(text)
    scores = log_posteriors(model, text)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    (c1, s1), (c2, s2) = ranked[0], ranked[1]

    margin_raw = float(s1 - s2)
    margin_norm = margin_raw / float(ngrams_count(text_n, model.n_min, model.n_max))
    contrib = top_ngram_contributions(model, text, c1=c1, c2=c2, topk=10)

    example = {
        "name": "NB interpretability: most confident wrong (ngram-normalized margin)",
        "index": int(best_i),
        "gold": gold[best_i],
        "pred": c1,
        "runner_up": c2,
        "margin_raw": float(margin_raw),
        "margin_norm_ngram": float(margin_norm),
        "text_snippet": text_n[:300],
        "prior_delta": float(contrib["prior_delta"]),
        "top_contributions": contrib["top_contributions"],
        "note": "top_contributions sorted by |total_delta| for margin=score(pred)-score(runner_up)",
        "model": {"n_min": int(model.n_min), "n_max": int(model.n_max), "alpha": float(model.alpha)},
    }

    out_example.parent.mkdir(parents=True, exist_ok=True)
    out_example.write_text(json.dumps(example, indent=2, ensure_ascii=False), encoding="utf-8")

    # Pretty print
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

    print("\nSaved example JSON :", out_example)
    print("Saved ngram JSON   :", out_inspect)


if __name__ == "__main__":
    main()
