from __future__ import annotations
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from evaluate import evaluate


class CharNgramNB:
    def __init__(self, n_min: int = 1, n_max: int = 5, alpha: float = 0.1):
        self.n_min = n_min
        self.n_max = n_max
        self.alpha = alpha
        self.class_counts = Counter()
        self.ngram_counts: Dict[str, Counter] = defaultdict(Counter)
        self.total_ngrams = Counter()
        self.vocab = set()
        self.labels = []

    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().split())

    def _extract_ngrams(self, text: str) -> List[str]:
        text = self._normalize(text)
        ngrams = []
        L = len(text)
        for n in range(self.n_min, self.n_max + 1):
            for i in range(L - n + 1):
                ngrams.append(text[i:i+n])
        return ngrams

    def fit(self, data: List[Tuple[str, str]]) -> None:
        for text, label in data:
            self.class_counts[label] += 1
            ngrams = self._extract_ngrams(text)
            for ng in ngrams:
                self.ngram_counts[label][ng] += 1
                self.total_ngrams[label] += 1
                self.vocab.add(ng)
        self.labels = sorted(self.class_counts.keys())

    def predict_one(self, text: str) -> str:
        ngrams = self._extract_ngrams(text)
        scores = {}

        V = len(self.vocab)
        total_docs = sum(self.class_counts.values())

        for c in self.labels:
            # log prior
            logp = math.log(self.class_counts[c] / total_docs)

            denom = self.total_ngrams[c] + self.alpha * V
            counts_c = self.ngram_counts[c]

            for ng in ngrams:
                cnt = counts_c.get(ng, 0)
                logp += math.log((cnt + self.alpha) / denom)

            scores[c] = logp

        return max(scores, key=scores.get)

    def predict(self, texts: List[str]) -> List[str]:
        return [self.predict_one(t) for t in texts]
    
    def predict_with_scores(self, texts: List[str]):
        """
        Returns:
            preds:   top-1 predicted labels
            margins: log-posterior margin (s1 - s2)
        """
        preds = []
        margins = []

        V = len(self.vocab)
        total_docs = sum(self.class_counts.values())

        for text in texts:
            ngrams = self._extract_ngrams(text)
            scores = {}

            for c in self.labels:
                # log prior
                logp = math.log(self.class_counts[c] / total_docs)

                denom = self.total_ngrams[c] + self.alpha * V
                counts_c = self.ngram_counts[c]

                for ng in ngrams:
                    cnt = counts_c.get(ng, 0)
                    logp += math.log((cnt + self.alpha) / denom)

                scores[c] = logp

            # sort and take top-2
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            (c1, s1), (_, s2) = sorted_scores[:2]

            preds.append(c1)
            margins.append(s1 - s2)

        return preds, margins




def load_train(path: Path) -> List[Tuple[str, str]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append((obj["text"], obj["lang"]))
    return data


if __name__ == "__main__":
    train_path = Path("data/train.jsonl")
    test_path = Path("data/test.jsonl")

    train_data = load_train(train_path)

    model = CharNgramNB(n_min=1, n_max=5, alpha=0.1)
    model.fit(train_data)

    labels = ["en","de","fr","es","it","tr","nl","sv","pl","ru","ar","zh"]

    evaluate(
        predictor=model.predict,
        test_path=test_path,
        name="Char n-gram NB (1–5, alpha=0.1)",
        labels_order=labels,
        out_path=Path("results/char_ngram_nb_v1.json"),
    )

