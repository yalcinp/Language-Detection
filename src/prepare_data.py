from __future__ import annotations

import json
import random
from pathlib import Path
from collections import Counter

from datasets import load_dataset

OUT_DIR = Path("data")
SEED = 42

# WiLI-2018 labels are ISO-639-3 (e.g., swe, rus, ara)
LANGS_12_ISO3 = ["eng", "deu", "fra", "spa", "ita", "tur", "nld", "swe", "pol", "rus", "ara", "zho"]

ISO3_TO_ISO2 = {
    "eng": "en",
    "deu": "de",
    "fra": "fr",
    "spa": "es",
    "ita": "it",
    "tur": "tr",
    "nld": "nl",
    "swe": "sv",
    "pol": "pl",
    "rus": "ru",
    "ara": "ar",
    "zho": "zh",
}

DATASET_NAME = "MartinThoma/wili_2018"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    random.seed(SEED)

    ds = load_dataset(DATASET_NAME)
    train = ds["train"]
    test = ds["test"]

    # Find text column robustly
    cols = train.column_names
    if "sentence" in cols:
        text_col = "sentence"
    elif "text" in cols:
        text_col = "text"
    else:
        raise ValueError(f"No text column found. Columns: {cols}")

    # Label column must exist
    if "label" not in cols:
        raise ValueError(f"No label column found. Columns: {cols}")
    label_col = "label"

    label_feature = train.features[label_col]
    if not hasattr(label_feature, "int2str"):
        raise ValueError("Expected ClassLabel with int2str for 'label' feature.")

    def iso3_of(ex) -> str:
        return label_feature.int2str(ex[label_col])  # e.g., "swe"

    def collect(split, split_name: str):
        rows = []
        cnt_seen = Counter()
        cnt_kept = Counter()

        for ex in split:
            iso3 = iso3_of(ex)
            cnt_seen[iso3] += 1
            if iso3 in LANGS_12_ISO3:
                iso2 = ISO3_TO_ISO2[iso3]
                rows.append({"text": ex[text_col], "lang": iso2})
                cnt_kept[iso3] += 1

        print(f"[{split_name}] total={len(split)} kept={len(rows)}")
        print(f"[{split_name}] kept per iso3:", {k: cnt_kept.get(k, 0) for k in LANGS_12_ISO3})

        # sanity: show top 10 labels in the split
        top10 = cnt_seen.most_common(10)
        print(f"[{split_name}] top10 seen:", top10)

        return rows

    train_rows = collect(train, "train")
    test_rows = collect(test, "test")

    if len(train_rows) == 0 or len(test_rows) == 0:
        raise RuntimeError("No rows collected. Something is wrong with label mapping or dataset.")

    # Create STRATIFIED dev split from train (10% per language)
    by_lang = {}
    for r in train_rows:
        by_lang.setdefault(r["lang"], []).append(r)

    dev_rows = []
    train_rows2 = []

    for lang, items in by_lang.items():
        random.shuffle(items)
        n_lang = len(items)
        n_dev_lang = max(1, int(0.10 * n_lang))  # at least 1 per language
        dev_rows.extend(items[:n_dev_lang])
        train_rows2.extend(items[n_dev_lang:])

    # Optional: shuffle final splits
    random.shuffle(dev_rows)
    random.shuffle(train_rows2)

    # Sanity: print per-language counts
    c_train = Counter(r["lang"] for r in train_rows2)
    c_dev = Counter(r["lang"] for r in dev_rows)
    print("[dev split] per-lang counts:")
    for lang in sorted(by_lang.keys()):
        print(f"  {lang}: train={c_train[lang]} dev={c_dev[lang]}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "train.jsonl", train_rows2)
    write_jsonl(OUT_DIR / "dev.jsonl", dev_rows)
    write_jsonl(OUT_DIR / "test.jsonl", test_rows)

    print(f"Saved: train={len(train_rows2)} dev={len(dev_rows)} test={len(test_rows)}")
    print("Languages (iso3):", LANGS_12_ISO3)
    print("Languages (iso2):", sorted(set(ISO3_TO_ISO2.values())))


if __name__ == "__main__":
    main()
