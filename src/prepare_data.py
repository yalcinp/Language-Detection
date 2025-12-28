from __future__ import annotations
import json
import random
from pathlib import Path
from collections import Counter
from datasets import load_dataset

OUT_DIR = Path("data")
SEED = 42
DATASET_NAME = "MartinThoma/wili_2018"
DEV_FRAC = 0.10  # dev fraction per language

# ISO-639-3 -> ISO-639-1
LANGS_ISO3 = ["eng","deu","fra","spa","ita","tur","nld","swe","pol","rus","ara","zho"]
ISO3_TO_ISO2 = {
    "eng": "en", "deu": "de", "fra": "fr", "spa": "es", "ita": "it", "tur": "tr",
    "nld": "nl", "swe": "sv", "pol": "pl", "rus": "ru", "ara": "ar", "zho": "zh",
}


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

    # Pick text column
    cols = train.column_names
    if "sentence" in cols:
        text_col = "sentence"
    elif "text" in cols:
        text_col = "text"
    else:
        raise ValueError(f"No text column found. Columns: {cols}")

    # Label mapping
    if "label" not in cols:
        raise ValueError(f"No label column found. Columns: {cols}")
    label_col = "label"

    label_feature = train.features[label_col]
    if not hasattr(label_feature, "int2str"):
        raise ValueError("Expected ClassLabel with int2str for 'label' feature.")

    def iso3_of(ex) -> str:
        return label_feature.int2str(ex[label_col])  # we expect labels like: "swe"

    def collect(split, name: str) -> list[dict]:
        rows = []
        cnt_seen = Counter()
        cnt_kept = Counter()

        for ex in split:
            iso3 = iso3_of(ex)
            cnt_seen[iso3] += 1
            if iso3 in LANGS_ISO3:
                iso2 = ISO3_TO_ISO2[iso3]
                rows.append({"text": ex[text_col], "lang": iso2})
                cnt_kept[iso3] += 1

        print(f"[{name}] total={len(split)} kept={len(rows)}")
        print(f"[{name}] kept per iso3:", {k: cnt_kept.get(k, 0) for k in LANGS_ISO3})
        print(f"[{name}] top10 seen:", cnt_seen.most_common(10))

        return rows

    train_rows = collect(train, "train")
    test_rows = collect(test, "test")

    if len(train_rows) == 0 or len(test_rows) == 0:
        raise RuntimeError("No rows collected. Something is wrong with label mapping or dataset.")

    # Sample dev per language
    by_lang = {}
    for r in train_rows:
        by_lang.setdefault(r["lang"], []).append(r)

    dev_rows = []
    train_rows2 = []

    for lang, items in by_lang.items():
        random.shuffle(items)
        n_lang = len(items)
        n_dev_lang = max(1, int(DEV_FRAC * n_lang))
        dev_rows.extend(items[:n_dev_lang])
        train_rows2.extend(items[n_dev_lang:])

    random.shuffle(dev_rows)
    random.shuffle(train_rows2)

    # Per-language counts
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
    print("Languages:", sorted(set(ISO3_TO_ISO2[x] for x in LANGS_ISO3)))


if __name__ == "__main__":
    main()
