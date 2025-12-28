# WiLI split statistics
from __future__ import annotations
import json
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional


LABELS = ["en","de","fr","es","it","tr","nl","sv","pl","ru","ar","zh"]
LABEL_SET = set(LABELS)


def load_jsonl(path: Path) -> List[Tuple[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            rows.append((o["text"], o["lang"]))
    return rows


def is_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L")


def is_latin_letter(ch: str) -> bool:
    if not is_letter(ch):
        return False
    name = unicodedata.name(ch, "")
    return "LATIN" in name


def split_stats(name: str, rows: List[Tuple[str, str]]) -> Dict:
    """Char n-gram models"""
    texts = [t for t, _ in rows]
    langs = [y for _, y in rows]

    # Length stats
    lengths = [len(t) for t in texts]
    avg_len = sum(lengths) / max(1, len(lengths))
    med_len = statistics.median(lengths) if lengths else 0.0

    lang_counts = Counter(langs) # per-language counts
    # Keep canonical order, but don’t drop unknowns if present
    ordered_lang_counts = {lab: int(lang_counts.get(lab, 0)) for lab in LABELS}
    extra = {k: int(v) for k, v in lang_counts.items() if k not in LABEL_SET}
    if extra:
        ordered_lang_counts["_other"] = extra

    # Script stats (letters only)
    total_letters = 0
    latin_letters = 0
    nonlatin_letters = 0

    # Per-language Latin ratio
    per_lang_letters = defaultdict(int)
    per_lang_latin = defaultdict(int)

    for t, y in rows:
        for ch in t:
            if not is_letter(ch):
                continue
            total_letters += 1
            per_lang_letters[y] += 1
            if is_latin_letter(ch):
                latin_letters += 1
                per_lang_latin[y] += 1
            else:
                nonlatin_letters += 1

    latin_ratio = (latin_letters / total_letters) if total_letters else 0.0
    nonlatin_ratio = (nonlatin_letters / total_letters) if total_letters else 0.0

    per_lang_script = {}
    for y, L in sorted(per_lang_letters.items(), key=lambda kv: kv[0]):
        lat = per_lang_latin[y]
        per_lang_script[y] = {
            "letters": int(L),
            "latin_letters": int(lat),
            "latin_ratio": (lat / L) if L else 0.0,
        }

    return {
        "split": name,
        "n_rows": int(len(rows)),
        "n_languages": int(len(set(langs))),
        "language_counts": ordered_lang_counts,
        "char_length": {
            "avg": float(avg_len),
            "median": float(med_len),
            "min": int(min(lengths)) if lengths else 0,
            "max": int(max(lengths)) if lengths else 0,
        },
        "script_letters_only": {
            "total_letters": int(total_letters),
            "latin_letters": int(latin_letters),
            "nonlatin_letters": int(nonlatin_letters),
            "latin_ratio": float(latin_ratio),
            "nonlatin_ratio": float(nonlatin_ratio),
        },
        "per_language_script": per_lang_script,
    }


def maybe_load(path: Path) -> Optional[List[Tuple[str, str]]]:
    return load_jsonl(path) if path.exists() else None


def main() -> None:
    data_dir = Path("data")
    train_path = data_dir / "train.jsonl"
    dev_path = data_dir / "dev.jsonl"
    test_path = data_dir / "test.jsonl"

    out_path = Path("results/data_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train = load_jsonl(train_path)
    test = load_jsonl(test_path)
    dev = maybe_load(dev_path)

    summaries = []
    summaries.append(split_stats("train", train))
    if dev is not None:
        summaries.append(split_stats("dev", dev))
    summaries.append(split_stats("test", test))

    out = {
        "label_space": LABELS,
        "splits": summaries,
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", out_path)

    
    print("\nMini data summary")
    for s in summaries:
        cl = s["char_length"]
        sc = s["script_letters_only"]
        print(f"\n[{s['split']}] n={s['n_rows']} | langs={s['n_languages']}")
        print(f"char_len: avg={cl['avg']:.2f} | median={cl['median']:.1f} | min={cl['min']} | max={cl['max']}")
        print(f"script (letters only): latin={sc['latin_ratio']*100:.2f}% | nonlatin={sc['nonlatin_ratio']*100:.2f}%")

        # top-5 languages by count
        lc = s["language_counts"]
        flat = [(k, v) for k, v in lc.items() if k != "_other"] # flatten counts excluding _other
        flat.sort(key=lambda kv: kv[1], reverse=True)
        top5 = ", ".join([f"{k}:{v}" for k, v in flat[:5]])
        print(f"top langs: {top5}")


if __name__ == "__main__":
    main()

