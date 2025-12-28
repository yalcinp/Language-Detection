from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_curve(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    curve = data["curve"]
    xs = [r["coverage"] for r in curve if r["accuracy"] is not None]
    ys = [r["accuracy"] for r in curve if r["accuracy"] is not None]
    name = data.get("name", path.stem)
    return xs, ys, name


def main() -> None:
    paths = [
        Path("results/abstain_nb_full.json"),
        Path("results/abstain_nb_short50.json"),
        Path("results/abstain_nb_short20.json"),
    ]

    # Create the figure 
    plt.figure(figsize=(6, 4))

    for p in paths:
        if not p.exists():
            print(f"Missing: {p}")
            continue
        x, y, name = load_curve(p)
        plt.plot(x, y, marker="o", linewidth=1.5, markersize=3, label=name)

    plt.xlabel("Coverage")
    plt.ylabel("Accuracy (on kept samples)")
    plt.title("NB selective prediction (ngram-normalized margin)")

    plt.xlim(0.0, 1.0)
    plt.ylim(0.90, 1.01) 

    plt.grid(True, alpha=0.3)
    plt.legend()

    out = Path("results/abstain_curves.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    print("Saved:", out)


if __name__ == "__main__":
    main()

