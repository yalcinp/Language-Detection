from pathlib import Path
import json
import matplotlib.pyplot as plt

def load_one(p: Path):
    d = json.loads(p.read_text(encoding="utf-8"))
    curve = d["curve"]
    xs, ys = [], []
    for r in curve:
        if r.get("accuracy") is None:
            continue
        xs.append(r["coverage"])
        ys.append(r["accuracy"])
    name = d.get("name", p.stem)
    return xs, ys, name

def main():
    paths = sorted(Path("results").glob("*.json"))
    paths = [p for p in paths if "abstain" in p.stem]  # esnek filtre
    if not paths:
        raise SystemExit("No abstain JSON found in results/")

    for p in paths:
        xs, ys, name = load_one(p)
        plt.plot(xs, ys, label=name)

    plt.xlabel("Coverage")
    plt.ylabel("Accuracy")
    plt.title("Coverage–Accuracy (Abstention)")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.01)
    plt.grid(True)
    plt.legend()

    out = Path("results/fig_abstain_coverage_accuracy_nb.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    print("Saved:", out)

if __name__ == "__main__":
    main()

