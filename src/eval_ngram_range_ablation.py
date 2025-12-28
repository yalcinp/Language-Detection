from pathlib import Path
from evaluate import evaluate
from own_char_ngram_nb import CharNgramNB, load_train

LABELS = ["en","de","fr","es","it","tr","nl","sv","pl","ru","ar","zh"]

def run(tag, n_max):
    train = load_train(Path("data/train.jsonl"))
    model = CharNgramNB(n_min=1, n_max=n_max, alpha=0.1)
    model.fit(train)

    evaluate(
        predictor=lambda xs: [model.predict_one(x) for x in xs],
        test_path=Path("data/test.jsonl"),
        name=f"NB n=1..{n_max}",
        labels_order=LABELS,
        out_path=Path(f"results/ngram_ablation_{tag}.json"),
    )

if __name__ == "__main__":
    run("1_3", n_max=3)
    run("1_5", n_max=5)

