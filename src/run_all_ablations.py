from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation suites without cross-product.")
    parser.add_argument("--preproc", action="store_true", help="Run preprocessing ablation (lower/NFKC/punct).")
    parser.add_argument("--space-short", action="store_true", help="Run whitespace + short-text ablation (L=20/30, keep vs drop spaces).")
    parser.add_argument("--all", action="store_true", help="Run all ablations.")
    args = parser.parse_args()

    # default: run all if no flag is given
    if not (args.preproc or args.space_short or args.all):
        args.all = True

    root = Path(__file__).resolve().parent
    py = sys.executable

    # Scripts (relative to src/)
    preproc_script = root / "eval_preproc_ablation.py"
    space_script = root / "eval_space_ablation.py"

    if args.all or args.preproc:
        if not preproc_script.exists():
            raise FileNotFoundError(f"Missing: {preproc_script}")
        run([py, str(preproc_script)])

    if args.all or args.space_short:
        if not space_script.exists():
            raise FileNotFoundError(f"Missing: {space_script}")
        run([py, str(space_script)])

    print("\nDone. Results are under:")
    print("- results/preproc_ablation/")
    print("- results/space_short_ablation/")


if __name__ == "__main__":
    main()

