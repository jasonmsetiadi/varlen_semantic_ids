#!/usr/bin/env python3
"""Create a CSV comparison table from sequential-recommendation results."""

import argparse
import csv
import json
from pathlib import Path


METRICS = [
    "recall@1", "recall@10", "recall@100",
    "ndcg@1", "ndcg@10", "ndcg@100",
    "coverage@1", "coverage@10", "coverage@100",
    "head_recall@1", "head_recall@10", "head_recall@100",
    "head_ndcg@1", "head_ndcg@10", "head_ndcg@100",
    "tail_recall@1", "tail_recall@10", "tail_recall@100",
    "tail_ndcg@1", "tail_ndcg@10", "tail_ndcg@100",
    "long_tail_share", "mean_candidate_sid_length",
]

METHOD_DIRS = {
    "dvae": "dvae",
    "varlen_dvae": "dvae_varlen_3",
    "rkmeans": "rkmeans",
    "rqvae": "rqvae",
}


def load_row(method: str, max_length: int, path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing result file for {method}: {path}")

    with path.open(encoding="utf-8") as f:
        result = json.load(f)

    metrics = result.get("metrics", result)
    return {
        "method": method,
        "max_length": max_length,
        **{key: metrics.get(key) for key in METRICS},
    }


def discover_results(results_dir: Path, method: str) -> list[tuple[int, Path]]:
    method_dir = results_dir / METHOD_DIRS[method]
    results = []
    for length_dir in method_dir.glob("maxlen_*"):
        try:
            max_length = int(length_dir.name.removeprefix("maxlen_"))
        except ValueError:
            continue
        path = length_dir / "seqrec.json"
        if path.exists():
            results.append((max_length, path))
    return sorted(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="amazon")
    parser.add_argument("--results-dir")
    parser.add_argument("--output")
    args = parser.parse_args()

    results_dir = Path(args.results_dir or f"results/RQ2/{args.dataset}")
    output_path = args.output or str(results_dir / "seqrec_comparison.csv")
    experiments = ("dvae", "varlen_dvae", "rkmeans", "rqvae")

    rows = []
    for method in experiments:
        for max_length, path in discover_results(results_dir, method):
            rows.append(load_row(method, max_length, path))
    rows.sort(key=lambda row: (row["max_length"], experiments.index(row["method"])))
    if not rows:
        raise FileNotFoundError(f"No maxlen_*/seqrec.json results found under {results_dir}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["method", "max_length", *METRICS]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
