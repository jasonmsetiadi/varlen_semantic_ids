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


def load_row(method: str, path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing result file for {method}: {path}")

    with path.open(encoding="utf-8") as f:
        result = json.load(f)

    metrics = result.get("metrics", result)
    return {"method": method, **{key: metrics.get(key) for key in METRICS}}


def resolve_result(results_dir: Path, method: str) -> Path:
    candidates = {
        "dvae": ("seqrec_dvae.json", "dvae.json"),
        "varlen_dvae": ("seqrec_varlen_dvae.json", "seqrec_dvae_varlen.json"),
        "rkmeans": ("seqrec_rkmeans.json", "rkmeans.json"),
    }
    for filename in candidates[method]:
        path = results_dir / filename
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find sequential-recommendation results for {method} under {results_dir}."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="amazon")
    parser.add_argument("--results-dir")
    parser.add_argument("--output")
    args = parser.parse_args()

    results_dir = Path(args.results_dir or f"results/RQ2/{args.dataset}")
    output_path = args.output or str(results_dir / "seqrec_comparison.csv")
    experiments = ("dvae", "varlen_dvae", "rkmeans")

    rows = [load_row(method, resolve_result(results_dir, method)) for method in experiments]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["method", *METRICS]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
