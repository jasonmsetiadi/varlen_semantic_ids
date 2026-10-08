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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/RQ2/amazon")
    parser.add_argument(
        "--output",
        default="results/RQ2/amazon/seqrec_comparison.csv",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    experiments = {
        "dvae": results_dir / "seqrec_dvae.json",
        "varlen_dvae": results_dir / "seqrec_varlen_dvae.json",
        "rkmeans": results_dir / "seqrec_rkmeans.json",
    }

    rows = [load_row(method, path) for method, path in experiments.items()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["method", *METRICS]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
