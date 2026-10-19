#!/usr/bin/env python3
"""Create a CSV comparison table from semantic-ID evaluation JSON files."""

import argparse
import csv
import json
from pathlib import Path


METHOD_DIRS = {
    "dvae": "dvae",
    "varlen_dvae": "dvae_varlen_3",
    "rkmeans": "rkmeans",
}


def get_metrics(method: str, max_length: int, path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing result file for {method}: {path}")

    with path.open(encoding="utf-8") as f:
        result = json.load(f)

    rows = []
    # RQ2 train_holdout evaluations use one combined training split.  Keep
    # the legacy train/holdout names for older flat result files.
    split_names = ("train_holdout", "train", "holdout", "cold")
    for split in split_names:
        values = result.get(split)
        if values is None:
            continue

        recon = values.get("recon", {}).get("unigram", {})
        codebook = values.get("codebook", {})
        length = values.get("length", {})
        distinctness = values.get("distinctness", {})

        rows.append({
            "method": method,
            "max_length": max_length,
            "split": split,
            "recon_at_last": recon.get("@last", recon.get(f"@{max_length}")),
            "recon_at_5": recon.get("@5"),
            "recon_varlen": recon.get("@varlen"),
            "perplexity": codebook.get("perplexity_macro"),
            "codebook_usage": codebook.get("codebook_usage_macro"),
            "mean_length": length.get("mean_uniform", length.get("fixed")),
            "weighted_length": length.get("mean_unigram"),
            "unique_codes_per_item": distinctness.get("unique_codes_per_item"),
        })

    return rows


def discover_results(results_dir: Path, method: str) -> list[tuple[int, Path]]:
    method_dir = results_dir / METHOD_DIRS[method]
    results = []
    for length_dir in method_dir.glob("maxlen_*"):
        try:
            max_length = int(length_dir.name.removeprefix("maxlen_"))
        except ValueError:
            continue
        path = length_dir / "metrics.json"
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
    output_path = args.output or str(results_dir / "semantic_id_comparison.csv")
    experiments = ("dvae", "varlen_dvae", "rkmeans")

    rows = []
    for method in experiments:
        for max_length, path in discover_results(results_dir, method):
            rows.extend(get_metrics(method, max_length, path))
    rows.sort(
        key=lambda row: (
            row["max_length"],
            experiments.index(row["method"]),
            row["split"],
        )
    )
    if not rows:
        raise FileNotFoundError(f"No maxlen_*/metrics.json results found under {results_dir}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["method", "max_length", "split"]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
