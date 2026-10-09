#!/usr/bin/env python3
"""Create a CSV comparison table from semantic-ID evaluation JSON files."""

import argparse
import csv
import json
from pathlib import Path


def get_metrics(method: str, path: Path) -> list[dict]:
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
            "split": split,
            "recon_at_last": recon.get("@last", recon.get("@5")),
            "recon_at_5": recon.get("@5"),
            "recon_varlen": recon.get("@varlen"),
            "perplexity": codebook.get("perplexity_macro"),
            "codebook_usage": codebook.get("codebook_usage_macro"),
            "mean_length": length.get("mean_uniform", length.get("fixed")),
            "weighted_length": length.get("mean_unigram"),
            "unique_codes_per_item": distinctness.get("unique_codes_per_item"),
        })

    return rows


def resolve_result(results_dir: Path, method: str) -> Path:
    """Resolve the canonical nested tokenizer metrics file."""
    method_dirs = {
        "dvae": "dvae",
        "varlen_dvae": "dvae_varlen_3",
        "rkmeans": "rkmeans",
    }
    path = results_dir / method_dirs[method] / "metrics.json"
    if path.exists():
        return path

    raise FileNotFoundError(
        f"Missing tokenizer metrics for {method}: {path}. "
        "Expected results/<method>/metrics.json."
    )


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
        rows.extend(get_metrics(method, resolve_result(results_dir, method)))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["method", "split"]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
