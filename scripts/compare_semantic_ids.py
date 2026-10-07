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
    for split in ("train", "holdout", "cold"):
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
            "recon_at_last": recon.get("@last"),
            "recon_at_5": recon.get("@5"),
            "recon_varlen": recon.get("@varlen"),
            "perplexity": codebook.get("perplexity_macro"),
            "codebook_usage": codebook.get("codebook_usage_macro"),
            "mean_length": length.get("mean_uniform", length.get("fixed")),
            "weighted_length": length.get("mean_unigram"),
            "unique_codes_per_item": distinctness.get("unique_codes_per_item"),
        })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/RQ1/amazon")
    parser.add_argument(
        "--output",
        default="results/RQ1/amazon/semantic_id_comparison.csv",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    experiments = {
        "dvae": results_dir / "dvae.json",
        "varlen_dvae": results_dir / "dvae_varlen_lengthcost5.json",
        "rkmeans": results_dir / "rkmeans.json",
    }

    rows = []
    for method, path in experiments.items():
        rows.extend(get_metrics(method, path))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["method", "split"]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
