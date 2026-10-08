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


def resolve_result(results_dir: Path, method: str) -> Path:
    """Resolve both flat RQ1 files and RQ2 method directories."""
    candidates = {
        "dvae": [
            results_dir / "dvae.json",
            results_dir / "dvae" / "metrics.json",
        ],
        "varlen_dvae": [
            results_dir / "dvae_varlen_lengthcost5.json",
            results_dir / "varlen_dvae.json",
            results_dir / "varlen_dvae" / "metrics.json",
        ],
        "rkmeans": [
            results_dir / "rkmeans.json",
            results_dir / "rkmeans" / "metrics.json",
        ],
    }
    for path in candidates[method]:
        if path.exists():
            return path

    # Support numbered experiment variants such as dvae_varlen_3.
    patterns = {
        "varlen_dvae": ("dvae_varlen*", "varlen_dvae*"),
        "dvae": ("dvae*",),
        "rkmeans": ("rkmeans*",),
    }
    matches = []
    for pattern in patterns[method]:
        matches.extend(results_dir.glob(pattern + ".json"))
        matches.extend(results_dir.glob(pattern + "/metrics.json"))
    if matches:
        return sorted(matches)[-1]

    raise FileNotFoundError(
        f"Could not find {method} results under {results_dir}. "
        "Expected a flat JSON file or <method>/metrics.json."
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
