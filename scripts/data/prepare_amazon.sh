#!/usr/bin/env bash
set -euo pipefail

# Prepare Amazon Reviews 2023 for the semantic-ID and sequential-recommendation
# experiments. Download the reviews and metadata files manually first.
#
# Put the downloaded Amazon Reviews files here using their original names:
#   ./data/amazon/raw/All_Beauty.jsonl
#   ./data/amazon/raw/meta_All_Beauty.jsonl
#   ./data/amazon/raw/Musical_Instruments.jsonl
#   ./data/amazon/raw/meta_Musical_Instruments.jsonl
#
# Usage:
#   ./scripts/data/prepare_amazon.sh beauty
#   ./scripts/data/prepare_amazon.sh instruments
#
# Optional variables:
#   OUTPUT_DIR=./data/beauty CORE_THRESHOLD=16 HOLDOUT_FRAC=0.1 SEED=42

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
category="${1:-beauty}"
input_dir="${INPUT_DIR:-${repo_dir}/data/amazon/raw}"

case "${category}" in
  beauty)
    amazon_name="All_Beauty"
    ;;
  instruments|musical_instruments)
    amazon_name="Musical_Instruments"
    ;;
  *)
    echo "Usage: $0 {beauty|instruments}" >&2
    exit 2
    ;;
esac

reviews_path="${input_dir}/${amazon_name}.jsonl"
meta_path="${input_dir}/meta_${amazon_name}.jsonl"
output_dir="${OUTPUT_DIR:-${repo_dir}/data/${category}}"
core_threshold="${CORE_THRESHOLD:-16}"
holdout_frac="${HOLDOUT_FRAC:-0.1}"
seed="${SEED:-42}"

[[ -f "${reviews_path}" ]] || { echo "Reviews file not found: ${reviews_path}" >&2; exit 1; }
[[ -f "${meta_path}" ]] || { echo "Metadata file not found: ${meta_path}" >&2; exit 1; }

mkdir -p "${output_dir}"
cd "${repo_dir}"

PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}" \
  python - "${reviews_path}" "${meta_path}" "${output_dir}" "${core_threshold}" "${holdout_frac}" "${seed}" <<'PY'
import os
import sys

from scripts.data.amazon import main, process

reviews_path, meta_path, output_dir, core_threshold, holdout_frac, seed = sys.argv[1:]

print("Generating Amazon item embeddings and interaction tables...")
interactions, embeddings = process(reviews_path, meta_path)
interactions.write_parquet(os.path.join(output_dir, "interactions.parquet"))
embeddings.write_parquet(os.path.join(output_dir, "embeddings.parquet"))

main(
    data_dir=output_dir,
    dst_dir=output_dir,
    core_threshold=int(core_threshold),
    holdout_frac=float(holdout_frac),
    seed=int(seed),
)
PY

echo "Amazon ${category} preprocessing complete: ${output_dir}"
