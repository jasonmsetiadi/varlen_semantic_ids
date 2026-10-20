#!/bin/bash
set -euo pipefail

# load python and cuda
module load python/3.10.8
module load cuda/12.1.0
module load gcc/12.2.0

# Create python environment
SCRATCH_DIR=/srv/scratch/z5611110
cd "$SCRATCH_DIR"
#python -m venv .venv
source .venv/bin/activate
cd varlen_semantic_ids

# Install dependencies
#pip install --upgrade pip setuptools wheel
# pip install -r requirements.txt

usage() {
  echo "Usage: $0 DATASET TOKENIZER MAX_LENGTH" >&2
}

DATASET="${1:-}"
TOKENIZER="${2:-}"
MAX_LENGTH="${3:-}"

if [[ -z "$DATASET" || -z "$TOKENIZER" || -z "$MAX_LENGTH" ]]; then
  echo "Dataset, tokenizer, and max length arguments are required" >&2
  usage
  exit 2
fi

case "$TOKENIZER" in
  dvae)
    TOKENIZER_MODULE="train_dvae"
    TOKENIZER_CONFIG="dvae.yaml"
    TOKENIZER_RESULTS="dvae"
    TOKENIZER_LABEL="fixed dVAE"
    ;;
  varlen_dvae)
    TOKENIZER_MODULE="train_dvae"
    TOKENIZER_CONFIG="dvae_varlen_3.yaml"
    TOKENIZER_RESULTS="dvae_varlen_3"
    TOKENIZER_LABEL="variable-length dVAE"
    ;;
  rkmeans)
    TOKENIZER_MODULE="train_rkmeans"
    TOKENIZER_CONFIG="rkmeans.yaml"
    TOKENIZER_RESULTS="rkmeans"
    TOKENIZER_LABEL="RKMeans"
    ;;
  rqvae)
    TOKENIZER_MODULE="train_rqvae"
    TOKENIZER_CONFIG="rqvae.yaml"
    TOKENIZER_RESULTS="rqvae"
    TOKENIZER_LABEL="RQ-VAE"
    ;;
  *)
    echo "Unknown tokenizer: $TOKENIZER (choose dvae, varlen_dvae, rkmeans, or rqvae)" >&2
    exit 2
    ;;
esac

RQ2_DIR="configs/RQ2/$DATASET"
RESULTS_DIR="results/RQ2/$DATASET"

TOKENIZER_ARGS=(--config "$RQ2_DIR/$TOKENIZER_CONFIG")
TOKENIZER_ARGS+=(--max-length "$MAX_LENGTH")
TOKENIZER_OUTPUT_DIR="$RESULTS_DIR/$TOKENIZER_RESULTS/maxlen_$MAX_LENGTH"

nvidia-smi

echo "===== $DATASET: preprocessing ====="
# python -m scripts.data.amazon beauty
# python -m scripts.data.amazon instruments
# python -m scripts.data.amazon video_games

# mkdir -p data/yambda
# python -c "from scripts.data.yambda import download; download('./data/yambda')"
# python -c "from scripts.data.yambda import main; main('./data/yambda', './data/yambda', core_threshold=16, holdout_frac=0.1, seed=42)"
# ls data/yambda/dvae_train_items.parquet
# python -m scripts.data.yambda

# mkdir -p data/vklsvd
# python -c "from scripts.data.vklsvd import download; download('./data/vklsvd')"
# python -c "from scripts.data.vklsvd import main; main('./data/vklsvd', './data/vklsvd', core_threshold=16, holdout_frac=0.1, seed=42)"
# ls data/vklsvd/dvae_train_items.parquet
# ls data/vklsvd/seqrec_test_sample_interactions.parquet

echo "===== $DATASET: $TOKENIZER_LABEL semantic IDs ====="
python -m "scripts.$TOKENIZER_MODULE" "${TOKENIZER_ARGS[@]}"

echo "===== $DATASET: $TOKENIZER_LABEL recommender ====="
python -m scripts.train_seqrec \
  --config "$RQ2_DIR/seqrec.yaml" \
  --semantic-ids-path "$TOKENIZER_OUTPUT_DIR/sids.parquet" \
  --summary-json "$TOKENIZER_OUTPUT_DIR/seqrec.json"
