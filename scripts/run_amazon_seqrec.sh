#!/usr/bin/env bash
set -euo pipefail

# Run one Amazon semantic-ID method followed by its sequential recommender.
# Usage: ./scripts/run_amazon_seqrec.sh {dvae|varlen-dvae|rkmeans}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"
method="${1:-}"

case "${method}" in
  dvae)
    semantic_config="configs/RQ2/amazon/dvae.yaml"
    seqrec_config="configs/RQ2/amazon/seqrec_dvae.yaml"
    semantic_module="scripts.train_dvae"
    label="fixed-length dVAE"
    ;;
  varlen-dvae)
    semantic_config="configs/RQ2/amazon/dvae_varlen_3.yaml"
    seqrec_config="configs/RQ2/amazon/seqrec_varlen_dvae.yaml"
    semantic_module="scripts.train_dvae"
    label="variable-length dVAE"
    ;;
  rkmeans)
    semantic_config="configs/RQ2/amazon/rkmeans.yaml"
    seqrec_config="configs/RQ2/amazon/seqrec_rkmeans.yaml"
    semantic_module="scripts.train_rkmeans"
    label="RKMeans"
    ;;
  *)
    echo "Usage: $0 {dvae|varlen-dvae|rkmeans}" >&2
    exit 2
    ;;
esac

require_config() {
  [[ -f "$1" ]] || {
    echo "Missing config: $1" >&2
    echo "Create the Amazon RQ2 configs from the Yambda/VK-LSVD templates first." >&2
    exit 1
  }
}

require_config "${semantic_config}"
require_config "${seqrec_config}"

echo "===== Amazon: ${label} semantic IDs ====="
python -m "${semantic_module}" --config "${semantic_config}"

echo "===== Amazon: sequential recommender with ${label} IDs ====="
python -m scripts.train_seqrec --config "${seqrec_config}"

echo "Amazon semantic-ID and sequential-recommendation experiments complete."
