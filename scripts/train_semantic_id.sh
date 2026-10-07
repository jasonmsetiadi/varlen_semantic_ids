#!/usr/bin/env bash
set -euo pipefail

# Train a semantic-ID model from a YAML configuration.
#
# Usage:
#   ./scripts/train_semantic_id.sh dvae
#   ./scripts/train_semantic_id.sh varlen-dvae
#   ./scripts/train_semantic_id.sh rkmeans
#   ./scripts/train_semantic_id.sh reinforce configs/RQ3/reinforce.yaml
#   ./scripts/train_semantic_id.sh varlen-reinforce
#
# If no config is supplied, the script uses the matching Amazon RQ1 config.

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
method="${1:-}"
config="${2:-}"

case "${method}" in
  dvae|fixed-dvae)
    module="scripts.train_dvae"
    default_config="${repo_dir}/configs/RQ1/amazon/dvae.yaml"
    ;;
  varlen-dvae)
    module="scripts.train_dvae"
    default_config="${repo_dir}/configs/RQ1/amazon/varlen_dvae.yaml"
    ;;
  rkmeans)
    module="scripts.train_rkmeans"
    default_config="${repo_dir}/configs/RQ1/amazon/rkmeans.yaml"
    ;;
  reinforce)
    module="scripts.train_reinforce"
    default_config="${repo_dir}/configs/RQ3/reinforce.yaml"
    ;;
  varlen-reinforce)
    module="scripts.train_reinforce"
    default_config="${repo_dir}/configs/RQ3/varlen_reinforce.yaml"
    ;;
  *)
    echo "Usage: $0 {dvae|varlen-dvae|rkmeans|reinforce|varlen-reinforce} [config.yaml]" >&2
    exit 2
    ;;
esac

config="${config:-${default_config}}"
[[ "${config}" = /* ]] || config="${repo_dir}/${config}"

[[ -f "${config}" ]] || {
  echo "Config file not found: ${config}" >&2
  exit 1
}

cd "${repo_dir}"
PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m "${module}" --config "${config}"
