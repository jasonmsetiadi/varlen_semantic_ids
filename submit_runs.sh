#!/bin/bash
set -euo pipefail

datasets=(
  "yambda"
  "instruments"
  "video_games"
)

tokenizers=(
  "varlen_dvae"
  "rkmeans"
  "dvae"
  "rqvae"
)

max_lengths=(
  5
  10
  20
)

PBS_SCRIPT="${PBS_SCRIPT:-run_rq2.pbs}"

for dataset in "${datasets[@]}"; do
  for tokenizer in "${tokenizers[@]}"; do
    for max_length in "${max_lengths[@]}"; do
      job_name="rq2_${dataset}_${tokenizer}_len${max_length}"
      echo "Submitting $dataset / $tokenizer / max length $max_length as $job_name"
      qsub -N "$job_name" \
        -v "DATASET=$dataset,TOKENIZER=$tokenizer,MAX_LENGTH=$max_length" \
        "$PBS_SCRIPT"
    done
  done
done
