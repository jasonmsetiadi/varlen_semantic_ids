import os
import tqdm
import json
import argparse

import numpy as np
import polars as pl

try:
    from varlen_sids.scripts.data.utils import preprocess_data
except ModuleNotFoundError:
    # Also support running this repository directly from its checkout.
    from scripts.data.utils import preprocess_data


TEST_INTERVAL = 7 * 24 * 60 * 60 * 4 # 4 weeks


# actual data should be downloaded through https://amazon-reviews-2023.github.io/
def process(reviews_path, meta_path, hf_token=None):
    from huggingface_hub import login
    from sentence_transformers import SentenceTransformer
    import torch

    # Use an explicitly supplied token or HF_TOKEN when available. Otherwise
    # rely on credentials stored by `huggingface-cli login`.
    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    if reviews_path.endswith('.csv'):
        interactions = pl.read_csv(reviews_path)
    else:
        def generator(path):
            with open(path, 'r') as f:
                for line in f:
                    yield json.loads(line.strip())

        interactions = pl.DataFrame(generator(reviews_path))

    interactions = interactions \
        .filter(pl.col('rating') >= 4.)
    items = interactions.select('parent_asin').unique()

    meta = pl.DataFrame(
        generator(meta_path),
        infer_schema_length=None,
    )
    meta = meta.join(items, on='parent_asin', how='semi')
    texts = meta.map_rows(lambda x: f'Title: {x[1]} | Store: {x[9]} | Main category: {x[0]}')

    model = SentenceTransformer("google/embeddinggemma-300m")
    with torch.autocast('cuda', torch.bfloat16):
        item_embeddings = model.encode_document(
            texts['map'].to_list(), 
            batch_size=128, 
            show_progress_bar=True, 
            truncate_dim=128, 
            normalize_embeddings=True,
            device='cuda'
        )
    item_embeddings = pl.DataFrame({'parent_asin': meta['parent_asin'], 'embed': item_embeddings}) \
        .rename({'parent_asin': 'item_id'})

    return interactions, item_embeddings


def main(data_dir, dst_dir, core_threshold=16, holdout_frac=0.1, seed=42):
    item_embeddings = pl.read_parquet(os.path.join(data_dir, 'embeddings.parquet'))

    interactions = pl.read_parquet(os.path.join(data_dir, 'interactions.parquet')) \
        .filter(pl.col('rating') >= 4.) \
        .select([
            pl.col('user_id'),
            pl.col('parent_asin').alias('item_id'),
            pl.col('timestamp') // 1000
        ])

    max_timestamp = interactions['timestamp'].max()
    train = interactions.filter(pl.col('timestamp') < max_timestamp - TEST_INTERVAL)
    test = interactions.filter(pl.col('timestamp') >= max_timestamp - TEST_INTERVAL)

    preprocess_data(train, test, item_embeddings, dst_dir, core_threshold, holdout_frac, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare an Amazon category from CSV reviews.")
    parser.add_argument("category", choices=["beauty", "instruments"])
    parser.add_argument("--input-dir", default="./data/amazon/raw")
    parser.add_argument("--output-dir")
    parser.add_argument("--core-threshold", type=int, default=16)
    parser.add_argument("--holdout-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    amazon_name = {
        "beauty": "All_Beauty",
        "instruments": "Musical_Instruments",
    }[args.category]
    output_dir = args.output_dir or f"./data/{args.category}"
    reviews_path = os.path.join(args.input_dir, f"{amazon_name}.csv")
    meta_path = os.path.join(args.input_dir, f"meta_{amazon_name}.jsonl")

    os.makedirs(output_dir, exist_ok=True)
    interactions, embeddings = process(reviews_path, meta_path)
    interactions.write_parquet(os.path.join(output_dir, "interactions.parquet"))
    embeddings.write_parquet(os.path.join(output_dir, "embeddings.parquet"))
    main(output_dir, output_dir, args.core_threshold, args.holdout_frac, args.seed)
