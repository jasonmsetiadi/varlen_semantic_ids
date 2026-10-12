import os
import json
import argparse

import polars as pl

try:
    from varlen_sids.scripts.data.utils import preprocess_data
except ModuleNotFoundError:
    # Also support running this repository directly from its checkout.
    from scripts.data.utils import preprocess_data


# actual data should be downloaded through https://amazon-reviews-2023.github.io/
def process(reviews_paths, meta_path, hf_token=None):
    from huggingface_hub import login
    from sentence_transformers import SentenceTransformer
    import torch

    # Use an explicitly supplied token or HF_TOKEN when available. Otherwise
    # rely on credentials stored by `huggingface-cli login`.
    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    def generator(path):
        with open(path, 'r') as f:
            for line in f:
                yield json.loads(line.strip())

    if isinstance(reviews_paths, (str, os.PathLike)):
        reviews_paths = [reviews_paths]

    interaction_frames = []
    for reviews_path in reviews_paths:
        reviews_path = os.fspath(reviews_path)
        if reviews_path.endswith('.csv'):
            interaction_frames.append(pl.read_csv(reviews_path))
        else:
            interaction_frames.append(pl.DataFrame(generator(reviews_path)))

    interactions = pl.concat(interaction_frames)

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


def _load_split_interactions(path):
    return pl.read_csv(path) \
        .filter(pl.col('rating') >= 4.) \
        .select([
            pl.col('user_id'),
            pl.col('parent_asin').alias('item_id'),
            pl.col('timestamp') // 1000
        ])


def main(train_path, test_path, embeddings_path, dst_dir,
         core_threshold=16, holdout_frac=0.1, seed=42):
    item_embeddings = pl.read_parquet(embeddings_path)
    train = _load_split_interactions(train_path)
    test = _load_split_interactions(test_path)

    preprocess_data(train, test, item_embeddings, dst_dir, core_threshold, holdout_frac, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare an Amazon category from CSV reviews.")
    parser.add_argument("category", choices=["beauty", "instruments", "video_games"])
    parser.add_argument("--input-dir", default="./data/amazon/raw",
                        help="Directory containing Amazon metadata JSONL files.")
    parser.add_argument("--split-dir", default="./data/amazon/split",
                        help="Directory containing pre-split <category>.train.csv and .test.csv files.")
    parser.add_argument("--output-dir")
    parser.add_argument("--core-threshold", type=int, default=16)
    parser.add_argument("--holdout-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    amazon_name = {
        "beauty": "All_Beauty",
        "instruments": "Musical_Instruments",
        "video_games": "Video_Games",
    }[args.category]
    output_dir = args.output_dir or f"./data/{args.category}"
    train_path = os.path.join(args.split_dir, f"{amazon_name}.train.csv")
    test_path = os.path.join(args.split_dir, f"{amazon_name}.test.csv")
    meta_path = os.path.join(args.input_dir, f"meta_{amazon_name}.jsonl")

    os.makedirs(output_dir, exist_ok=True)
    interactions, embeddings = process([train_path, test_path], meta_path)
    interactions.write_parquet(os.path.join(output_dir, "interactions.parquet"))
    embeddings_path = os.path.join(output_dir, "embeddings.parquet")
    embeddings.write_parquet(embeddings_path)
    main(
        train_path,
        test_path,
        embeddings_path,
        output_dir,
        args.core_threshold,
        args.holdout_frac,
        args.seed,
    )
