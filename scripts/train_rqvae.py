#!/usr/bin/env python3
import argparse
import json
import os

import polars as pl
import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm

from lib.data import EventsDataset
from lib.evaluate import evaluate_all, evaluate_for_tb
from lib.rqvae import RQVAE
from lib.utils import configure_torch, encode_to_sids_df


def _autocast(device: str):
    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=str(device).startswith("cuda"),
    )


def main(cfg: dict) -> None:
    configure_torch()

    data_dir = cfg["data_dir"]
    device = cfg["device"]
    mode = cfg.get("mode", "train")
    id_col = cfg.get("id_col", "item_id")

    torch.manual_seed(cfg["seed"])
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(cfg["seed"])

    train_items = pl.read_parquet(os.path.join(data_dir, "dvae_train_items.parquet"))
    holdout_items = pl.read_parquet(os.path.join(data_dir, "dvae_holdout_items.parquet"))
    cold_items = pl.read_parquet(os.path.join(data_dir, "dvae_cold_items.parquet"))

    if mode == "train":
        fit_items = train_items
    elif mode == "train_holdout":
        fit_items = pl.concat([train_items, holdout_items], how="vertical", rechunk=True)
    else:
        raise ValueError(f"Unknown mode={mode!r}. Expected 'train' or 'train_holdout'.")

    fit_embeddings = fit_items["embed"].to_torch().to(torch.float32)
    if fit_embeddings.ndim != 2:
        raise ValueError(f"Expected fit embeddings [N,D], got {tuple(fit_embeddings.shape)}")
    if fit_embeddings.size(1) != int(cfg["embed_dim"]):
        raise ValueError(
            f"Configured embed_dim={cfg['embed_dim']} does not match data dimension "
            f"{fit_embeddings.size(1)}"
        )
    fit_embeddings = F.normalize(fit_embeddings, dim=-1)
    if str(device).startswith("cuda"):
        fit_embeddings = fit_embeddings.to(torch.bfloat16)
    fit_embeddings = fit_embeddings.to(device, non_blocking=True).contiguous()

    interactions = pl.read_parquet(os.path.join(data_dir, cfg["train_interactions_file"]))
    if mode == "train_holdout":
        holdout_file = cfg.get("holdout_interactions_file")
        if holdout_file is None:
            raise ValueError("train_holdout mode requires holdout_interactions_file")
        holdout_interactions = pl.read_parquet(os.path.join(data_dir, holdout_file))
        interactions = pl.concat(
            [interactions, holdout_interactions], how="vertical", rechunk=True
        )

    mapping = fit_items.select(id_col).with_row_index("token_id")
    interactions = (
        interactions.join(mapping, on=id_col, how="left")
        .filter(pl.col("token_id").is_not_null())
    )
    dataloader = EventsDataset(
        interactions,
        batch_size=cfg["batch_num_tokens"],
        device=device,
        pin_memory=str(device).startswith("cuda"),
        shuffle=cfg["shuffle"],
        drop_last=cfg["drop_last"],
        embedding_table=fit_embeddings,
    )
    if len(dataloader) == 0:
        raise ValueError(
            "The interaction dataloader has no batches; lower batch_num_tokens or disable drop_last"
        )

    model = RQVAE(
        input_dim=cfg["embed_dim"],
        hidden_dims=cfg["hidden_dims"],
        latent_dim=cfg["latent_dim"],
        vocab_size=cfg["vocab_size"],
        maxlen=cfg["maxlen"],
        commitment_beta=cfg["commitment_beta"],
        quantization_loss_weight=cfg.get("quantization_loss_weight", 1.0),
        reconstruction_loss_weight=cfg.get("reconstruction_loss_weight", 1.0),
        distance_batch_size=cfg["distance_batch_size"],
    ).to(device)

    print(
        f"[init] Initializing {model.maxlen} codebooks with {model.vocab_size} entries "
        f"from {fit_embeddings.size(0):,} unique items...",
        flush=True,
    )
    model.initialize_codebooks(
        fit_embeddings,
        seed=cfg["seed"],
        encode_batch_size=cfg.get("init_encode_batch_size", cfg["eval_batch_size"]),
    )

    encoder = model.inference_encoder()
    decoder = model.inference_decoder()

    compile_setting = cfg.get("compile", "auto")
    if isinstance(compile_setting, str) and compile_setting.lower() == "auto":
        if str(device).startswith("cuda") and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(torch.device(device))
            use_compile = capability >= (8, 0)
        else:
            use_compile = False
    else:
        use_compile = bool(compile_setting)
    graph = torch.compile(model, dynamic=False) if use_compile else model
    graph.train()

    optimizer = torch.optim.AdamW(
        graph.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    optimizer.zero_grad(set_to_none=True)

    writer = None
    if cfg.get("tensorboard_logdir") is not None:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=cfg["tensorboard_logdir"])

    sampled_train_items = train_items.sample(
        fraction=cfg["train_sample_frac_for_eval"],
        seed=cfg["seed"],
        shuffle=True,
    )

    steps_per_epoch = len(dataloader)
    total_steps = cfg["num_epochs"] * steps_per_epoch
    tokens_passed = 0
    global_step = 0
    best_train_loss = float("inf")
    best_train_epoch = 0
    best_train_step = 0
    best_epoch_train_loss = float("inf")
    best_epoch = 0
    early_stopping_patience = int(cfg.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(cfg.get("early_stopping_min_delta", 0.0))
    epochs_without_improvement = 0
    best_epoch_state = None

    pbar = tqdm(
        total=total_steps,
        desc=f"train_rqvae({mode})",
        disable=not cfg["enable_progress_bar"],
        dynamic_ncols=True,
    )

    for epoch in range(cfg["num_epochs"]):
        graph.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0
        for batch in dataloader:
            with _autocast(device):
                loss, metrics = graph(batch.embeddings)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            train_loss = float(loss.detach())
            epoch_loss_sum += train_loss
            epoch_batches += 1
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_train_epoch = epoch + 1
                best_train_step = global_step + 1

            if global_step % cfg["log_every_steps"] == 0:
                metric_values = " ".join(
                    f"{name}={float(value):.4g}" for name, value in metrics.items()
                )
                tqdm.write(
                    f"step={global_step}/{total_steps} epoch={epoch + 1}/{cfg['num_epochs']} "
                    f"tokens={tokens_passed} {metric_values}"
                )
                if writer is not None:
                    for name, value in metrics.items():
                        writer.add_scalar(f"train/{name}", float(value), tokens_passed)

            tokens_passed += int(getattr(batch, "size", 0))
            global_step += 1
            pbar.update(1)

            if cfg["eval_every_steps"] > 0 and global_step % cfg["eval_every_steps"] == 0:
                graph.eval()
                eval_metrics = evaluate_for_tb(
                    encoder=encoder,
                    decoder=decoder,
                    train_items=sampled_train_items,
                    holdout_items=holdout_items,
                    cold_items=None,
                    device=device,
                    batch_size=cfg["eval_batch_size"],
                    detail=cfg["eval_detail"],
                )
                if writer is not None:
                    for name, value in eval_metrics.items():
                        writer.add_scalar(name, float(value), tokens_passed)
                eval_values = " ".join(
                    f"{name}={float(value):.4g}" for name, value in eval_metrics.items()
                )
                tqdm.write(
                    f"eval step={global_step}/{total_steps} epoch={epoch + 1}/{cfg['num_epochs']} "
                    f"tokens={tokens_passed} {eval_values}"
                )
                graph.train()

            if global_step % cfg["progress_bar_every"] == 0:
                pbar.set_postfix(epoch=epoch, loss=train_loss)

        if epoch_batches > 0 and early_stopping_patience > 0:
            epoch_train_loss = epoch_loss_sum / epoch_batches
            if epoch_train_loss < best_epoch_train_loss - early_stopping_min_delta:
                best_epoch_train_loss = epoch_train_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                best_epoch_state = {
                    name: value.detach().cpu().clone()
                    for name, value in graph.state_dict().items()
                }
            else:
                epochs_without_improvement += 1

            tqdm.write(
                f"epoch_end epoch={epoch + 1}/{cfg['num_epochs']} "
                f"train_loss={epoch_train_loss:.6g} "
                f"best_train_loss={best_epoch_train_loss:.6g}@{best_epoch} "
                f"patience={epochs_without_improvement}/{early_stopping_patience}"
            )
            if epochs_without_improvement >= early_stopping_patience:
                tqdm.write(
                    f"early_stopping epoch={epoch + 1} best_epoch={best_epoch} "
                    f"best_train_loss={best_epoch_train_loss:.6g}"
                )
                break

    pbar.close()
    if best_epoch_state is not None:
        graph.load_state_dict(best_epoch_state)
    tqdm.write(
        f"best_train_loss={best_train_loss:.4g} epoch={best_train_epoch}/"
        f"{cfg['num_epochs']} step={best_train_step}/{total_steps}"
    )

    graph.eval()
    with torch.inference_mode():
        if mode == "train":
            metrics = evaluate_all(
                train_items=train_items,
                holdout_items=holdout_items,
                cold_items=cold_items,
                encoder=encoder,
                decoder=decoder,
                batch_size=cfg["eval_batch_size"],
                device=device,
                per_step=cfg["per_step"],
                expected_holdout_frac=cfg["expected_holdout_frac"],
            )
        else:
            metrics = evaluate_all(
                splits={"train_holdout": fit_items, "cold": cold_items},
                encoder=encoder,
                decoder=decoder,
                batch_size=cfg["eval_batch_size"],
                device=device,
                per_step=cfg["per_step"],
            )

    if mode == "train_holdout":
        out_dir = cfg["out_dir"]
        os.makedirs(out_dir, exist_ok=True)
        metrics_path = os.path.join(out_dir, "metrics.json")
        sids_path = os.path.join(out_dir, "sids.parquet")
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        sids = encode_to_sids_df(
            fit_items,
            encoder,
            id_col=id_col,
            batch_size=cfg["eval_batch_size"],
            device=device,
        )
        sids.write_parquet(sids_path)
        print(f"Saved metrics: {metrics_path}")
        print(f"Saved semantic IDs: {sids_path}")
    else:
        metrics_path = cfg["metrics_json_path"]
        os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        print(f"Saved metrics: {metrics_path}")

    if cfg.get("save_weights_path") is not None:
        weights_path = cfg["save_weights_path"]
        os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
        torch.save(graph.state_dict(), weights_path)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.max_length < 1:
        parser.error("--max-length must be at least 1")
    config["maxlen"] = args.max_length
    config["out_dir"] = os.path.join(config["out_dir"], f"maxlen_{args.max_length}")
    main(config)
