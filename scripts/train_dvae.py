#!/usr/bin/env python3
import argparse
import json
import os

import yaml
import polars as pl
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from lib.data import EventsDataset
from lib.utils import (
    configure_torch,
    encode_to_sids_df,
    get_cosine_scheduler,
    get_fractional_warmup_steps,
)
from lib.layers import Encoder, Decoder
from lib.game import Game, VarlenGame
from lib.evaluate import evaluate_all, evaluate_for_tb
import lib.evaluate as ev


def main(cfg):
    configure_torch()

    data_dir = cfg["data_dir"]
    device = cfg["device"]
    mode = cfg.get("mode", "train")  # "train" | "train_holdout"
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

    fit_embeddings = fit_items["embed"].to_torch().to(torch.float32).to(device, non_blocking=True)
    fit_embeddings = F.normalize(fit_embeddings, dim=-1).to(torch.bfloat16).contiguous()

    interactions = pl.read_parquet(os.path.join(data_dir, cfg["train_interactions_file"]))
    if mode == "train_holdout":
        assert cfg.get("holdout_interactions_file") is not None
        interactions_hold = pl.read_parquet(os.path.join(data_dir, cfg["holdout_interactions_file"]))
        interactions = pl.concat([interactions, interactions_hold], how="vertical", rechunk=True)

    mapping = fit_items.select(id_col).with_row_index("token_id")
    interactions = (
        interactions
        .join(mapping, on=id_col, how="left")
        .filter(pl.col("token_id").is_not_null())
    )

    dataloader = EventsDataset(
        interactions,
        batch_size=cfg["batch_num_tokens"],
        shuffle=cfg["shuffle"],
        drop_last=cfg["drop_last"],
        embedding_table=fit_embeddings,
    )

    encoder = Encoder(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_size=cfg["hidden_size"],
        maxlen=cfg["maxlen"],
        codebook_dropout=cfg["codebook_dropout"],
        dropout=cfg["dropout"],
        varlen=cfg["varlen"],
        shared_codebooks=cfg["shared_codebooks"],
        init_logit_scale=cfg["init_logit_scale"],
        init_gamma=cfg["init_gamma"]
    )
    decoder = Decoder(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_size=cfg["hidden_size"] * cfg["decoder_hidden_mul"],
        maxlen=cfg["maxlen"],
        dropout=cfg["dropout"],
        num_layers=cfg["decoder_num_layers"],
    )
    graph = (VarlenGame(encoder, decoder) if cfg["varlen"] else Game(encoder, decoder)).to(device)
    compile_setting = cfg.get("compile", "auto")
    if isinstance(compile_setting, str) and compile_setting.lower() == "auto":
        if str(device).startswith("cuda") and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(torch.device(device))
            use_compile = capability >= (8, 0)
        else:
            use_compile = False
    else:
        use_compile = bool(compile_setting)
    if use_compile:
        graph = torch.compile(graph, dynamic=False)
    graph.train()

    enc_inf = encoder.inference(mode="argmax")
    dec_inf = decoder.inference()
    if use_compile:
        enc_inf = torch.compile(enc_inf, dynamic=False)
        dec_inf = torch.compile(dec_inf, dynamic=False)

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

    length_cost_t = None
    if cfg["varlen"]:
        length_cost_t = torch.tensor(cfg["length_cost"], device=device)

    sampled_train_items = train_items.sample(
        fraction=cfg["train_sample_frac_for_eval"],
        seed=cfg["seed"],
        shuffle=True,
    )

    steps_per_epoch = len(dataloader)
    total_steps = cfg["num_epochs"] * steps_per_epoch
    beta_schedule_steps = steps_per_epoch
    if cfg.get("beta_warmup_fraction") is not None:
        beta_schedule_steps = get_fractional_warmup_steps(
            total_steps, float(cfg["beta_warmup_fraction"])
        )
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
    best_validation_loss = float("inf")
    best_validation_step = 0
    best_validation_epoch = 0

    pbar = tqdm(
        total=total_steps,
        desc=f"train({mode})",
        disable=not cfg["enable_progress_bar"],
        dynamic_ncols=True,
    )

    for epoch in range(cfg["num_epochs"]):
        graph.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0
        for batch in dataloader:
            tau = get_cosine_scheduler(
                global_step,
                start=cfg["tau_start"],
                end=cfg["tau_end"],
                total_steps=steps_per_epoch,
            )

            beta_t = None
            if cfg.get("beta_end") is not None:
                b = get_cosine_scheduler(
                    global_step,
                    start=cfg["beta_start"],
                    end=cfg["beta_end"],
                    total_steps=beta_schedule_steps,
                )
                beta_t = torch.tensor(b, device=device)

            free_bits_t = None
            if cfg.get("free_bits") is not None:
                free_bits_t = torch.tensor(cfg["free_bits"], device=device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                kwargs = dict(
                    x=batch.embeddings,
                    tau=torch.tensor(tau, device=device),
                    beta=beta_t,
                    free_bits=free_bits_t,
                )
                if cfg["varlen"]:
                    kwargs["length_cost"] = length_cost_t

                loss, metrics = graph(**kwargs)
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

            if writer is not None and (global_step % cfg["log_every_steps"] == 0):
                for name, value in metrics.items():
                    writer.add_scalar(f"train/{name}", float(value), tokens_passed)
                writer.add_scalar("params/tau", float(tau), tokens_passed)
                if beta_t is not None:
                    writer.add_scalar("params/beta", float(beta_t.item()), tokens_passed)

            if global_step % cfg["log_every_steps"] == 0:
                metric_order = [
                    "loss",
                    "recon_loss",
                    "vocab_loss",
                    "KL_term",
                    "length_loss",
                    "KL_length_term",
                    "E_L",
                    "length_entropy",
                ]
                if cfg["varlen"]:
                    metric_order.extend(
                        f"recon_loss@{length}"
                        for length in range(1, cfg["maxlen"] + 1)
                    )
                metric_values = " ".join(
                    f"{name}={float(metrics[name]):.4g}"
                    for name in metric_order if name in metrics
                )
                if cfg["varlen"]:
                    length_distribution = ", ".join(
                        f"L{length}:p={float(metrics[f'length_probability@{length}']) * 100:.2f}%"
                        f"/argmax={float(metrics[f'length_fraction@{length}']) * 100:.2f}%"
                        for length in range(1, cfg["maxlen"] + 1)
                        if f"length_fraction@{length}" in metrics
                    )
                    metric_values += f" {length_distribution}"
                beta_value = f" beta={beta_t.item():.4g}" if beta_t is not None else ""
                tqdm.write(
                    f"step={global_step}/{total_steps} epoch={epoch + 1}/{cfg['num_epochs']} "
                    f"tokens={tokens_passed} tau={tau:.4g}{beta_value} {metric_values}"
                )

            tokens_passed += int(getattr(batch, "size", 0))
            global_step += 1
            pbar.update(1)

            if cfg["eval_every_steps"] > 0 and (global_step % cfg["eval_every_steps"] == 0):
                graph.eval()
                m = evaluate_for_tb(
                    encoder=enc_inf,
                    decoder=dec_inf,
                    train_items=sampled_train_items,
                    holdout_items=holdout_items,
                    cold_items=None,
                    device=device,
                    batch_size=cfg["eval_batch_size"],
                    detail=cfg["eval_detail"],
                )
                validation_loss = m.get("recon/holdout_@last")
                if validation_loss is not None and validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_validation_step = global_step
                    best_validation_epoch = epoch + 1
                if writer is not None:
                    for k, v in m.items():
                        writer.add_scalar(k, float(v), tokens_passed)
                eval_values = " ".join(f"{name}={float(value):.4g}" for name, value in m.items())
                tqdm.write(
                    f"eval step={global_step}/{total_steps} epoch={epoch + 1}/{cfg['num_epochs']} "
                    f"tokens={tokens_passed} {eval_values}"
                )
                graph.train()

            if global_step % cfg["progress_bar_every"] == 0:
                pbar.set_postfix(epoch=epoch, loss=float(loss))

        if epoch_batches > 0 and early_stopping_patience > 0:
            epoch_train_loss = epoch_loss_sum / epoch_batches
            if epoch_train_loss < best_epoch_train_loss - early_stopping_min_delta:
                best_epoch_train_loss = epoch_train_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                # Keep the best epoch's weights on CPU so early stopping can
                # restore them without holding a second model copy on the GPU.
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
                    f"early_stopping epoch={epoch + 1} "
                    f"best_epoch={best_epoch} "
                    f"best_train_loss={best_epoch_train_loss:.6g}"
                )
                break

    pbar.close()
    if best_epoch_state is not None:
        graph.load_state_dict(best_epoch_state)
    tqdm.write(
        f"best_train_loss={best_train_loss:.4g} "
        f"epoch={best_train_epoch}/{cfg['num_epochs']} step={best_train_step}/{total_steps}"
    )
    if best_epoch > 0:
        tqdm.write(
            f"best_epoch_train_loss={best_epoch_train_loss:.6g} "
            f"epoch={best_epoch}/{cfg['num_epochs']}"
        )
    if best_validation_loss < float("inf"):
        tqdm.write(
        f"best_validation_recon={best_validation_loss:.4g} "
            f"metric=recon/holdout_@last "
            f"epoch={best_validation_epoch}/{cfg['num_epochs']} "
            f"step={best_validation_step}/{total_steps}"
        )

    graph.eval()
    with torch.inference_mode():
        if mode == "train":
            metrics = evaluate_all(
                train_items=train_items,
                holdout_items=holdout_items,
                cold_items=cold_items,
                encoder=enc_inf,
                decoder=dec_inf,
                batch_size=cfg["eval_batch_size"],
                device=device,
                per_step=cfg["per_step"],
                expected_holdout_frac=cfg["expected_holdout_frac"],
            )
        else:
            metrics = evaluate_all(
                splits={
                    "train_holdout": fit_items,
                    "cold": cold_items,
                },
                encoder=enc_inf,
                decoder=dec_inf,
                batch_size=cfg["eval_batch_size"],
                device=device,
                per_step=cfg["per_step"],
            )

    # ---- Saving: unified under out_dir in train_holdout mode ----
    if mode == "train_holdout":
        out_dir = cfg["out_dir"]
        os.makedirs(out_dir, exist_ok=True)

        metrics_path = os.path.join(out_dir, "metrics.json")
        sids_path = os.path.join(out_dir, "sids.parquet")

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        sids = encode_to_sids_df(
            fit_items,
            enc_inf,
            id_col=id_col,
            batch_size=cfg["eval_batch_size"],
            device=device,
        )
        sids.write_parquet(sids_path)
        print(f"Saved metrics: {metrics_path}")
        print(f"Saved semantic IDs: {sids_path}")
    else:
        # Backward-compatible behavior for plain train mode
        os.makedirs(os.path.dirname(cfg["metrics_json_path"]) or ".", exist_ok=True)
        with open(cfg["metrics_json_path"], "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics: {cfg['metrics_json_path']}")

    if cfg.get("save_weights_path") is not None:
        os.makedirs(os.path.dirname(cfg["save_weights_path"]) or ".", exist_ok=True)
        torch.save(graph.state_dict(), cfg["save_weights_path"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-length", type=int, required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.max_length < 1:
        ap.error("--max-length must be at least 1")
    cfg["maxlen"] = args.max_length
    cfg["out_dir"] = os.path.join(cfg["out_dir"], f"maxlen_{args.max_length}")
    main(cfg)
