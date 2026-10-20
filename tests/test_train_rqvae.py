import json
import tempfile
import unittest
import warnings
from pathlib import Path

import polars as pl

from lib.utils import get_fractional_warmup_steps
from scripts.train_rqvae import main


def _items(item_ids: list[int]) -> pl.DataFrame:
    embeddings = []
    for item_id in item_ids:
        embedding = [0.0] * 4
        embedding[item_id % 4] = 1.0
        embeddings.append(embedding)
    return pl.DataFrame(
        {
            "item_id": item_ids,
            "embed": pl.Series(
                "embed", embeddings, dtype=pl.Array(pl.Float32, 4)
            ),
            "train_count": [1] * len(item_ids),
        }
    )


class TrainRQVAETest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tempdir.name)
        _items([0, 1, 2, 3]).write_parquet(
            self.data_dir / "dvae_train_items.parquet"
        )
        _items([4, 5]).write_parquet(
            self.data_dir / "dvae_holdout_items.parquet"
        )
        _items([6, 7]).write_parquet(
            self.data_dir / "dvae_cold_items.parquet"
        )
        pl.DataFrame({"item_id": [0, 1, 2, 3, 0, 1]}).write_parquet(
            self.data_dir / "dvae_train_interactions.parquet"
        )
        pl.DataFrame({"item_id": [4, 5]}).write_parquet(
            self.data_dir / "dvae_holdout_interactions.parquet"
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def config(self) -> dict:
        return {
            "data_dir": str(self.data_dir),
            "train_interactions_file": "dvae_train_interactions.parquet",
            "holdout_interactions_file": "dvae_holdout_interactions.parquet",
            "device": "cpu",
            "seed": 42,
            "num_epochs": 1,
            "vocab_size": 4,
            "embed_dim": 4,
            "hidden_dims": [4],
            "latent_dim": 3,
            "maxlen": 2,
            "commitment_beta": 0.25,
            "quantization_loss_weight": 1.0,
            "reconstruction_loss_weight": 1.0,
            "distance_batch_size": 2,
            "init_encode_batch_size": 2,
            "compile": False,
            "batch_num_tokens": 2,
            "shuffle": True,
            "drop_last": False,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "train_sample_frac_for_eval": 1.0,
            "log_every_steps": 100,
            "eval_every_steps": 0,
            "eval_detail": False,
            "enable_progress_bar": False,
            "progress_bar_every": 100,
            "eval_batch_size": 2,
            "per_step": True,
            "expected_holdout_frac": 0.5,
            "early_stopping_patience": 0,
        }

    def run_without_cuda_warning(self, config: dict) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="User provided device_type of 'cuda'.*"
            )
            warnings.filterwarnings(
                "ignore", message="Please use the new API settings to control TF32.*"
            )
            main(config)

    def test_train_mode_writes_metrics(self):
        metrics_path = self.data_dir / "train_metrics.json"
        config = self.config()
        config.update(mode="train", metrics_json_path=str(metrics_path))
        self.run_without_cuda_warning(config)

        self.assertTrue(metrics_path.exists())
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(metrics["meta"]["maxlen"], 2)
        self.assertEqual(metrics["meta"]["vocab_size"], 4)
        self.assertIn("@1", metrics["train"]["recon"]["uniform"])
        self.assertIn("@2", metrics["train"]["recon"]["uniform"])

    def test_length_beta_warmup_uses_training_fraction(self):
        self.assertEqual(get_fractional_warmup_steps(48_210, 0.2), 9_642)
        self.assertEqual(get_fractional_warmup_steps(5, 0.2), 1)
        with self.assertRaises(ValueError):
            get_fractional_warmup_steps(100, 0.0)

    def test_train_holdout_writes_metrics_and_sids(self):
        output_dir = self.data_dir / "out"
        config = self.config()
        config.update(mode="train_holdout", out_dir=str(output_dir))
        self.run_without_cuda_warning(config)

        metrics_path = output_dir / "metrics.json"
        sids_path = output_dir / "sids.parquet"
        self.assertTrue(metrics_path.exists())
        self.assertTrue(sids_path.exists())
        sids = pl.read_parquet(sids_path)
        self.assertEqual(sids.columns, ["item_id", "sid", "length"])
        self.assertEqual(sids.height, 6)
        self.assertEqual(sids["sid"].dtype, pl.Array(pl.Int64, 2))
        self.assertEqual(sids["length"].unique().to_list(), [2])

    def test_varlen_train_holdout_writes_learned_lengths(self):
        output_dir = self.data_dir / "varlen_out"
        config = self.config()
        config.update(
            mode="train_holdout",
            out_dir=str(output_dir),
            varlen=True,
            stop_num_heads=1,
            stop_dropout=0.0,
            length_cost=5.0,
            length_beta_start=0.0,
            length_beta_end=0.002 / 4,
            length_inference="argmax",
        )
        self.run_without_cuda_warning(config)

        metrics = json.loads(
            (output_dir / "metrics.json").read_text(encoding="utf-8")
        )
        split = metrics["train_holdout"]
        self.assertIn("@varlen", split["recon"]["uniform"])
        self.assertIn("length", split)
        sids = pl.read_parquet(output_dir / "sids.parquet")
        self.assertEqual(sids.columns, ["item_id", "sid", "length"])
        self.assertTrue(bool((sids["length"] >= 1).all()))
        self.assertTrue(bool((sids["length"] <= 2).all()))


if __name__ == "__main__":
    unittest.main()
