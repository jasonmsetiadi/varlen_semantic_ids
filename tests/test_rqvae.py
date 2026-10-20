import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from lib.rqvae import RQVAE


class RQVAETest(unittest.TestCase):
    def make_model(self) -> RQVAE:
        torch.manual_seed(7)
        return RQVAE(
            input_dim=8,
            hidden_dims=[8],
            latent_dim=4,
            vocab_size=7,
            maxlen=3,
            distance_batch_size=2,
        )

    def make_varlen_model(self, maxlen: int = 3) -> RQVAE:
        torch.manual_seed(7)
        return RQVAE(
            input_dim=8,
            hidden_dims=[8],
            latent_dim=4,
            vocab_size=7,
            maxlen=maxlen,
            distance_batch_size=2,
            varlen=True,
            stop_num_heads=1,
            stop_dropout=0.0,
        )

    def test_encoder_and_decoder_contract(self):
        model = self.make_model()
        x = F.normalize(torch.randn(6, 8), dim=-1)
        codes = model.inference_encoder()(x)
        reconstructions = model.inference_decoder()(codes)

        self.assertEqual(codes.shape, (6, 3))
        self.assertEqual(codes.dtype, torch.long)
        self.assertEqual(reconstructions.shape, (6, 3, 8))
        self.assertTrue(bool((codes >= 0).all()))
        self.assertTrue(bool((codes < model.vocab_size).all()))
        self.assertTrue(
            torch.allclose(
                reconstructions.norm(dim=-1), torch.ones(6, 3), atol=1e-5
            )
        )
        self.assertFalse(model.inference_encoder().varlen)

    def test_chunked_nearest_neighbor_matches_full_distance(self):
        model = self.make_model()
        x = torch.randn(11, 4)
        codebook = torch.randn(7, 4)
        expected = model._squared_euclidean(x, codebook).argmin(dim=-1)
        actual = model.nearest_codes(x, codebook, batch_size=3)
        self.assertTrue(torch.equal(actual, expected))

    def test_prefix_decoder_uses_cumulative_codewords(self):
        model = RQVAE(
            input_dim=2,
            hidden_dims=[],
            latent_dim=2,
            vocab_size=2,
            maxlen=2,
        )
        with torch.no_grad():
            model.decoder[0].weight.copy_(torch.eye(2))
            model.decoder[0].bias.zero_()
            model.codebooks.copy_(
                torch.tensor(
                    [
                        [[1.0, 0.0], [0.0, 1.0]],
                        [[1.0, 1.0], [-1.0, 1.0]],
                    ]
                )
            )

        decoded = model.decode_prefixes(torch.tensor([[0, 0], [1, 1]]))
        expected = F.normalize(
            torch.tensor(
                [
                    [[1.0, 0.0], [2.0, 1.0]],
                    [[0.0, 1.0], [-1.0, 2.0]],
                ]
            ),
            dim=-1,
        )
        self.assertTrue(torch.allclose(decoded, expected, atol=1e-6))

    def test_initialization_is_deterministic_and_supports_small_catalogs(self):
        x = F.normalize(torch.randn(5, 8), dim=-1)
        first = self.make_model()
        second = self.make_model()
        second.load_state_dict(first.state_dict())

        first.initialize_codebooks(x, seed=123, encode_batch_size=2)
        second.initialize_codebooks(x, seed=123, encode_batch_size=3)

        self.assertTrue(torch.equal(first.codebooks, second.codebooks))
        self.assertTrue(bool(torch.isfinite(first.codebooks).all()))

    def test_losses_route_gradients_to_all_parameter_groups(self):
        model = self.make_model()
        x = F.normalize(torch.randn(16, 8), dim=-1)
        loss, metrics = model(x)
        loss.backward()

        self.assertGreater(model.encoder[0].weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.decoder[0].weight.grad.abs().sum().item(), 0.0)
        for level in range(model.maxlen):
            self.assertGreater(model.codebooks.grad[level].abs().sum().item(), 0.0)
        self.assertIn("recon_loss", metrics)
        self.assertIn("codebook_loss", metrics)
        self.assertIn("commitment_loss", metrics)
        self.assertIn("quantization_loss@3", metrics)

    def test_state_dict_round_trip_preserves_codes_and_reconstructions(self):
        model = self.make_model()
        x = F.normalize(torch.randn(6, 8), dim=-1)
        model.initialize_codebooks(x, seed=3)
        expected_codes = model.encode(x)
        expected_reconstructions = model.decode_prefixes(expected_codes)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rqvae.pt"
            torch.save(model.state_dict(), path)
            restored = self.make_model()
            restored.load_state_dict(torch.load(path, weights_only=True))

        actual_codes = restored.encode(x)
        actual_reconstructions = restored.decode_prefixes(actual_codes)
        self.assertTrue(torch.equal(actual_codes, expected_codes))
        self.assertTrue(
            torch.equal(actual_reconstructions, expected_reconstructions)
        )

    def test_tiny_model_can_reduce_its_training_objective(self):
        torch.manual_seed(13)
        model = RQVAE(
            input_dim=4,
            hidden_dims=[8],
            latent_dim=3,
            vocab_size=4,
            maxlen=2,
            distance_batch_size=4,
        )
        x = F.normalize(torch.randn(12, 4), dim=-1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        initial = float(model(x)[0].detach())
        for _ in range(60):
            loss, _ = model(x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        final = float(model(x)[0].detach())
        self.assertLess(final, initial)

    def test_stop_hazards_produce_length_and_survival_probabilities(self):
        stop_logits = torch.zeros(2, 2)
        length_logits, survival_logits = RQVAE._process_stop_logits(
            stop_logits, return_survival_logits=True
        )
        expected_lengths = torch.tensor([0.5, 0.25, 0.25]).expand(2, -1)
        expected_survival = torch.tensor([1.0, 0.5, 0.25]).expand(2, -1)
        self.assertTrue(
            torch.allclose(length_logits.softmax(dim=-1), expected_lengths)
        )
        self.assertTrue(
            torch.allclose(survival_logits.exp(), expected_survival)
        )

    def test_varlen_objective_matches_manual_calculation(self):
        model = self.make_varlen_model()
        model.eval()
        x = F.normalize(torch.randn(5, 8), dim=-1)
        length_cost = torch.tensor(5.0)
        length_beta = torch.tensor(0.002 / 8)

        latent = model.encoder(x)
        _, hard_prefixes, residuals, codebook, commitment = model._quantize(latent)
        prefix_st = latent[:, None] + (hard_prefixes - latent[:, None]).detach()
        reconstructions = F.normalize(
            model.decoder(prefix_st.reshape(-1, model.latent_dim)).float(), dim=-1
        ).view(5, model.maxlen, model.input_dim)
        length_logits, _ = model._length_outputs(x, residuals)
        probs = length_logits.softmax(dim=-1)
        smoothed = 0.9 * probs + 0.1 / model.maxlen
        alive = probs.flip(-1).cumsum(-1).flip(-1)
        target = F.normalize(x.float(), dim=-1)
        expected_recon = (
            smoothed * (reconstructions - target[:, None]).square().mean(-1)
        ).sum(-1).mean()
        expected_vq = (alive * (codebook + commitment)).sum(-1).mean()
        values = torch.arange(1, model.maxlen + 1, dtype=probs.dtype)
        expected_length = (probs * values[None]).sum(-1).mean()
        entropy = -(probs * length_logits.log_softmax(-1)).sum(-1).mean()
        expected = expected_recon + expected_vq + length_beta * (
            -entropy + length_cost * expected_length
        )

        actual, metrics = model(
            x, length_cost=length_cost, length_beta=length_beta
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertTrue(torch.allclose(metrics["E_L"], expected_length, atol=1e-6))
        self.assertTrue(torch.equal(metrics["length_beta"], length_beta))

    def test_varlen_inference_contract_and_modes(self):
        model = self.make_varlen_model()
        model.eval()
        x = F.normalize(torch.randn(6, 8), dim=-1)
        for mode in ("argmax", "mean", "median"):
            encoder = model.inference_encoder(mode=mode)
            codes, lengths = encoder(x)
            self.assertTrue(encoder.varlen)
            self.assertEqual(codes.shape, (6, 3))
            self.assertEqual(lengths.shape, (6,))
            self.assertTrue(bool((lengths >= 1).all()))
            self.assertTrue(bool((lengths <= 3).all()))

    def test_stop_transformer_is_causal(self):
        model = self.make_varlen_model()
        model.eval()
        x = F.normalize(torch.randn(3, 8), dim=-1)
        residuals = torch.randn(3, 3, 4)
        changed = residuals.clone()
        changed[:, 1] += 100.0
        original_logits, _ = model._length_outputs(x, residuals)
        changed_logits, _ = model._length_outputs(x, changed)
        self.assertTrue(
            torch.allclose(original_logits[:, 0], changed_logits[:, 0], atol=1e-6)
        )

    def test_varlen_maxlen_one_is_deterministic(self):
        model = self.make_varlen_model(maxlen=1)
        x = F.normalize(torch.randn(4, 8), dim=-1)
        loss, metrics = model(x, length_cost=5.0, length_beta=0.002 / 8)
        codes, lengths = model.inference_encoder()(x)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(codes.shape, (4, 1))
        self.assertEqual(lengths.tolist(), [1, 1, 1, 1])
        self.assertEqual(float(metrics["E_L"]), 1.0)

    def test_varlen_gradients_and_stop_codebook_isolation(self):
        model = self.make_varlen_model()
        model.eval()
        x = F.normalize(torch.randn(12, 8), dim=-1)
        loss, _ = model(x, length_cost=5.0, length_beta=0.002 / 8)
        loss.backward()
        self.assertGreater(model.encoder[0].weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.decoder[0].weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.stop_predictor.weight.grad.abs().sum().item(), 0.0)
        for level in range(model.maxlen):
            self.assertGreater(model.codebooks.grad[level].abs().sum().item(), 0.0)

        model.zero_grad(set_to_none=True)
        latent = model.encoder(x)
        _, _, residuals, _, _ = model._quantize(latent)
        logits, _ = model._length_outputs(x, residuals)
        probs = logits.softmax(dim=-1)
        values = torch.arange(1, model.maxlen + 1, dtype=probs.dtype)
        length_only = (probs * values[None]).sum(dim=-1).mean()
        length_only.backward()
        self.assertIsNone(model.codebooks.grad)
        self.assertGreater(model.encoder[0].weight.grad.abs().sum().item(), 0.0)

    def test_varlen_state_dict_round_trip(self):
        model = self.make_varlen_model()
        model.eval()
        x = F.normalize(torch.randn(6, 8), dim=-1)
        expected_codes, expected_lengths = model.inference_encoder()(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "varlen_rqvae.pt"
            torch.save(model.state_dict(), path)
            restored = self.make_varlen_model()
            restored.load_state_dict(torch.load(path, weights_only=True))
            restored.eval()

        actual_codes, actual_lengths = restored.inference_encoder()(x)
        self.assertTrue(torch.equal(actual_codes, expected_codes))
        self.assertTrue(torch.equal(actual_lengths, expected_lengths))


if __name__ == "__main__":
    unittest.main()
