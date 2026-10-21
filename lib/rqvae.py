from __future__ import annotations

from contextlib import nullcontext
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
) -> nn.Sequential:
    dims = [input_dim, *hidden_dims, output_dim]
    layers: list[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if index < len(dims) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class RQVAE(nn.Module):
    """Residual-quantized autoencoder for fixed- or variable-length IDs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        vocab_size: int,
        maxlen: int,
        commitment_beta: float = 0.25,
        quantization_loss_weight: float = 1.0,
        reconstruction_loss_weight: float = 1.0,
        distance_batch_size: int = 512,
        varlen: bool = False,
        stop_num_heads: int = 1,
        stop_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim < 1 or latent_dim < 1:
            raise ValueError("input_dim and latent_dim must be positive")
        if vocab_size < 1 or maxlen < 1:
            raise ValueError("vocab_size and maxlen must be positive")
        if distance_batch_size < 1:
            raise ValueError("distance_batch_size must be positive")
        if stop_num_heads < 1:
            raise ValueError("stop_num_heads must be positive")
        if latent_dim % stop_num_heads != 0:
            raise ValueError("latent_dim must be divisible by stop_num_heads")

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.vocab_size = int(vocab_size)
        self.maxlen = int(maxlen)
        self.commitment_beta = float(commitment_beta)
        self.quantization_loss_weight = float(quantization_loss_weight)
        self.reconstruction_loss_weight = float(reconstruction_loss_weight)
        self.distance_batch_size = int(distance_batch_size)
        self._varlen = bool(varlen)

        hidden_dims = [int(dim) for dim in hidden_dims]
        if any(dim < 1 for dim in hidden_dims):
            raise ValueError("hidden_dims must contain only positive dimensions")

        self.encoder = _mlp(self.input_dim, hidden_dims, self.latent_dim)
        self.decoder = _mlp(self.latent_dim, list(reversed(hidden_dims)), self.input_dim)
        self.codebooks = nn.Parameter(
            torch.empty(self.maxlen, self.vocab_size, self.latent_dim)
        )
        nn.init.normal_(self.codebooks, mean=0.0, std=0.02)

        if self._varlen and self.maxlen > 1:
            self.stop_backbone = nn.Sequential(
                nn.Linear(self.input_dim, self.latent_dim),
                nn.LayerNorm(self.latent_dim),
                nn.GELU(),
            )
            self.stop_positions = nn.Parameter(
                torch.empty(self.maxlen, self.latent_dim)
            )
            nn.init.normal_(self.stop_positions, mean=0.0, std=0.02)
            self.stop_transformer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=int(stop_num_heads),
                dim_feedforward=4 * self.latent_dim,
                dropout=float(stop_dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.stop_norm = nn.LayerNorm(self.latent_dim)
            self.stop_predictor = nn.Linear(self.latent_dim, 1)
            self.stop_bias = nn.Parameter(torch.zeros(self.maxlen - 1))

    @property
    def varlen(self) -> bool:
        return self._varlen

    @staticmethod
    def _process_stop_logits(
        stop_logits: torch.Tensor,
        return_survival_logits: bool = False,
    ):
        """Convert conditional stop hazards into length probabilities."""
        log_cont = F.logsigmoid(-stop_logits)
        survival_after = torch.cumsum(log_cont, dim=-1)
        length_logits = torch.cat(
            [
                survival_after - log_cont + F.logsigmoid(stop_logits),
                survival_after[..., -1:],
            ],
            dim=-1,
        )
        if return_survival_logits:
            survival_logits = torch.cat(
                [
                    survival_after.new_zeros(survival_after.shape[:-1] + (1,)),
                    survival_after,
                ],
                dim=-1,
            )
            return length_logits, survival_logits
        return length_logits

    @staticmethod
    def _squared_euclidean(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        x_norm = x.square().sum(dim=-1, keepdim=True)
        c_norm = codebook.square().sum(dim=-1).unsqueeze(0)
        return x_norm + c_norm - 2.0 * (x @ codebook.t())

    def nearest_codes(
        self,
        x: torch.Tensor,
        codebook: torch.Tensor,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Return nearest code indices while bounding the distance matrix size."""
        if x.ndim != 2 or codebook.ndim != 2:
            raise ValueError("x and codebook must both be rank-2 tensors")
        if x.size(1) != codebook.size(1):
            raise ValueError(
                f"Feature mismatch: x has {x.size(1)}, codebook has {codebook.size(1)}"
            )

        chunk_size = int(batch_size or self.distance_batch_size)
        codes = []
        # Assignments are non-differentiable. Computing them in float32 avoids
        # building a large autograd graph and improves bf16 distance stability.
        x_detached = x.detach().float()
        codebook_detached = codebook.detach().float()
        for start in range(0, x.size(0), chunk_size):
            distances = self._squared_euclidean(
                x_detached[start : start + chunk_size], codebook_detached
            )
            codes.append(distances.argmin(dim=-1))
        return torch.cat(codes, dim=0)

    def _quantize(
        self, latent: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        residual = latent
        hard_sum = torch.zeros_like(latent)
        codes = []
        hard_prefixes = []
        residual_states = []
        codebook_losses = []
        commitment_losses = []

        for level in range(self.maxlen):
            codebook = self.codebooks[level]
            level_codes = self.nearest_codes(residual, codebook)
            quantized = F.embedding(level_codes, codebook)

            codebook_loss = (quantized - residual.detach()).square().mean(dim=-1)
            commitment_loss = self.commitment_beta * (
                residual - quantized.detach()
            ).square().mean(dim=-1)

            codes.append(level_codes)
            codebook_losses.append(codebook_loss)
            commitment_losses.append(commitment_loss)
            hard_sum = hard_sum + quantized
            hard_prefixes.append(hard_sum)

            # Later residual losses should train the encoder but not earlier
            # codebooks. Codebooks receive gradients from their own codebook loss.
            residual = residual - quantized.detach()
            residual_states.append(residual)

        return (
            torch.stack(codes, dim=-1),
            torch.stack(hard_prefixes, dim=1),
            torch.stack(residual_states, dim=1),
            torch.stack(codebook_losses, dim=1),
            torch.stack(commitment_losses, dim=1),
        )

    def _length_outputs(
        self,
        x: torch.Tensor,
        residual_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._varlen:
            raise RuntimeError("Length outputs are only available when varlen=True")
        if self.maxlen == 1:
            zeros = x.new_zeros((x.size(0), 1), dtype=torch.float32)
            return zeros, zeros

        initial = self.stop_backbone(x).unsqueeze(1)
        stop_states = torch.cat([initial, residual_states[:, :-1]], dim=1)
        stop_states = stop_states + self.stop_positions[None].to(stop_states.dtype)
        causal_mask = torch.triu(
            torch.ones(
                self.maxlen,
                self.maxlen,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        stop_hidden = self.stop_transformer(stop_states, src_mask=causal_mask)
        stop_hidden = self.stop_norm(stop_hidden)
        stop_logits = (
            self.stop_predictor(stop_hidden[:, 1:]).squeeze(dim=-1)
            + self.stop_bias[None]
        )
        return self._process_stop_logits(
            stop_logits.float(), return_survival_logits=True
        )

    def forward(
        self,
        x: torch.Tensor,
        length_cost: torch.Tensor | float | None = None,
        length_beta: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"Expected x [B,{self.input_dim}], got {tuple(x.shape)}")

        target = F.normalize(x.float(), dim=-1)
        latent = self.encoder(x)
        (
            _,
            hard_prefixes,
            residual_states,
            codebook_losses,
            commitment_losses,
        ) = self._quantize(latent)

        if self._varlen:
            if length_cost is None or length_beta is None:
                raise ValueError(
                    "length_cost and length_beta are required when varlen=True"
                )
            prefix_st = latent[:, None] + (hard_prefixes - latent[:, None]).detach()
            flat_prefixes = prefix_st.reshape(-1, self.latent_dim)
            reconstructions = F.normalize(
                self.decoder(flat_prefixes).float(), dim=-1
            ).view(x.size(0), self.maxlen, self.input_dim)

            length_logits, _ = self._length_outputs(x, residual_states)
            length_probs = length_logits.softmax(dim=-1)
            length_logprobs = length_logits.log_softmax(dim=-1)
            smoothed_length_probs = 0.9 * length_probs + 0.1 / self.maxlen
            alive = length_probs.flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))

            reconstruction_per_level = (
                reconstructions - target[:, None]
            ).square().mean(dim=-1)
            reconstruction_loss = (
                smoothed_length_probs * reconstruction_per_level
            ).sum(dim=-1).mean()
            quantization_alive = alive.detach()
            codebook_loss = (quantization_alive * codebook_losses).sum(dim=-1).mean()
            commitment_loss = (quantization_alive * commitment_losses).sum(dim=-1).mean()
            quantization_loss = codebook_loss + commitment_loss

            lengths = torch.arange(
                1,
                self.maxlen + 1,
                device=length_probs.device,
                dtype=length_probs.dtype,
            )
            expected_length = (length_probs * lengths[None]).sum(dim=-1).mean()
            length_entropy = -(
                length_probs * length_logprobs
            ).sum(dim=-1).mean()
            length_cost_tensor = torch.as_tensor(
                length_cost, device=x.device, dtype=length_probs.dtype
            )
            length_beta_tensor = torch.as_tensor(
                length_beta, device=x.device, dtype=length_probs.dtype
            )
            length_term = -length_entropy + length_cost_tensor * expected_length
            length_loss = length_beta_tensor * length_term
            loss = (
                self.reconstruction_loss_weight * reconstruction_loss
                + self.quantization_loss_weight * quantization_loss
                + length_loss
            )

            metrics: dict[str, torch.Tensor] = {
                "loss": loss.detach(),
                "recon_loss": reconstruction_loss.detach(),
                "quantization_loss": quantization_loss.detach(),
                "codebook_loss": codebook_loss.detach(),
                "commitment_loss": commitment_loss.detach(),
                "length_term": length_term.detach(),
                "length_loss": length_loss.detach(),
                "length_beta": length_beta_tensor.detach(),
                "E_L": expected_length.detach(),
                "length_entropy": length_entropy.detach(),
            }
            discrete_lengths = length_probs.argmax(dim=-1)
            length_counts = torch.bincount(
                discrete_lengths, minlength=self.maxlen
            )
            for level in range(self.maxlen):
                metrics[f"recon_loss@{level + 1}"] = (
                    reconstruction_per_level[:, level].mean().detach()
                )
                raw_level_loss = (
                    codebook_losses[:, level] + commitment_losses[:, level]
                ).mean()
                metrics[f"quantization_loss@{level + 1}"] = raw_level_loss.detach()
                metrics[f"survival@{level + 1}"] = alive[:, level].mean().detach()
                metrics[f"length_fraction@{level + 1}"] = (
                    length_counts[level].float() / x.size(0)
                ).detach()
                metrics[f"length_probability@{level + 1}"] = (
                    length_probs[:, level].mean().detach()
                )
            return loss, metrics

        # The forward value is the hard sum, while the reconstruction gradient
        # with respect to the encoder is the identity straight-through estimate.
        hard_sum = hard_prefixes[:, -1]
        quantized_st = latent + (hard_sum - latent).detach()
        reconstruction = F.normalize(self.decoder(quantized_st).float(), dim=-1)
        reconstruction_loss = F.mse_loss(reconstruction, target)
        codebook_loss = codebook_losses.sum(dim=-1).mean()
        commitment_loss = commitment_losses.sum(dim=-1).mean()
        quantization_loss = codebook_loss + commitment_loss
        loss = (
            self.reconstruction_loss_weight * reconstruction_loss
            + self.quantization_loss_weight * quantization_loss
        )

        metrics: dict[str, torch.Tensor] = {
            "loss": loss.detach(),
            "recon_loss": reconstruction_loss.detach(),
            "quantization_loss": quantization_loss.detach(),
            "codebook_loss": codebook_loss.detach(),
            "commitment_loss": commitment_loss.detach(),
        }
        for level in range(self.maxlen):
            level_loss = (
                codebook_losses[:, level] + commitment_losses[:, level]
            ).mean()
            metrics[f"quantization_loss@{level + 1}"] = level_loss.detach()
        return loss, metrics

    @torch.no_grad()
    def encode(self, x: torch.Tensor, length_mode: str = "argmax"):
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"Expected x [B,{self.input_dim}], got {tuple(x.shape)}")
        latent = self.encoder(x)
        codes, _, residual_states, _, _ = self._quantize(latent)
        codes = codes.to(torch.long)
        if not self._varlen:
            return codes

        length_logits, _ = self._length_outputs(x, residual_states)
        if length_mode == "argmax":
            lengths = length_logits.argmax(dim=-1) + 1
        elif length_mode == "mean":
            probs = length_logits.softmax(dim=-1)
            values = torch.arange(
                1,
                self.maxlen + 1,
                device=probs.device,
                dtype=probs.dtype,
            )
            lengths = (probs * values[None]).sum(dim=-1).round().to(torch.long)
        elif length_mode == "median":
            cdf = length_logits.softmax(dim=-1).cumsum(dim=-1)
            lengths = (cdf >= 0.5).to(torch.long).argmax(dim=-1) + 1
        else:
            raise ValueError(f"Unknown length inference mode: {length_mode!r}")
        return codes, lengths.clamp(1, self.maxlen)

    @torch.no_grad()
    def decode_prefixes(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 2 or codes.size(1) != self.maxlen:
            raise ValueError(f"Expected codes [B,{self.maxlen}], got {tuple(codes.shape)}")
        if codes.numel() and (
            bool((codes < 0).any()) or bool((codes >= self.vocab_size).any())
        ):
            raise ValueError("codes contain values outside the codebook vocabulary")

        cumulative = torch.zeros(
            codes.size(0),
            self.latent_dim,
            device=codes.device,
            dtype=self.codebooks.dtype,
        )
        prefixes = []
        for level in range(self.maxlen):
            cumulative = cumulative + F.embedding(
                codes[:, level].to(torch.long), self.codebooks[level]
            )
            prefixes.append(cumulative.clone())
        latent_prefixes = torch.stack(prefixes, dim=1)
        flat = latent_prefixes.reshape(-1, self.latent_dim)
        decoded = F.normalize(self.decoder(flat).float(), dim=-1)
        return decoded.view(codes.size(0), self.maxlen, self.input_dim)

    @torch.no_grad()
    def initialize_codebooks(
        self,
        embeddings: torch.Tensor,
        seed: int = 42,
        encode_batch_size: int = 4096,
    ) -> None:
        """Initialize every level from unique-item latent residual samples."""
        if embeddings.ndim != 2 or embeddings.size(1) != self.input_dim:
            raise ValueError(
                f"Expected embeddings [N,{self.input_dim}], got {tuple(embeddings.shape)}"
            )
        if embeddings.size(0) == 0:
            raise ValueError("Cannot initialize codebooks from an empty tensor")
        if not bool(torch.isfinite(embeddings).all()):
            raise ValueError("Initialization embeddings contain non-finite values")

        was_training = self.training
        self.eval()
        encoded = []
        device_type = embeddings.device.type
        autocast_context = (
            lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device_type == "cuda"
            else nullcontext()
        )
        for start in range(0, embeddings.size(0), encode_batch_size):
            with autocast_context():
                encoded.append(self.encoder(embeddings[start : start + encode_batch_size]))
        residual = torch.cat(encoded, dim=0).float()
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("Encoder produced non-finite initialization latents")

        generator = torch.Generator(device=residual.device)
        generator.manual_seed(int(seed))
        num_items = residual.size(0)
        for level in range(self.maxlen):
            if num_items >= self.vocab_size:
                sample_indices = torch.randperm(
                    num_items, generator=generator, device=residual.device
                )[: self.vocab_size]
            else:
                unique_indices = torch.randperm(
                    num_items, generator=generator, device=residual.device
                )
                extra_indices = torch.randint(
                    num_items,
                    (self.vocab_size - num_items,),
                    generator=generator,
                    device=residual.device,
                )
                sample_indices = torch.cat([unique_indices, extra_indices])

            initial_codebook = residual.index_select(0, sample_indices)
            self.codebooks[level].copy_(initial_codebook.to(self.codebooks.dtype))
            assignments = self.nearest_codes(residual, self.codebooks[level])
            residual = residual - F.embedding(assignments, self.codebooks[level]).float()
            if not bool(torch.isfinite(residual).all()):
                raise ValueError(f"Non-finite residuals after initialization level {level + 1}")

        self.train(was_training)

    def inference_encoder(self, mode: str = "argmax") -> "RQVAEInferenceEncoder":
        return RQVAEInferenceEncoder(self, mode=mode)

    def inference_decoder(self) -> "RQVAEInferenceDecoder":
        return RQVAEInferenceDecoder(self)


class RQVAEInferenceEncoder(nn.Module):
    def __init__(self, model: RQVAE, mode: str = "argmax") -> None:
        super().__init__()
        self.model = model
        if mode not in {"argmax", "mean", "median"}:
            raise ValueError(f"Unknown length inference mode: {mode!r}")
        self.mode = mode

    @property
    def vocab_size(self) -> int:
        return self.model.vocab_size

    @property
    def maxlen(self) -> int:
        return self.model.maxlen

    @property
    def varlen(self) -> bool:
        return self.model.varlen

    def forward(self, x: torch.Tensor):
        return self.model.encode(x, length_mode=self.mode)


class RQVAEInferenceDecoder(nn.Module):
    def __init__(self, model: RQVAE) -> None:
        super().__init__()
        self.model = model

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.model.decode_prefixes(codes)
