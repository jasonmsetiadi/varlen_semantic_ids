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
    """Fixed-length residual-quantized autoencoder for semantic IDs."""

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
    ) -> None:
        super().__init__()
        if input_dim < 1 or latent_dim < 1:
            raise ValueError("input_dim and latent_dim must be positive")
        if vocab_size < 1 or maxlen < 1:
            raise ValueError("vocab_size and maxlen must be positive")
        if distance_batch_size < 1:
            raise ValueError("distance_batch_size must be positive")

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.vocab_size = int(vocab_size)
        self.maxlen = int(maxlen)
        self.commitment_beta = float(commitment_beta)
        self.quantization_loss_weight = float(quantization_loss_weight)
        self.reconstruction_loss_weight = float(reconstruction_loss_weight)
        self.distance_batch_size = int(distance_batch_size)

        hidden_dims = [int(dim) for dim in hidden_dims]
        if any(dim < 1 for dim in hidden_dims):
            raise ValueError("hidden_dims must contain only positive dimensions")

        self.encoder = _mlp(self.input_dim, hidden_dims, self.latent_dim)
        self.decoder = _mlp(self.latent_dim, list(reversed(hidden_dims)), self.input_dim)
        self.codebooks = nn.Parameter(
            torch.empty(self.maxlen, self.vocab_size, self.latent_dim)
        )
        nn.init.normal_(self.codebooks, mean=0.0, std=0.02)

    @property
    def varlen(self) -> bool:
        return False

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        residual = latent
        hard_sum = torch.zeros_like(latent)
        codes = []
        codebook_losses = []
        commitment_losses = []
        per_level_losses = []

        for level in range(self.maxlen):
            codebook = self.codebooks[level]
            level_codes = self.nearest_codes(residual, codebook)
            quantized = F.embedding(level_codes, codebook)

            codebook_loss = F.mse_loss(quantized, residual.detach())
            commitment_loss = self.commitment_beta * F.mse_loss(
                residual, quantized.detach()
            )
            level_loss = codebook_loss + commitment_loss

            codes.append(level_codes)
            codebook_losses.append(codebook_loss)
            commitment_losses.append(commitment_loss)
            per_level_losses.append(level_loss)
            hard_sum = hard_sum + quantized

            # Later residual losses should train the encoder but not earlier
            # codebooks. Codebooks receive gradients from their own codebook loss.
            residual = residual - quantized.detach()

        return (
            torch.stack(codes, dim=-1),
            hard_sum,
            torch.stack(codebook_losses).sum(),
            torch.stack(commitment_losses).sum(),
            per_level_losses,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"Expected x [B,{self.input_dim}], got {tuple(x.shape)}")

        target = F.normalize(x.float(), dim=-1)
        latent = self.encoder(x)
        _, hard_sum, codebook_loss, commitment_loss, per_level = self._quantize(
            latent
        )

        # The forward value is the hard sum, while the reconstruction gradient
        # with respect to the encoder is the identity straight-through estimate.
        quantized_st = latent + (hard_sum - latent).detach()
        reconstruction = F.normalize(self.decoder(quantized_st).float(), dim=-1)
        reconstruction_loss = F.mse_loss(reconstruction, target)
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
        for level, level_loss in enumerate(per_level, start=1):
            metrics[f"quantization_loss@{level}"] = level_loss.detach()
        return loss, metrics

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"Expected x [B,{self.input_dim}], got {tuple(x.shape)}")
        latent = self.encoder(x)
        residual = latent
        codes = []
        for level in range(self.maxlen):
            level_codes = self.nearest_codes(residual, self.codebooks[level])
            quantized = F.embedding(level_codes, self.codebooks[level])
            codes.append(level_codes)
            residual = residual - quantized
        return torch.stack(codes, dim=-1).to(torch.long)

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

    def inference_encoder(self) -> "RQVAEInferenceEncoder":
        return RQVAEInferenceEncoder(self)

    def inference_decoder(self) -> "RQVAEInferenceDecoder":
        return RQVAEInferenceDecoder(self)


class RQVAEInferenceEncoder(nn.Module):
    def __init__(self, model: RQVAE) -> None:
        super().__init__()
        self.model = model

    @property
    def vocab_size(self) -> int:
        return self.model.vocab_size

    @property
    def maxlen(self) -> int:
        return self.model.maxlen

    @property
    def varlen(self) -> bool:
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode(x)


class RQVAEInferenceDecoder(nn.Module):
    def __init__(self, model: RQVAE) -> None:
        super().__init__()
        self.model = model

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.model.decode_prefixes(codes)
