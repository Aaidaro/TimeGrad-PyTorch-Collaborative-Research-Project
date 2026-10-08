"""Neural building blocks for the TimeGrad noise-prediction network."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DiffusionEmbedding(nn.Module):
    """Fourier embedding of a discrete diffusion step.

    The construction follows the public PyTorchTS TimeGrad implementation:
    sinusoidal features are fixed, then transformed by two learned layers.
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        projection_dim: int = 64,
        max_steps: int = 500,
    ) -> None:
        super().__init__()
        if embedding_dim < 1 or max_steps < 1:
            raise ValueError("embedding_dim and max_steps must be positive")

        steps = torch.arange(max_steps, dtype=torch.float32).unsqueeze(1)
        dims = torch.arange(embedding_dim, dtype=torch.float32).unsqueeze(0)
        frequencies = 10.0 ** (dims * 4.0 / embedding_dim)
        table = steps * frequencies
        table = torch.cat((table.sin(), table.cos()), dim=1)
        self.register_buffer("embedding", table, persistent=False)

        self.projection1 = nn.Linear(2 * embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, step: torch.Tensor) -> torch.Tensor:
        if step.dtype != torch.long:
            step = step.long()
        if torch.any(step < 0) or torch.any(step >= self.embedding.shape[0]):
            raise ValueError("diffusion step lies outside the embedding table")
        hidden = F.silu(self.projection1(self.embedding[step]))
        return F.silu(self.projection2(hidden))


class ConditionerUpsampler(nn.Module):
    """Broadcast an RNN conditioning vector over target dimensions."""

    def __init__(self, conditioning_length: int, target_dim: int) -> None:
        super().__init__()
        middle = max(1, target_dim // 2)
        self.linear1 = nn.Linear(conditioning_length, middle)
        self.linear2 = nn.Linear(middle, target_dim)

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        hidden = F.leaky_relu(self.linear1(conditioning), negative_slope=0.4)
        return F.leaky_relu(self.linear2(hidden), negative_slope=0.4)


class ResidualBlock(nn.Module):
    """Conditional gated WaveNet block with circular spatial padding."""

    def __init__(
        self,
        residual_channels: int,
        diffusion_projection_dim: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.diffusion_projection = nn.Linear(
            diffusion_projection_dim, residual_channels
        )
        self.conditioner_projection = nn.Conv1d(
            1, 2 * residual_channels, kernel_size=1
        )
        self.output_projection = nn.Conv1d(
            residual_channels, 2 * residual_channels, kernel_size=1
        )

        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(
        self,
        inputs: torch.Tensor,
        conditioner: torch.Tensor,
        diffusion_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        diffusion_term = self.diffusion_projection(diffusion_embedding).unsqueeze(-1)
        conditioner_term = self.conditioner_projection(conditioner)
        hidden = self.dilated_conv(inputs + diffusion_term) + conditioner_term
        gate, value = hidden.chunk(2, dim=1)
        hidden = torch.sigmoid(gate) * torch.tanh(value)
        hidden = F.leaky_relu(
            self.output_projection(hidden), negative_slope=0.4
        )
        residual, skip = hidden.chunk(2, dim=1)
        return (inputs + residual) / math.sqrt(2.0), skip


class ConditionalWaveNet(nn.Module):
    """TimeGrad's conditional 1-D dilated convolutional denoiser.

    Inputs use ``(batch, channel=1, target_dim)`` layout. Convolutions run
    across the fixed target-dimension ordering, as in Figure 2 of the paper.
    """

    def __init__(
        self,
        target_dim: int,
        conditioning_length: int = 100,
        diffusion_embedding_dim: int = 16,
        diffusion_projection_dim: int = 64,
        residual_layers: int = 8,
        residual_channels: int = 8,
        dilation_cycle_length: int = 2,
        max_diffusion_steps: int = 500,
    ) -> None:
        super().__init__()
        if target_dim < 1:
            raise ValueError("target_dim must be positive")
        if residual_layers < 1 or residual_channels < 1:
            raise ValueError("residual_layers and residual_channels must be positive")

        self.target_dim = target_dim
        self.input_projection = nn.Conv1d(1, residual_channels, kernel_size=1)
        self.diffusion_embedding = DiffusionEmbedding(
            embedding_dim=diffusion_embedding_dim,
            projection_dim=diffusion_projection_dim,
            max_steps=max_diffusion_steps,
        )
        self.conditioner_upsampler = ConditionerUpsampler(
            conditioning_length=conditioning_length,
            target_dim=target_dim,
        )
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    diffusion_projection_dim=diffusion_projection_dim,
                    dilation=2 ** (layer % dilation_cycle_length),
                )
                for layer in range(residual_layers)
            ]
        )
        self.skip_projection = nn.Conv1d(
            residual_channels,
            residual_channels,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.output_projection = nn.Conv1d(
            residual_channels,
            1,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        # A zero final layer starts epsilon_theta at zero, matching DiffWave.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        noisy_target: torch.Tensor,
        diffusion_step: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_target.ndim != 3 or noisy_target.shape[1] != 1:
            raise ValueError("noisy_target must have shape (batch, 1, target_dim)")
        if noisy_target.shape[-1] != self.target_dim:
            raise ValueError(
                f"expected target_dim={self.target_dim}, got {noisy_target.shape[-1]}"
            )
        if conditioning.ndim == 3 and conditioning.shape[1] == 1:
            conditioning = conditioning[:, 0]
        if conditioning.ndim != 2:
            raise ValueError(
                "conditioning must have shape (batch, conditioning_length)"
            )

        hidden = F.leaky_relu(
            self.input_projection(noisy_target), negative_slope=0.4
        )
        step_embedding = self.diffusion_embedding(diffusion_step)
        conditioner = self.conditioner_upsampler(conditioning).unsqueeze(1)

        skips: list[torch.Tensor] = []
        for block in self.residual_blocks:
            hidden, skip = block(hidden, conditioner, step_embedding)
            skips.append(skip)

        hidden = torch.stack(skips).sum(dim=0) / math.sqrt(len(skips))
        hidden = F.leaky_relu(
            self.skip_projection(hidden), negative_slope=0.4
        )
        return self.output_projection(hidden)
