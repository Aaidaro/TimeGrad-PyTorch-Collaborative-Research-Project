"""Implementation of TimeGrad."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .components import ConditionalWaveNet
from .diffusion import GaussianDiffusion


@dataclass(frozen=True)
class TimeGradConfig:
    target_dim: int
    time_feature_dim: int
    prediction_length: int = 24
    context_length: int = 24
    lags: tuple[int, ...] = (1, 4, 12, 24, 48)
    rnn_layers: int = 2
    rnn_hidden_size: int = 40
    rnn_dropout: float = 0.1
    dimension_embedding_dim: int = 1
    conditioning_length: int = 100
    diffusion_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.1
    diffusion_embedding_dim: int = 16
    diffusion_projection_dim: int = 64
    residual_layers: int = 8
    residual_channels: int = 8
    dilation_cycle_length: int = 2
    scaling: bool = True

    @property
    def history_length(self) -> int:
        return self.context_length + max(self.lags)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["lags"] = list(self.lags)
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TimeGradConfig":
        values = dict(values)
        values["lags"] = tuple(int(lag) for lag in values.get("lags", cls.lags))
        return cls(**values)


class TimeGrad(nn.Module):
    """Autoregressive RNN whose conditional emission is a DDPM."""

    def __init__(self, config: TimeGradConfig) -> None:
        super().__init__()
        self.config = config
        if min(config.lags) < 1:
            raise ValueError("lags must be positive")
        if config.context_length < 1 or config.prediction_length < 1:
            raise ValueError("context_length and prediction_length must be positive")

        self.dimension_embedding = nn.Embedding(
            config.target_dim, config.dimension_embedding_dim
        )
        rnn_input_size = (
            config.target_dim * len(config.lags)
            + config.target_dim * config.dimension_embedding_dim
            + config.time_feature_dim
        )
        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=config.rnn_hidden_size,
            num_layers=config.rnn_layers,
            dropout=config.rnn_dropout if config.rnn_layers > 1 else 0.0,
            batch_first=True,
        )
        self.condition_projection = nn.Linear(
            config.rnn_hidden_size, config.conditioning_length
        )
        denoiser = ConditionalWaveNet(
            target_dim=config.target_dim,
            conditioning_length=config.conditioning_length,
            diffusion_embedding_dim=config.diffusion_embedding_dim,
            diffusion_projection_dim=config.diffusion_projection_dim,
            residual_layers=config.residual_layers,
            residual_channels=config.residual_channels,
            dilation_cycle_length=config.dilation_cycle_length,
            max_diffusion_steps=max(500, config.diffusion_steps),
        )
        self.diffusion = GaussianDiffusion(
            denoiser=denoiser,
            diffusion_steps=config.diffusion_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        )

    def _validate_history(self, history: torch.Tensor) -> None:
        expected = (self.config.history_length, self.config.target_dim)
        if history.ndim != 3 or tuple(history.shape[1:]) != expected:
            raise ValueError(
                "history must have shape "
                f"(batch, {expected[0]}, {expected[1]}), got {tuple(history.shape)}"
            )

    def _context_scale(self, history: torch.Tensor) -> torch.Tensor:
        if not self.config.scaling:
            return torch.ones(
                (history.shape[0], 1, self.config.target_dim),
                dtype=history.dtype,
                device=history.device,
            )
        context = history[:, -self.config.context_length :]
        scale = context.abs().mean(dim=1, keepdim=True)
        return torch.where(scale > 0.0, scale, torch.ones_like(scale))

    def _lagged_values(
        self,
        sequence: torch.Tensor,
        positions: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        lagged = [sequence.index_select(1, positions - lag) for lag in self.config.lags]
        # (batch, sequence, target_dim, number_of_lags)
        return torch.stack(lagged, dim=-1) / scale.unsqueeze(-1)

    def _rnn_inputs(
        self,
        sequence: torch.Tensor,
        positions: torch.Tensor,
        scale: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        lagged = self._lagged_values(sequence, positions, scale)
        batch_size, sequence_length = lagged.shape[:2]
        lagged = lagged.reshape(batch_size, sequence_length, -1)

        dimension_ids = torch.arange(
            self.config.target_dim, device=sequence.device, dtype=torch.long
        )
        dimension_embedding = self.dimension_embedding(dimension_ids).reshape(1, 1, -1)
        dimension_embedding = dimension_embedding.expand(
            batch_size, sequence_length, -1
        )
        return torch.cat((lagged, dimension_embedding, time_features), dim=-1)

    def training_loss(
        self,
        history: torch.Tensor,
        future: torch.Tensor,
        time_features: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Compute the simplified conditional DDPM objective (paper Eq. 11)."""

        self._validate_history(history)
        expected_future = (self.config.prediction_length, self.config.target_dim)
        if future.ndim != 3 or tuple(future.shape[1:]) != expected_future:
            raise ValueError(
                f"future must have shape (batch, {expected_future[0]}, "
                f"{expected_future[1]})"
            )

        sequence = torch.cat((history, future), dim=1)
        first_target = self.config.history_length - self.config.context_length
        positions = torch.arange(
            first_target,
            sequence.shape[1],
            device=sequence.device,
            dtype=torch.long,
        )
        if time_features.ndim != 3 or time_features.shape[1] != positions.numel():
            raise ValueError(
                "time_features must cover the context and prediction target positions"
            )

        scale = self._context_scale(history)
        inputs = self._rnn_inputs(sequence, positions, scale, time_features)
        rnn_outputs, _ = self.rnn(inputs)
        conditioning = self.condition_projection(rnn_outputs)
        scaled_target = sequence.index_select(1, positions) / scale

        batch_size, length, target_dim = scaled_target.shape
        return self.diffusion.training_loss(
            clean_target=scaled_target.reshape(batch_size * length, 1, target_dim),
            conditioning=conditioning.reshape(
                batch_size * length, self.config.conditioning_length
            ),
            generator=generator,
        )

    def forward(
        self,
        history: torch.Tensor,
        future: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.training_loss(history, future, time_features)

    @torch.no_grad()
    def forecast(
        self,
        history: torch.Tensor,
        context_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        num_samples: int = 100,
        nonnegative: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample autoregressive trajectories.

        Returns a tensor in ``(batch, samples, horizon, target_dim)`` layout.
        """

        self._validate_history(history)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        batch_size = history.shape[0]
        context_length = self.config.context_length
        prediction_length = self.config.prediction_length

        if context_time_features.shape != (
            batch_size,
            context_length,
            self.config.time_feature_dim,
        ):
            raise ValueError("context_time_features has an unexpected shape")
        if future_time_features.shape != (
            batch_size,
            prediction_length,
            self.config.time_feature_dim,
        ):
            raise ValueError("future_time_features has an unexpected shape")

        scale = self._context_scale(history)
        first_context_target = self.config.history_length - context_length
        context_positions = torch.arange(
            first_context_target,
            self.config.history_length,
            device=history.device,
            dtype=torch.long,
        )
        warm_inputs = self._rnn_inputs(
            history,
            context_positions,
            scale,
            context_time_features,
        )
        _, state = self.rnn(warm_inputs)

        sequence = history.repeat_interleave(num_samples, dim=0)
        repeated_scale = scale.repeat_interleave(num_samples, dim=0)
        repeated_future_features = future_time_features.repeat_interleave(
            num_samples, dim=0
        )
        state = tuple(item.repeat_interleave(num_samples, dim=1) for item in state)

        generated: list[torch.Tensor] = []
        for horizon_step in range(prediction_length):
            target_position = sequence.shape[1]
            positions = torch.tensor(
                [target_position], device=history.device, dtype=torch.long
            )
            step_features = repeated_future_features[
                :, horizon_step : horizon_step + 1
            ]
            step_input = self._rnn_inputs(
                sequence, positions, repeated_scale, step_features
            )
            rnn_output, state = self.rnn(step_input, state)
            conditioning = self.condition_projection(rnn_output[:, 0])
            scaled_sample = self.diffusion.sample(
                conditioning=conditioning,
                target_dim=self.config.target_dim,
                generator=generator,
            )[:, 0]
            sample = scaled_sample * repeated_scale[:, 0]
            if nonnegative:
                sample = sample.clamp_min(0.0)
            generated.append(sample)
            sequence = torch.cat((sequence, sample.unsqueeze(1)), dim=1)

        samples = torch.stack(generated, dim=1)
        return samples.reshape(
            batch_size,
            num_samples,
            prediction_length,
            self.config.target_dim,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
