"""Denoising diffusion process used as TimeGrad's emission distribution."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _extract(
    values: torch.Tensor,
    steps: torch.Tensor,
    shape: Sequence[int],
) -> torch.Tensor:
    gathered = values.gather(0, steps)
    return gathered.reshape(steps.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    """DDPM with epsilon prediction and the TimeGrad linear beta schedule."""

    def __init__(
        self,
        denoiser: nn.Module,
        diffusion_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.1,
    ) -> None:
        super().__init__()
        if diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least 2")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("expected 0 < beta_start < beta_end < 1")

        self.denoiser = denoiser
        self.diffusion_steps = diffusion_steps

        betas = torch.linspace(beta_start, beta_end, diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = torch.cat((torch.ones(1), alpha_bars[:-1]))
        posterior_variance = (
            betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer(
            "sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt()
        )
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(0.0))

    def q_sample(
        self,
        clean_target: torch.Tensor,
        steps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        return (
            _extract(self.sqrt_alpha_bars, steps, clean_target.shape) * clean_target
            + _extract(self.sqrt_one_minus_alpha_bars, steps, clean_target.shape)
            * noise
        )

    def training_loss(
        self,
        clean_target: torch.Tensor,
        conditioning: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch_size = clean_target.shape[0]
        steps = torch.randint(
            0,
            self.diffusion_steps,
            (batch_size,),
            device=clean_target.device,
            generator=generator,
        )
        noise = torch.randn(
            clean_target.shape,
            device=clean_target.device,
            dtype=clean_target.dtype,
            generator=generator,
        )
        noisy_target = self.q_sample(clean_target, steps, noise)
        predicted_noise = self.denoiser(noisy_target, steps, conditioning)
        return F.mse_loss(predicted_noise, noise, reduction="mean")

    @torch.no_grad()
    def sample(
        self,
        conditioning: torch.Tensor,
        target_dim: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw scaled samples with Algorithm 2 from the paper."""

        batch_size = conditioning.shape[0]
        sample = torch.randn(
            (batch_size, 1, target_dim),
            device=conditioning.device,
            dtype=conditioning.dtype,
            generator=generator,
        )

        for step_index in range(self.diffusion_steps - 1, -1, -1):
            steps = torch.full(
                (batch_size,),
                step_index,
                device=conditioning.device,
                dtype=torch.long,
            )
            predicted_noise = self.denoiser(sample, steps, conditioning)
            alpha = _extract(self.alphas, steps, sample.shape)
            beta = _extract(self.betas, steps, sample.shape)
            alpha_bar = _extract(self.alpha_bars, steps, sample.shape)
            mean = (sample - beta * predicted_noise / (1.0 - alpha_bar).sqrt())
            mean = mean / alpha.sqrt()

            if step_index > 0:
                noise = torch.randn(
                    sample.shape,
                    device=sample.device,
                    dtype=sample.dtype,
                    generator=generator,
                )
                variance = _extract(self.posterior_variance, steps, sample.shape)
                sample = mean + variance.sqrt() * noise
            else:
                sample = mean

        return sample
