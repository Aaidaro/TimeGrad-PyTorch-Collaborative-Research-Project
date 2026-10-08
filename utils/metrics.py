"""Sample-based probabilistic metrics, including the paper's CRPS-sum."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def empirical_crps(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Exact CRPS of an empirical sample distribution.

    ``samples`` has a leading sample axis. The sorted-sample identity avoids an
    ``O(num_samples**2)`` tensor, which matters for Taxi's 1,214 dimensions.
    """

    samples = np.asarray(samples, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if samples.ndim < 2 or samples.shape[1:] != target.shape:
        raise ValueError("samples and target shapes are incompatible")
    sample_count = samples.shape[0]
    if sample_count < 1:
        raise ValueError("at least one sample is required")

    absolute_error = np.mean(np.abs(samples - target[None, ...]), axis=0)
    if sample_count == 1:
        return absolute_error

    ordered = np.sort(samples, axis=0)
    coefficients = 2.0 * np.arange(1, sample_count + 1) - sample_count - 1.0
    pairwise_mean = 2.0 * np.tensordot(coefficients, ordered, axes=(0, 0))
    pairwise_mean /= float(sample_count * sample_count)
    return absolute_error - 0.5 * pairwise_mean


def quantile_loss(prediction: np.ndarray, target: np.ndarray, quantile: float) -> float:
    """GluonTS-compatible (factor-two) aggregate quantile loss."""

    error = np.asarray(target) - np.asarray(prediction)
    pinball = np.where(error >= 0.0, quantile * error, (quantile - 1.0) * error)
    return float(2.0 * pinball.sum())


@dataclass
class MetricAccumulator:
    """Streaming metrics across rolling forecast windows."""

    quantiles: np.ndarray = field(
        default_factory=lambda: np.arange(0.05, 1.0, 0.05, dtype=np.float64)
    )
    dimension_crps: float = 0.0
    sum_crps: float = 0.0
    absolute_target: float = 0.0
    absolute_sum_target: float = 0.0
    absolute_error: float = 0.0
    squared_error: float = 0.0
    target_squared_error_from_mean: float = 0.0
    element_count: int = 0
    sum_time_count: int = 0
    coverage_50_hits: int = 0
    coverage_90_hits: int = 0
    coverage_count: int = 0
    quantile_losses: np.ndarray = field(
        default_factory=lambda: np.zeros(19, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        self.quantiles = np.asarray(self.quantiles, dtype=np.float64)
        self.quantile_losses = np.zeros(len(self.quantiles), dtype=np.float64)

    def update(self, samples: np.ndarray, target: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if samples.ndim != 3 or target.ndim != 2:
            raise ValueError("expected samples=(S,H,D) and target=(H,D)")
        if samples.shape[1:] != target.shape:
            raise ValueError("sample and target shapes do not match")
        if not np.isfinite(samples).all() or not np.isfinite(target).all():
            raise ValueError("metrics received NaN or infinite values")

        crps_by_dimension = empirical_crps(samples, target)
        samples_sum = samples.sum(axis=-1)
        target_sum = target.sum(axis=-1)
        crps_for_sum = empirical_crps(samples_sum, target_sum)

        self.dimension_crps += float(crps_by_dimension.sum())
        self.sum_crps += float(crps_for_sum.sum())
        self.absolute_target += float(np.abs(target).sum())
        self.absolute_sum_target += float(np.abs(target_sum).sum())

        point_forecast = samples.mean(axis=0)
        residual = point_forecast - target
        self.absolute_error += float(np.abs(residual).sum())
        self.squared_error += float(np.square(residual).sum())
        self.element_count += int(target.size)
        self.sum_time_count += int(target_sum.size)

        for index, quantile in enumerate(self.quantiles):
            prediction = np.quantile(samples_sum, quantile, axis=0)
            self.quantile_losses[index] += quantile_loss(
                prediction, target_sum, float(quantile)
            )

        lower_50, upper_50 = np.quantile(samples, [0.25, 0.75], axis=0)
        lower_90, upper_90 = np.quantile(samples, [0.05, 0.95], axis=0)
        self.coverage_50_hits += int(
            np.logical_and(target >= lower_50, target <= upper_50).sum()
        )
        self.coverage_90_hits += int(
            np.logical_and(target >= lower_90, target <= upper_90).sum()
        )
        self.coverage_count += int(target.size)

    def compute(self) -> dict[str, float]:
        epsilon = np.finfo(np.float64).eps
        absolute_target = max(self.absolute_target, epsilon)
        absolute_sum_target = max(self.absolute_sum_target, epsilon)
        paper_crps_sum = float(
            np.mean(self.quantile_losses / absolute_sum_target)
        )
        return {
            # Table 2 uses this 19-quantile approximation via GluonTS.
            "CRPS_sum": paper_crps_sum,
            "empirical_CRPS_sum": self.sum_crps / absolute_sum_target,
            "empirical_CRPS": self.dimension_crps / absolute_target,
            "CRPS_sum_unnormalized_mean": self.sum_crps
            / max(self.sum_time_count, 1),
            "ND": self.absolute_error / absolute_target,
            "NRMSE": (
                np.sqrt(self.squared_error / max(self.element_count, 1))
                / (absolute_target / max(self.element_count, 1))
            ),
            "coverage_50": self.coverage_50_hits / max(self.coverage_count, 1),
            "coverage_90": self.coverage_90_hits / max(self.coverage_count, 1),
        }


def evaluate_samples(samples: np.ndarray, target: np.ndarray) -> dict[str, float]:
    accumulator = MetricAccumulator()
    accumulator.update(samples, target)
    return accumulator.compute()
