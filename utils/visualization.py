"""Non-interactive plots for training diagnostics and probabilistic forecasts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


def plot_training_history(
    history: Sequence[Mapping[str, float]], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(item["epoch"]) for item in history]
    train_loss = [float(item["train_loss"]) for item in history]
    validation_loss = [float(item.get("validation_loss", np.nan)) for item in history]

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(epochs, train_loss, marker="o", label="train denoising loss")
    if np.isfinite(validation_loss).any():
        axis.plot(epochs, validation_loss, marker="s", label="validation loss")
    axis.set(xlabel="epoch", ylabel="MSE", title="TimeGrad training history")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_forecast(
    history: np.ndarray,
    target: np.ndarray,
    samples: np.ndarray,
    output_path: Path,
    max_dimensions: int = 6,
) -> None:
    """Plot median, 50%, and 90% intervals for selected dimensions."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = np.asarray(history)
    target = np.asarray(target)
    samples = np.asarray(samples)
    dimensions = min(max_dimensions, target.shape[-1])
    context = min(3 * target.shape[0], history.shape[0])
    history_x = np.arange(-context, 0)
    future_x = np.arange(target.shape[0])
    median = np.quantile(samples, 0.5, axis=0)
    lower_50, upper_50 = np.quantile(samples, [0.25, 0.75], axis=0)
    lower_90, upper_90 = np.quantile(samples, [0.05, 0.95], axis=0)

    figure, axes = plt.subplots(
        dimensions,
        1,
        figsize=(10.0, max(2.2 * dimensions, 3.4)),
        sharex=True,
        squeeze=False,
    )
    for dimension in range(dimensions):
        axis = axes[dimension, 0]
        axis.plot(
            history_x,
            history[-context:, dimension],
            color="#4C566A",
            linewidth=1.1,
            label="history" if dimension == 0 else None,
        )
        axis.plot(
            future_x,
            target[:, dimension],
            color="#2E3440",
            linewidth=1.3,
            label="target" if dimension == 0 else None,
        )
        axis.fill_between(
            future_x,
            lower_90[:, dimension],
            upper_90[:, dimension],
            color="#88C0D0",
            alpha=0.25,
            label="90% interval" if dimension == 0 else None,
        )
        axis.fill_between(
            future_x,
            lower_50[:, dimension],
            upper_50[:, dimension],
            color="#5E81AC",
            alpha=0.3,
            label="50% interval" if dimension == 0 else None,
        )
        axis.plot(
            future_x,
            median[:, dimension],
            color="#BF616A",
            linewidth=1.2,
            label="median" if dimension == 0 else None,
        )
        axis.axvline(-0.5, color="black", linewidth=0.7, alpha=0.5)
        axis.set_ylabel(f"dim {dimension}")
        axis.grid(alpha=0.18)

    axes[0, 0].legend(ncol=5, fontsize=8, loc="upper left")
    axes[-1, 0].set_xlabel("forecast step (history is negative)")
    figure.suptitle("TimeGrad probabilistic forecast", y=0.995)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
