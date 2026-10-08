"""Rolling-window sampling, metrics, reports, and forecast plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from data import DatasetBundle, get_benchmark_spec, window_tensors
from models import TimeGrad
from utils.config import write_json
from utils.metrics import MetricAccumulator
from utils.visualization import plot_forecast

LOGGER = logging.getLogger(__name__)


def _paper_protocol_checks(
    model: TimeGrad,
    dataset: DatasetBundle,
    config: Mapping[str, Any],
    windows_evaluated: int,
    num_samples: int,
) -> dict[str, bool]:
    """Return auditable checks for every published, testable paper setting."""

    spec = get_benchmark_spec(dataset.benchmark or dataset.name)
    if spec is None:
        return {}
    training = config["training"]
    validation = config["validation"]
    model_config = model.config
    return {
        "full_target_dimension": dataset.target_dim == spec.target_dim,
        "all_rolling_windows": windows_evaluated == spec.rolling_windows,
        "one_hundred_samples": num_samples == 100,
        "context_equals_horizon": (
            model_config.context_length == spec.prediction_length
        ),
        "reference_lags": model_config.lags == spec.lags,
        "reference_scaling": model_config.scaling == spec.scaling,
        "diffusion_schedule": (
            model_config.diffusion_steps == 100
            and model_config.beta_start == 1e-4
            and model_config.beta_end == 0.1
        ),
        "lstm_architecture": (
            model_config.rnn_layers == 2 and model_config.rnn_hidden_size == 40
        ),
        "noise_embedding": (
            model_config.diffusion_embedding_dim == 16
            and model_config.diffusion_projection_dim == 64
        ),
        "wavenet_architecture": (
            model_config.residual_layers == 8
            and model_config.residual_channels == 8
            and model_config.dilation_cycle_length == 2
        ),
        "optimizer_settings": (
            int(training["batch_size"]) == 64
            and float(training["learning_rate"]) == 1e-3
        ),
        "validation_interval": (
            int(validation["windows"]) == spec.rolling_windows
        ),
    }


@torch.no_grad()
def evaluate_model(
    model: TimeGrad,
    dataset: DatasetBundle,
    config: Mapping[str, Any],
    device: torch.device,
    report_path: Path,
    plot_path: Path,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    requested_windows = evaluation.get("max_windows")
    windows = dataset.test_windows
    if requested_windows is not None:
        windows = windows[: int(requested_windows)]
    num_samples = int(evaluation.get("num_samples", 100))
    nonnegative = bool(evaluation.get("clip_nonnegative", False))
    generator = torch.Generator(device=device).manual_seed(
        int(config["experiment"]["seed"]) + 20_000
    )

    accumulator = MetricAccumulator()
    first_samples: np.ndarray | None = None
    for index, window in enumerate(windows):
        tensors = window_tensors(window, dataset.freq, model.config.context_length)
        samples = model.forecast(
            history=tensors["history"].to(device),
            context_time_features=tensors["context_time_features"].to(device),
            future_time_features=tensors["future_time_features"].to(device),
            num_samples=num_samples,
            nonnegative=nonnegative,
            generator=generator,
        )[0].cpu().numpy()
        accumulator.update(samples, window.target)
        if first_samples is None:
            first_samples = samples
        LOGGER.info("Evaluated rolling window %d/%d", index + 1, len(windows))

    if not windows or first_samples is None:
        raise ValueError("no test windows were selected for evaluation")
    metrics = accumulator.compute()
    spec = get_benchmark_spec(dataset.benchmark or dataset.name)
    result: dict[str, Any] = {
        "dataset": dataset.name,
        "benchmark": dataset.benchmark,
        "target_dim": dataset.target_dim,
        "full_target_dim": dataset.full_target_dim,
        "prediction_length": dataset.prediction_length,
        "rolling_windows_evaluated": len(windows),
        "num_samples": num_samples,
        "clip_nonnegative": nonnegative,
        "scaling": model.config.scaling,
        "metrics": metrics,
    }
    if spec:
        protocol_checks = _paper_protocol_checks(
            model=model,
            dataset=dataset,
            config=config,
            windows_evaluated=len(windows),
            num_samples=num_samples,
        )
        result["paper_reference"] = spec.paper_reference()
        result["paper_protocol_checks"] = protocol_checks
        result["paper_protocol_complete"] = all(protocol_checks.values())
    write_json(report_path, result)
    plot_forecast(
        history=windows[0].history,
        target=windows[0].target,
        samples=first_samples,
        output_path=plot_path,
        max_dimensions=int(evaluation.get("plot_dimensions", 6)),
    )
    LOGGER.info("Evaluation metrics: %s", metrics)
    return result
