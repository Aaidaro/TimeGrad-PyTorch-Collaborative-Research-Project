"""Command-line entry point for download, training, and evaluation."""

from __future__ import annotations

import argparse
import logging
import platform
import time
from pathlib import Path
from typing import Any

import torch

from data import get_benchmark_spec, load_dataset
from data.data_loader import fourier_time_features
from scripts.evaluate import evaluate_model
from scripts.train import load_trained_model, train_experiment
from utils import (
    apply_overrides,
    choose_device,
    configure_logging,
    load_config,
    resolve_project_path,
    seed_everything,
    write_json,
)
from utils.visualization import plot_training_history

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the TimeGrad paper")
    parser.add_argument(
        "--config", default="config/exchange.yaml", help="YAML experiment config"
    )
    parser.add_argument(
        "--mode",
        choices=("download", "train", "evaluate", "all"),
        default="all",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="override a value, e.g. --set training.epochs=2",
    )
    return parser.parse_args()


def _absolute_paths(config: dict[str, Any]) -> dict[str, Any]:
    config["dataset"]["root"] = str(
        resolve_project_path(config, config["dataset"].get("root", "data/dataset"))
    )
    return config


def run(config: dict[str, Any], mode: str) -> dict[str, Any]:
    configure_logging(config)
    seed = int(config["experiment"]["seed"])
    seed_everything(
        seed,
        deterministic=bool(config["experiment"].get("deterministic", True)),
    )
    device = choose_device(str(config["experiment"].get("device", "auto")))
    context_length = int(config["model"]["context_length"])
    lags = [int(value) for value in config["model"]["lags"]]
    history_length = context_length + max(lags)

    LOGGER.info("Loading dataset on device=%s", device)
    dataset = load_dataset(config["dataset"], history_length=history_length, seed=seed)
    spec = get_benchmark_spec(dataset.benchmark or dataset.name)
    time_feature_dim = fourier_time_features(
        dataset.train_start,
        history_length + dataset.prediction_length,
        dataset.freq,
    ).shape[1]

    if mode == "download":
        return {
            "status": "downloaded",
            "dataset": dataset.name,
            "benchmark": dataset.benchmark,
            "train_shape": list(dataset.train_target.shape),
            "full_target_dim": dataset.full_target_dim,
            "frequency": dataset.freq,
            "prediction_length": dataset.prediction_length,
            "rolling_windows": len(dataset.test_windows),
            "paper_reference": spec.paper_reference() if spec else None,
        }

    checkpoint_path = resolve_project_path(
        config, config["outputs"]["checkpoint"]
    )
    history_path = resolve_project_path(config, config["outputs"]["history_csv"])
    loss_plot_path = resolve_project_path(config, config["outputs"]["loss_plot"])
    evaluation_path = resolve_project_path(
        config, config["outputs"]["evaluation_report"]
    )
    forecast_plot_path = resolve_project_path(
        config, config["outputs"]["forecast_plot"]
    )

    start_time = time.perf_counter()
    training_summary: dict[str, Any] | None = None
    if mode in {"train", "all"}:
        training = train_experiment(
            dataset=dataset,
            config=config,
            time_feature_dim=int(time_feature_dim),
            device=device,
            checkpoint_path=checkpoint_path,
            history_path=history_path,
        )
        model = training.model
        plot_training_history(training.history, loss_plot_path)
        training_summary = {
            "best_epoch": training.best_epoch,
            "loss_improved_over_run": training.improved,
            "epochs_executed": len(training.history),
            "first_train_loss": float(training.history[0]["train_loss"]),
            "last_train_loss": float(training.history[-1]["train_loss"]),
        }
    else:
        model, checkpoint = load_trained_model(checkpoint_path, device)
        training_summary = {
            "loaded_checkpoint_epoch": int(checkpoint["epoch"]),
            "loaded_checkpoint_phase": checkpoint["phase"],
        }

    evaluation_summary = None
    if mode in {"evaluate", "all"}:
        evaluation_summary = evaluate_model(
            model=model,
            dataset=dataset,
            config=config,
            device=device,
            report_path=evaluation_path,
            plot_path=forecast_plot_path,
        )

    summary = {
        "status": "completed",
        "mode": mode,
        "experiment": config["experiment"]["name"],
        "seed": seed,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "dataset": {
            "name": dataset.name,
            "benchmark": dataset.benchmark,
            "train_shape": list(dataset.train_target.shape),
            "full_target_dim": dataset.full_target_dim,
            "frequency": dataset.freq,
            "prediction_length": dataset.prediction_length,
            "rolling_windows_available": len(dataset.test_windows),
        },
        "model_parameters": model.parameter_count(),
        "training": training_summary,
        "evaluation": evaluation_summary,
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    summary_path = resolve_project_path(config, config["outputs"]["run_summary"])
    write_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config = load_config(config_path)
    config = apply_overrides(config, args.overrides)
    config = _absolute_paths(config)
    summary = run(config, args.mode)
    print(f"Completed: {summary}")


if __name__ == "__main__":
    main()
