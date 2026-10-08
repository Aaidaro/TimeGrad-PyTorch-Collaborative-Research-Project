"""Run and aggregate a paper benchmark's ten-seed protocol."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from data import BenchmarkSpec, require_benchmark_spec
from scripts.main import run
from utils import load_config, resolve_project_path, write_json


def _seed_config(
    base: dict[str, Any], seed: int, spec: BenchmarkSpec
) -> dict[str, Any]:
    config = deepcopy(base)
    config["experiment"]["seed"] = seed
    config["experiment"]["name"] = f"timegrad_{spec.key}_paper_seed_{seed}"
    report_root = (
        Path("outputs/reports/paper_runs") / spec.key / f"seed_{seed}"
    )
    plot_root = Path("outputs/plots/paper_runs") / spec.key / f"seed_{seed}"
    config["outputs"].update(
        {
            "checkpoint": (
                f"models/saved_models/paper_runs/{spec.key}/seed_{seed}.pt"
            ),
            "history_csv": str(report_root / "training_history.csv"),
            "evaluation_report": str(report_root / "evaluation.json"),
            "run_summary": str(report_root / "run_summary.json"),
            "loss_plot": str(plot_root / "training_loss.png"),
            "forecast_plot": str(plot_root / "forecast.png"),
        }
    )
    config["dataset"]["root"] = str(
        resolve_project_path(config, config["dataset"].get("root", "data/dataset"))
    )
    return config


def _aggregate(
    results: list[dict[str, Any]],
    seeds: list[int],
    spec: BenchmarkSpec,
) -> dict[str, Any]:
    values = np.asarray(
        [result["evaluation"]["metrics"]["CRPS_sum"] for result in results],
        dtype=np.float64,
    )
    standard_error = (
        float(values.std(ddof=1) / np.sqrt(len(values)))
        if len(values) > 1
        else 0.0
    )
    return {
        "benchmark": spec.key,
        "runs": len(values),
        "seeds": seeds,
        "CRPS_sum_mean": float(values.mean()),
        "CRPS_sum_standard_error": standard_error,
        "CRPS_sum_values": values.tolist(),
        "paper_reference": spec.paper_reference(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and aggregate independent TimeGrad runs"
    )
    parser.add_argument("--config", default="config/exchange.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(0,30,10))) # seeds: [0, 10, 20]
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    base = load_config(config_path)
    benchmark_name = str(
        base["dataset"].get("benchmark") or base["dataset"]["name"]
    )
    spec = require_benchmark_spec(benchmark_name)
    seeds = list(dict.fromkeys(args.seeds))
    if not seeds:
        raise ValueError("at least one seed is required")

    results = [
        run(_seed_config(base, seed, spec), mode="all") for seed in seeds
    ]
    aggregate = _aggregate(results, seeds, spec)
    summary = args.summary or (
        f"outputs/reports/paper_runs/{spec.key}/aggregate.json"
    )
    write_json(resolve_project_path(base, summary), aggregate)
    print(aggregate)


if __name__ == "__main__":
    main()
