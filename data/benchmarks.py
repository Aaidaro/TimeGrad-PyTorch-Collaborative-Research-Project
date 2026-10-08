"""Paper benchmark registry and GluonTS alias resolution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSpec:
    """Immutable metadata from Tables 1 and 2 of the TimeGrad paper."""

    key: str
    gluonts_name: str
    aliases: tuple[str, ...]
    target_dim: int
    train_length: int
    prediction_length: int
    rolling_windows: int
    frequency: str
    lags: tuple[int, ...]
    scaling: bool
    domain: str
    crps_sum_mean: float
    crps_sum_uncertainty: float

    def paper_reference(self) -> dict[str, float | int | str]:
        return {
            "CRPS_sum_mean": self.crps_sum_mean,
            "CRPS_sum_reported_uncertainty": self.crps_sum_uncertainty,
            "independent_runs": 10,
            "source": "TimeGrad paper Table 2",
        }


PAPER_BENCHMARKS: dict[str, BenchmarkSpec] = {
    "exchange": BenchmarkSpec(
        key="exchange",
        gluonts_name="exchange_rate_nips",
        aliases=("exchange", "exchange_rate_nips"),
        target_dim=8,
        train_length=6071,
        prediction_length=30,
        rolling_windows=5,
        frequency="B",
        lags=(1, 2),
        scaling=True,
        domain="nonnegative_real",
        crps_sum_mean=0.006,
        crps_sum_uncertainty=0.001,
    ),
    "solar": BenchmarkSpec(
        key="solar",
        gluonts_name="solar_nips",
        aliases=("solar", "solar_nips"),
        target_dim=137,
        train_length=7009,
        prediction_length=24,
        rolling_windows=7,
        frequency="h",
        lags=(1, 24, 168),
        scaling=True,
        domain="nonnegative_real",
        crps_sum_mean=0.287,
        crps_sum_uncertainty=0.020,
    ),
    "electricity": BenchmarkSpec(
        key="electricity",
        gluonts_name="electricity_nips",
        aliases=("electricity", "electricity_nips"),
        target_dim=370,
        train_length=5833,
        prediction_length=24,
        rolling_windows=7,
        frequency="h",
        lags=(1, 24, 168),
        scaling=True,
        domain="nonnegative_real",
        crps_sum_mean=0.0206,
        crps_sum_uncertainty=0.001,
    ),
    "traffic": BenchmarkSpec(
        key="traffic",
        gluonts_name="traffic_nips",
        aliases=("traffic", "traffic_nips"),
        target_dim=963,
        train_length=4001,
        prediction_length=24,
        rolling_windows=7,
        frequency="h",
        lags=(1, 24, 168),
        scaling=False,
        domain="unit_interval",
        crps_sum_mean=0.044,
        crps_sum_uncertainty=0.006,
    ),
    "taxi": BenchmarkSpec(
        key="taxi",
        gluonts_name="taxi_30min",
        aliases=("taxi", "taxi_30min"),
        target_dim=1214,
        train_length=1488,
        prediction_length=24,
        rolling_windows=56,
        frequency="30min",
        lags=(1, 4, 12, 24, 48),
        scaling=True,
        domain="nonnegative_count",
        crps_sum_mean=0.114,
        crps_sum_uncertainty=0.020,
    ),
}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


_ALIASES = {
    _normalize_name(alias): spec
    for spec in PAPER_BENCHMARKS.values()
    for alias in spec.aliases
}


def get_benchmark_spec(name: str | None) -> BenchmarkSpec | None:
    """Resolve a friendly or canonical name to a paper benchmark."""

    if name is None:
        return None
    return _ALIASES.get(_normalize_name(name))


def require_benchmark_spec(name: str) -> BenchmarkSpec:
    spec = get_benchmark_spec(name)
    if spec is None:
        supported = ", ".join(PAPER_BENCHMARKS)
        raise ValueError(
            f"unknown paper benchmark {name!r}; choose one of: {supported}"
        )
    return spec
