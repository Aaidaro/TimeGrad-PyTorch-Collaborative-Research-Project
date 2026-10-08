"""Benchmark acquisition, multivariate grouping, caching, and windows."""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .benchmarks import BenchmarkSpec, get_benchmark_spec

LOGGER = logging.getLogger(__name__)

# GluonTS 0.16 still represents the Exchange benchmark with ``Period[B]``.
# Pandas warns about that legacy representation even though it remains the
# dataset's declared frequency. Limit suppression to those two upstream
# deprecation messages; all other warnings remain visible.
warnings.filterwarnings(
    "ignore",
    message="Period with BDay freq is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"PeriodDtype\[B\] is deprecated.*",
    category=FutureWarning,
)


@dataclass(frozen=True)
class ForecastWindow:
    """One rolling forecast origin with only the history needed by TimeGrad."""

    history: np.ndarray  # (history_length, target_dim)
    target: np.ndarray  # (prediction_length, target_dim)
    history_start: pd.Period


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    train_target: np.ndarray  # (time, target_dim)
    train_start: pd.Period
    test_windows: tuple[ForecastWindow, ...]
    freq: str
    prediction_length: int
    full_target_dim: int
    benchmark: str | None = None

    @property
    def target_dim(self) -> int:
        return int(self.train_target.shape[1])


def _to_period(value: Any, freq: str) -> pd.Period:
    if isinstance(value, pd.Period):
        # ``Period.asfreq`` defaults to the end of the source interval and can
        # turn 00:00 at 30-minute frequency into 00:29. Alignment must retain
        # the timestamp at which the observation interval starts.
        return pd.Period(value.start_time, freq=freq)
    return pd.Period(value, freq=freq)


def _entry_series(entry: Mapping[str, Any], freq: str) -> pd.Series:
    values = np.asarray(entry["target"], dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"expected a univariate target, got shape {values.shape}")
    start = _to_period(entry["start"], freq)
    index = pd.period_range(start=start, periods=len(values), freq=freq)
    return pd.Series(values, index=index)


def group_training_entries(
    entries: Sequence[Mapping[str, Any]],
    freq: str,
    max_target_dim: int | None = None,
) -> tuple[np.ndarray, pd.Period]:
    """Align univariate entries and return a time-major multivariate array.

    This mirrors GluonTS ``MultivariateGrouper`` training behavior. Restricting
    dimensionality intentionally retains the *last* dimensions, which is an
    easy-to-miss detail of the reference benchmark code.
    """

    if not entries:
        raise ValueError("training dataset is empty")
    selected = list(entries[-max_target_dim:]) if max_target_dim else list(entries)
    series = [_entry_series(entry, freq) for entry in selected]
    first_timestamp = min(item.index[0] for item in series)
    last_timestamp = max(item.index[-1] for item in series)
    common_index = pd.period_range(first_timestamp, last_timestamp, freq=freq)

    aligned = []
    for item in series:
        fill_value = float(item.mean()) if len(item) else 0.0
        aligned.append(item.reindex(common_index, fill_value=fill_value).to_numpy())
    dimension_major = np.vstack(aligned).astype(np.float32, copy=False)
    return np.ascontiguousarray(dimension_major.T), first_timestamp


def group_test_batch(
    entries: Sequence[Mapping[str, Any]],
    freq: str,
    history_length: int,
    prediction_length: int,
) -> ForecastWindow:
    """Group one forecast date and retain exactly ``history + horizon``.

    GluonTS stores these benchmarks as one univariate entry per dimension and
    forecast date. Keeping every growing prefix wastes memory; slicing here is
    equivalent for an autoregressive model with finite lags.
    """

    if not entries:
        raise ValueError("test batch is empty")
    series = [_entry_series(entry, freq) for entry in entries]
    first_timestamp = min(item.index[0] for item in series)
    last_timestamp = max(item.index[-1] for item in series)
    common_index = pd.period_range(first_timestamp, last_timestamp, freq=freq)
    required = history_length + prediction_length
    if len(common_index) < required:
        raise ValueError(
            f"test prefix has {len(common_index)} points but {required} are required"
        )

    aligned = [
        item.reindex(common_index, fill_value=0.0).to_numpy(dtype=np.float32)
        for item in series
    ]
    target = np.vstack(aligned)[:, -required:].T
    history_start = common_index[-required]
    return ForecastWindow(
        history=np.ascontiguousarray(target[:history_length]),
        target=np.ascontiguousarray(target[history_length:]),
        history_start=history_start,
    )


def fourier_time_features(
    start: pd.Period | pd.Timestamp, length: int, freq: str
) -> np.ndarray:
    """Reference-style Fourier calendar covariates in time-major layout."""

    offset_name = pd.tseries.frequencies.to_offset(freq).name
    if offset_name == "B":
        # Pandas is deprecating PeriodIndex specifically for business-day
        # frequency. DatetimeIndex exposes the identical calendar attributes
        # without altering the benchmark's declared frequency.
        timestamp = start.start_time if isinstance(start, pd.Period) else start
        index = pd.date_range(start=timestamp, periods=length, freq=freq)
    else:
        index = pd.period_range(start=start, periods=length, freq=freq)
    if offset_name in {"min", "T"}:
        attributes = ("minute", "hour", "dayofweek")
    elif offset_name in {"h", "H"}:
        attributes = ("hour", "dayofweek")
    elif offset_name == "D":
        attributes = ("dayofweek",)
    elif offset_name == "B":
        attributes = ("dayofweek", "dayofyear")
    else:
        raise ValueError(f"unsupported frequency for Fourier features: {freq}")

    # ``FourierDateFeatures`` in the archived implementation obtains each
    # period as ``max(values) + 1``. Its transform sees the complete series,
    # so hour/day-of-week always resolve to 24/7. Our rolling-window cache only
    # retains a short tail; deriving the period from that tail would make the
    # same timestamp encode differently at train and test time. Keep the
    # full-series periods explicit. The minute quirk remains reference-faithful:
    # 30-minute data observes {0, 30}, hence a period of 31 rather than 60.
    periods = {"hour": 24.0, "dayofweek": 7.0, "dayofyear": 367.0}
    if "minute" in attributes:
        offset = pd.tseries.frequencies.to_offset(freq)
        step_minutes = int(pd.Timedelta(offset).total_seconds() // 60)
        if step_minutes < 1 or 60 % step_minutes:
            raise ValueError(f"unsupported minute frequency: {freq}")
        periods["minute"] = float(max(range(0, 60, step_minutes)) + 1)

    columns: list[np.ndarray] = []
    for attribute in attributes:
        values = np.asarray(getattr(index, attribute), dtype=np.float32)
        period = periods[attribute]
        angles = values * (2.0 * np.pi / period)
        columns.extend((np.cos(angles), np.sin(angles)))
    return np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.float32)


class SlidingWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """All valid random-window candidates from a contiguous training prefix."""

    def __init__(
        self,
        target: np.ndarray,
        start: pd.Period,
        freq: str,
        history_length: int,
        context_length: int,
        prediction_length: int,
    ) -> None:
        super().__init__()
        target = np.asarray(target, dtype=np.float32)
        if target.ndim != 2:
            raise ValueError("target must use (time, dimension) layout")
        required = history_length + prediction_length
        if len(target) < required:
            raise ValueError(
                f"series length {len(target)} is shorter than required {required}"
            )
        self.target = np.ascontiguousarray(target)
        self.start = start
        self.freq = freq
        self.history_length = history_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.features = fourier_time_features(start, len(target), freq)
        self.num_windows = len(target) - required + 1

    @property
    def time_feature_dim(self) -> int:
        return int(self.features.shape[1])

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += self.num_windows
        if index < 0 or index >= self.num_windows:
            raise IndexError(index)
        history_end = index + self.history_length
        future_end = history_end + self.prediction_length
        target_start = history_end - self.context_length
        return {
            "history": torch.from_numpy(self.target[index:history_end]),
            "future": torch.from_numpy(self.target[history_end:future_end]),
            "time_features": torch.from_numpy(
                self.features[target_start:future_end]
            ),
        }


# def make_validation_windows(
#     bundle: DatasetBundle,
#     count: int,
#     history_length: int,
# ) -> tuple[np.ndarray, tuple[ForecastWindow, ...]]:
#     """Back-test split used for early stopping in the paper."""

#     if count < 1:
#         return bundle.train_target, tuple()
#     horizon = bundle.prediction_length
#     first_forecast = len(bundle.train_target) - count * horizon
#     if first_forecast < history_length:
#         raise ValueError(
#             "validation split leaves insufficient history; reduce validation.windows"
#         )

#     windows: list[ForecastWindow] = []
#     for window_index in range(count):
#         forecast_start = first_forecast + window_index * horizon
#         history_start_index = forecast_start - history_length
#         history_start = bundle.train_start + history_start_index
#         windows.append(
#             ForecastWindow(
#                 history=np.ascontiguousarray(
#                     bundle.train_target[history_start_index:forecast_start]
#                 ),
#                 target=np.ascontiguousarray(
#                     bundle.train_target[forecast_start : forecast_start + horizon]
#                 ),
#                 history_start=history_start,
#             )
#         )
#     return bundle.train_target[:first_forecast], tuple(windows)
def make_validation_windows(
    bundle: DatasetBundle,
    count: int,
    history_length: int,
    stride: int | None = None,
) -> tuple[np.ndarray, tuple[ForecastWindow, ...]]:
    """Create chronological rolling validation forecasts.

    stride=None preserves the original behavior:
    forecast origins are prediction_length steps apart.

    A smaller stride creates overlapping rolling validation forecasts
    while reserving a shorter unique tail of the training series.
    """

    if count < 1:
        return bundle.train_target, tuple()

    horizon = bundle.prediction_length

    # Preserve existing behavior unless explicitly overridden.
    stride = horizon if stride is None else int(stride)

    if stride < 1:
        raise ValueError("validation.stride must be at least 1")

    # Number of UNIQUE time points that must be held out.
    #
    # Example Taxi:
    # horizon = 24
    # count   = 56
    # stride  = 1
    #
    # span = 24 + 55 = 79
    validation_span = horizon + (count - 1) * stride

    first_forecast = len(bundle.train_target) - validation_span

    # The remaining training prefix must itself contain at least
    # one complete training sample.
    required_training_points = history_length + horizon

    if first_forecast < required_training_points:
        raise ValueError(
            "validation split leaves insufficient data for one training window; "
            "reduce validation.windows or validation.stride"
        )

    windows: list[ForecastWindow] = []

    for window_index in range(count):
        forecast_start = first_forecast + window_index * stride
        history_start_index = forecast_start - history_length
        history_start = bundle.train_start + history_start_index

        windows.append(
            ForecastWindow(
                history=np.ascontiguousarray(
                    bundle.train_target[
                        history_start_index:forecast_start
                    ]
                ),
                target=np.ascontiguousarray(
                    bundle.train_target[
                        forecast_start:forecast_start + horizon
                    ]
                ),
                history_start=history_start,
            )
        )

    return bundle.train_target[:first_forecast], tuple(windows)


def window_tensors(
    window: ForecastWindow,
    freq: str,
    context_length: int,
) -> dict[str, torch.Tensor]:
    history_length = len(window.history)
    all_features = fourier_time_features(
        window.history_start,
        history_length + len(window.target),
        freq,
    )
    return {
        "history": torch.from_numpy(window.history).unsqueeze(0),
        "target": torch.from_numpy(window.target).unsqueeze(0),
        "context_time_features": torch.from_numpy(
            all_features[history_length - context_length : history_length]
        ).unsqueeze(0),
        "future_time_features": torch.from_numpy(
            all_features[history_length:]
        ).unsqueeze(0),
    }


def _cache_path(config: Mapping[str, Any], history_length: int) -> Path:
    root = Path(config.get("root", "data/dataset"))
    selected_dim = config.get("max_target_dim") or "all"
    spec = get_benchmark_spec(str(config.get("benchmark") or config["name"]))
    dataset_name = spec.gluonts_name if spec else str(config["name"])
    return root / f"{dataset_name}_d{selected_dim}_h{history_length}.npz"


def _save_cache(path: Path, bundle: DatasetBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    histories = np.stack([item.history for item in bundle.test_windows])
    targets = np.stack([item.target for item in bundle.test_windows])
    metadata = {
        "name": bundle.name,
        "train_start": str(bundle.train_start),
        "test_starts": [str(item.history_start) for item in bundle.test_windows],
        "freq": bundle.freq,
        "prediction_length": bundle.prediction_length,
        "full_target_dim": bundle.full_target_dim,
        "benchmark": bundle.benchmark,
    }
    np.savez_compressed(
        path,
        train_target=bundle.train_target,
        test_histories=histories,
        test_targets=targets,
        metadata=np.asarray(json.dumps(metadata)),
    )


def _load_cache(path: Path) -> DatasetBundle:
    with np.load(path, allow_pickle=False) as cached:
        metadata = json.loads(str(cached["metadata"]))
        histories = cached["test_histories"].astype(np.float32, copy=False)
        targets = cached["test_targets"].astype(np.float32, copy=False)
        starts = metadata["test_starts"]
        windows = tuple(
            ForecastWindow(
                history=np.ascontiguousarray(histories[index]),
                target=np.ascontiguousarray(targets[index]),
                history_start=pd.Period(starts[index], freq=metadata["freq"]),
            )
            for index in range(len(histories))
        )
        return DatasetBundle(
            name=metadata["name"],
            train_target=np.ascontiguousarray(
                cached["train_target"].astype(np.float32, copy=False)
            ),
            train_start=pd.Period(metadata["train_start"], freq=metadata["freq"]),
            test_windows=windows,
            freq=metadata["freq"],
            prediction_length=int(metadata["prediction_length"]),
            full_target_dim=int(metadata["full_target_dim"]),
            benchmark=metadata.get("benchmark"),
        )


def _frequency_signature(freq: str) -> tuple[str, int]:
    offset = pd.tseries.frequencies.to_offset(freq)
    return offset.name, offset.n


def validate_benchmark_bundle(
    bundle: DatasetBundle,
    spec: BenchmarkSpec,
    strict: bool,
) -> None:
    """Check a grouped benchmark against the paper's published data shape."""

    expected = {
        "target dimensions": (bundle.full_target_dim, spec.target_dim),
        "training steps": (len(bundle.train_target), spec.train_length),
        "prediction length": (bundle.prediction_length, spec.prediction_length),
        "rolling windows": (len(bundle.test_windows), spec.rolling_windows),
        "frequency": (
            _frequency_signature(bundle.freq),
            _frequency_signature(spec.frequency),
        ),
    }
    problems = [
        f"{name}: observed {observed}, expected {wanted}"
        for name, (observed, wanted) in expected.items()
        if observed != wanted
    ]
    if problems and strict:
        raise ValueError(
            f"{spec.key} data does not match the paper: " + "; ".join(problems)
        )
    for problem in problems:
        LOGGER.warning("%s paper-shape check: %s", spec.key, problem)

    arrays = [bundle.train_target]
    arrays.extend(window.history for window in bundle.test_windows)
    arrays.extend(window.target for window in bundle.test_windows)
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError(f"{spec.key} target contains NaN or infinite values")
    minimum = min(float(array.min()) for array in arrays)
    maximum = max(float(array.max()) for array in arrays)
    if spec.domain in {"nonnegative_real", "nonnegative_count", "unit_interval"}:
        if minimum < -1e-7:
            raise ValueError(f"{spec.key} target must be nonnegative")
    if spec.domain == "unit_interval" and maximum > 1.0 + 1e-7:
        raise ValueError("traffic target must remain inside the unit interval")


def load_gluonts_dataset(
    config: Mapping[str, Any],
    history_length: int,
) -> DatasetBundle:
    requested_name = str(config.get("benchmark") or config["name"])
    spec = get_benchmark_spec(requested_name)
    cache_path = _cache_path(config, history_length)
    if cache_path.exists() and not config.get("regenerate", False):
        LOGGER.info("Loading grouped dataset cache from %s", cache_path)
        bundle = _load_cache(cache_path)
        needs_metadata_migration = bool(spec and bundle.benchmark != spec.key)
        if spec and needs_metadata_migration:
            # Migrate caches created before benchmark identities were stored.
            # The cache filename and the strict shape checks below still guard
            # against accidentally relabeling a different dataset.
            bundle = replace(
                bundle,
                name=spec.gluonts_name,
                benchmark=spec.key,
            )
        cached_spec = spec or get_benchmark_spec(bundle.benchmark or bundle.name)
        if cached_spec:
            validate_benchmark_bundle(
                bundle, cached_spec, strict=bool(config.get("strict_paper_shape", True))
            )
        if needs_metadata_migration:
            _save_cache(cache_path, bundle)
        return bundle

    try:
        from gluonts.dataset.repository.datasets import get_dataset
    except ImportError as error:
        raise RuntimeError(
            "GluonTS is required to download repository datasets; "
            "install requirements.txt"
        ) from error

    name = spec.gluonts_name if spec else str(config["name"])
    root = Path(config.get("root", "data/dataset")) / "gluonts"
    raw = get_dataset(name, path=root, regenerate=bool(config.get("regenerate", False)))
    freq = str(raw.metadata.freq)
    prediction_length = int(
        config.get("prediction_length", raw.metadata.prediction_length)
    )
    if prediction_length != int(raw.metadata.prediction_length):
        LOGGER.warning(
            "Overriding GluonTS prediction length %s with %s",
            raw.metadata.prediction_length,
            prediction_length,
        )

    train_entries = list(raw.train)
    full_target_dim = len(train_entries)
    max_target_dim = config.get("max_target_dim")
    max_target_dim = int(max_target_dim) if max_target_dim is not None else None
    if max_target_dim is not None and not 1 <= max_target_dim <= full_target_dim:
        raise ValueError(
            f"max_target_dim must lie in [1, {full_target_dim}], got {max_target_dim}"
        )
    train_target, train_start = group_training_entries(
        train_entries, freq=freq, max_target_dim=max_target_dim
    )

    num_test_entries = len(raw.test)
    if num_test_entries % full_target_dim:
        raise ValueError(
            "test entry count is not divisible by training target dimensionality"
        )
    num_test_dates = num_test_entries // full_target_dim
    selected_start = full_target_dim - (max_target_dim or full_target_dim)
    test_windows: list[ForecastWindow] = []
    selected_batch: list[Mapping[str, Any]] = []
    for index, entry in enumerate(raw.test):
        dimension_index = index % full_target_dim
        if dimension_index >= selected_start:
            selected_batch.append(entry)
        if dimension_index == full_target_dim - 1:
            test_windows.append(
                group_test_batch(
                    selected_batch,
                    freq=freq,
                    history_length=history_length,
                    prediction_length=prediction_length,
                )
            )
            selected_batch = []
    if len(test_windows) != num_test_dates:
        raise RuntimeError("internal error while grouping rolling test dates")

    bundle = DatasetBundle(
        name=name,
        train_target=train_target,
        train_start=train_start,
        test_windows=tuple(test_windows),
        freq=freq,
        prediction_length=prediction_length,
        full_target_dim=full_target_dim,
        benchmark=spec.key if spec else None,
    )
    if spec:
        validate_benchmark_bundle(
            bundle, spec, strict=bool(config.get("strict_paper_shape", True))
        )
    _save_cache(cache_path, bundle)
    LOGGER.info(
        "Prepared %s: train=%s, test_windows=%d",
        name,
        bundle.train_target.shape,
        len(bundle.test_windows),
    )
    return bundle


def load_synthetic_dataset(
    config: Mapping[str, Any],
    history_length: int,
    seed: int,
) -> DatasetBundle:
    """Small correlated count benchmark used only for executable verification."""

    rng = np.random.default_rng(seed)
    target_dim = int(config.get("target_dim", 16))
    train_length = int(config.get("train_length", 240))
    prediction_length = int(config.get("prediction_length", 24))
    num_test_windows = int(config.get("test_windows", 3))
    total_length = train_length + num_test_windows * prediction_length
    time = np.arange(total_length, dtype=np.float32)

    common = (
        1.2 * np.sin(2.0 * np.pi * time / 48.0)
        + 0.5 * np.cos(2.0 * np.pi * time / 336.0)
        + 0.25 * np.sin(2.0 * np.pi * time / 12.0)
    )
    loadings = rng.uniform(0.6, 1.4, size=(target_dim,)).astype(np.float32)
    offsets = rng.uniform(2.0, 8.0, size=(target_dim,)).astype(np.float32)
    rates = offsets[None, :] + common[:, None] * loadings[None, :]
    rates += 0.15 * rng.normal(size=rates.shape)
    target = rng.poisson(np.clip(rates, 0.1, None)).astype(np.float32)

    start = pd.Period("2015-01-01 00:00", freq="30min")
    windows: list[ForecastWindow] = []
    for index in range(num_test_windows):
        forecast_start = train_length + index * prediction_length
        history_start_index = forecast_start - history_length
        windows.append(
            ForecastWindow(
                history=np.ascontiguousarray(
                    target[history_start_index:forecast_start]
                ),
                target=np.ascontiguousarray(
                    target[forecast_start : forecast_start + prediction_length]
                ),
                history_start=start + history_start_index,
            )
        )
    return DatasetBundle(
        name="synthetic_correlated_counts",
        train_target=np.ascontiguousarray(target[:train_length]),
        train_start=start,
        test_windows=tuple(windows),
        freq="30min",
        prediction_length=prediction_length,
        full_target_dim=target_dim,
        benchmark=None,
    )


def load_dataset(
    config: Mapping[str, Any],
    history_length: int,
    seed: int,
) -> DatasetBundle:
    source = str(config.get("source", "gluonts"))
    if source == "gluonts":
        return load_gluonts_dataset(config, history_length)
    if source == "synthetic":
        return load_synthetic_dataset(config, history_length, seed)
    raise ValueError(f"unsupported dataset source: {source}")
