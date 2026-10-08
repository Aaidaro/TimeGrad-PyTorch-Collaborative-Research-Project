# """Training and checkpointing for a single TimeGrad experiment."""

# from __future__ import annotations

# import csv
# import logging
# from copy import deepcopy
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Mapping, Sequence

# import numpy as np
# import torch
# from torch.nn.utils import clip_grad_norm_
# from torch.utils.data import DataLoader, RandomSampler

# from data import (
#     DatasetBundle,
#     ForecastWindow,
#     SlidingWindowDataset,
#     make_validation_windows,
#     window_tensors,
# )
# from models import TimeGrad, TimeGradConfig

# LOGGER = logging.getLogger(__name__)


# @dataclass
# class TrainingResult:
#     model: TimeGrad
#     history: list[dict[str, float | int | str]]
#     best_epoch: int
#     checkpoint_path: Path
#     improved: bool


# def build_model(
#     config: Mapping[str, Any],
#     target_dim: int,
#     time_feature_dim: int,
#     prediction_length: int,
# ) -> TimeGrad:
#     model_values = deepcopy(dict(config["model"]))
#     model_values.update(
#         {
#             "target_dim": target_dim,
#             "time_feature_dim": time_feature_dim,
#             "prediction_length": prediction_length,
#         }
#     )
#     return TimeGrad(TimeGradConfig.from_dict(model_values))


# def _make_train_loader(
#     dataset: SlidingWindowDataset,
#     batch_size: int,
#     steps_per_epoch: int,
#     seed: int,
#     num_workers: int,
# ) -> DataLoader[dict[str, torch.Tensor]]:
#     generator = torch.Generator().manual_seed(seed)
#     sampler = RandomSampler(
#         dataset,
#         replacement=True,
#         num_samples=batch_size * steps_per_epoch,
#         generator=generator,
#     )
#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         sampler=sampler,
#         num_workers=num_workers,
#         pin_memory=torch.cuda.is_available(),
#         drop_last=True,
#     )


# def _validation_loss(
#     model: TimeGrad,
#     windows: Sequence[ForecastWindow],
#     freq: str,
#     device: torch.device,
#     seed: int,
# ) -> float:
#     if not windows:
#         return float("nan")
#     model.eval()
#     generator = torch.Generator(device=device).manual_seed(seed)
#     losses = []
#     with torch.no_grad():
#         for window in windows:
#             tensors = window_tensors(window, freq, model.config.context_length)
#             history = tensors["history"].to(device)
#             future = tensors["target"].to(device)
#             features = torch.cat(
#                 (
#                     tensors["context_time_features"],
#                     tensors["future_time_features"],
#                 ),
#                 dim=1,
#             ).to(device)
#             loss = model.training_loss(
#                 history=history,
#                 future=future,
#                 time_features=features,
#                 generator=generator,
#             )
#             losses.append(float(loss.detach().cpu()))
#     return float(np.mean(losses))


# def _save_checkpoint(
#     path: Path,
#     model: TimeGrad,
#     optimizer: torch.optim.Optimizer,
#     epoch: int,
#     history: Sequence[Mapping[str, Any]],
#     dataset: DatasetBundle,
#     phase: str,
# ) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     temporary = path.with_suffix(path.suffix + ".tmp")
#     torch.save(
#         {
#             "model_state": model.state_dict(),
#             "optimizer_state": optimizer.state_dict(),
#             "model_config": model.config.to_dict(),
#             "epoch": epoch,
#             "phase": phase,
#             "history": list(history),
#             "dataset": {
#                 "name": dataset.name,
#                 "benchmark": dataset.benchmark,
#                 "freq": dataset.freq,
#                 "prediction_length": dataset.prediction_length,
#                 "target_dim": dataset.target_dim,
#             },
#         },
#         temporary,
#     )
#     temporary.replace(path)


# def _fit(
#     model: TimeGrad,
#     train_target: np.ndarray,
#     train_start,
#     validation_windows: Sequence[ForecastWindow],
#     dataset: DatasetBundle,
#     config: Mapping[str, Any],
#     device: torch.device,
#     checkpoint_path: Path,
#     phase: str,
#     max_epochs: int,
# ) -> tuple[list[dict[str, float | int | str]], int, bool]:
#     training = config["training"]
#     window_dataset = SlidingWindowDataset(
#         target=train_target,
#         start=train_start,
#         freq=dataset.freq,
#         history_length=model.config.history_length,
#         context_length=model.config.context_length,
#         prediction_length=model.config.prediction_length,
#     )
#     loader = _make_train_loader(
#         dataset=window_dataset,
#         batch_size=int(training["batch_size"]),
#         steps_per_epoch=int(training["steps_per_epoch"]),
#         seed=int(config["experiment"]["seed"]),
#         num_workers=int(training.get("num_workers", 0)),
#     )

#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=float(training["learning_rate"]),
#         weight_decay=float(training.get("weight_decay", 0.0)),
#     )
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode="min",
#         factor=float(training.get("lr_plateau_factor", 0.5)),
#         patience=int(training.get("lr_plateau_patience", 1)),
#     )
#     amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
#     scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
#     gradient_clip = float(training.get("gradient_clip", 1.0))
#     patience = int(training.get("early_stopping_patience", max_epochs))
#     minimum_epochs = int(training.get("minimum_epochs", 2))
#     minimum_delta = float(training.get("minimum_delta", 0.0))

#     history: list[dict[str, float | int | str]] = []
#     best_metric = float("inf")
#     best_epoch = 0
#     epochs_without_improvement = 0
#     best_state: dict[str, torch.Tensor] | None = None

#     for epoch in range(1, max_epochs + 1):
#         model.train()
#         epoch_loss = 0.0
#         batches = 0
#         gradient_norm = 0.0
#         for batch in loader:
#             history_tensor = batch["history"].to(device, non_blocking=True)
#             future_tensor = batch["future"].to(device, non_blocking=True)
#             features_tensor = batch["time_features"].to(device, non_blocking=True)
#             optimizer.zero_grad(set_to_none=True)
#             with torch.autocast(
#                 device_type=device.type,
#                 dtype=torch.float16,
#                 enabled=amp_enabled,
#             ):
#                 loss = model.training_loss(
#                     history_tensor, future_tensor, features_tensor
#                 )
#             if not torch.isfinite(loss):
#                 raise FloatingPointError(f"non-finite training loss at epoch {epoch}")

#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             gradient_norm = float(clip_grad_norm_(model.parameters(), gradient_clip))
#             scaler.step(optimizer)
#             scaler.update()
#             epoch_loss += float(loss.detach().cpu())
#             batches += 1

#         train_loss = epoch_loss / max(batches, 1)
#         validation_loss = _validation_loss(
#             model=model,
#             windows=validation_windows,
#             freq=dataset.freq,
#             device=device,
#             seed=int(config["experiment"]["seed"]) + 10_000,
#         )
#         monitored = validation_loss if np.isfinite(validation_loss) else train_loss
#         scheduler.step(monitored)
#         row: dict[str, float | int | str] = {
#             "phase": phase,
#             "epoch": epoch,
#             "train_loss": train_loss,
#             "validation_loss": validation_loss,
#             "learning_rate": float(optimizer.param_groups[0]["lr"]),
#             "gradient_norm": gradient_norm,
#         }
#         history.append(row)
#         LOGGER.info(
#             "%s epoch %d/%d - train_loss=%.6f val_loss=%s lr=%.3g",
#             phase,
#             epoch,
#             max_epochs,
#             train_loss,
#             f"{validation_loss:.6f}" if np.isfinite(validation_loss) else "n/a",
#             optimizer.param_groups[0]["lr"],
#         )

#         if monitored < best_metric - minimum_delta:
#             best_metric = monitored
#             best_epoch = epoch
#             epochs_without_improvement = 0
#             best_state = {
#                 key: value.detach().cpu().clone()
#                 for key, value in model.state_dict().items()
#             }
#             _save_checkpoint(
#                 checkpoint_path,
#                 model,
#                 optimizer,
#                 epoch,
#                 history,
#                 dataset,
#                 phase,
#             )
#         else:
#             epochs_without_improvement += 1

#         if epoch >= minimum_epochs and epochs_without_improvement >= patience:
#             LOGGER.info("Early stopping %s at epoch %d", phase, epoch)
#             break

#     if best_state is None:
#         raise RuntimeError("training completed without a valid checkpoint")
#     model.load_state_dict(best_state)
#     model.to(device)
#     improved = len(history) >= 2 and float(history[-1]["train_loss"]) < float(
#         history[0]["train_loss"]
#     )
#     return history, best_epoch, improved


# def save_history_csv(
#     path: Path, history: Sequence[Mapping[str, float | int | str]]
# ) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     fieldnames = [
#         "phase",
#         "epoch",
#         "train_loss",
#         "validation_loss",
#         "learning_rate",
#         "gradient_norm",
#     ]
#     with path.open("w", encoding="utf-8", newline="") as stream:
#         writer = csv.DictWriter(stream, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(history)


# def train_experiment(
#     dataset: DatasetBundle,
#     config: Mapping[str, Any],
#     time_feature_dim: int,
#     device: torch.device,
#     checkpoint_path: Path,
#     history_path: Path,
# ) -> TrainingResult:
#     # validation_count = int(config.get("validation", {}).get("windows", 0))
#     # train_target, validation_windows = make_validation_windows(
#     #     dataset,
#     #     count=validation_count,
#     #     history_length=int(config["model"]["context_length"])
#     #     + max(int(value) for value in config["model"]["lags"]),
#     # )
#     validation_config = config.get("validation", {})

#     validation_count = int(
#         validation_config.get("windows", 0)
#     )

#     validation_stride = validation_config.get("stride")

#     train_target, validation_windows = make_validation_windows(
#         dataset,
#         count=validation_count,
#         history_length=(
#             int(config["model"]["context_length"])
#             + max(int(value) for value in config["model"]["lags"])
#         ),
#         stride=(
#             int(validation_stride)
#             if validation_stride is not None
#             else None
#         ),
#     )
#     model = build_model(
#         config=config,
#         target_dim=dataset.target_dim,
#         time_feature_dim=time_feature_dim,
#         prediction_length=dataset.prediction_length,
#     ).to(device)
#     LOGGER.info(
#         "TimeGrad parameters=%d, train_windows_prefix=%d, validation_windows=%d",
#         model.parameter_count(),
#         len(train_target),
#         len(validation_windows),
#     )

#     tuning_history, best_epoch, improved = _fit(
#         model=model,
#         train_target=train_target,
#         train_start=dataset.train_start,
#         validation_windows=validation_windows,
#         dataset=dataset,
#         config=config,
#         device=device,
#         checkpoint_path=checkpoint_path,
#         phase="tune",
#         max_epochs=int(config["training"]["epochs"]),
#     )
#     complete_history = list(tuning_history)

#     should_retrain = bool(
#         config.get("validation", {}).get("retrain_full", True)
#     )
#     if validation_windows and should_retrain:
#         LOGGER.info("Retraining on all training points for %d epoch(s)", best_epoch)
#         model = build_model(
#             config=config,
#             target_dim=dataset.target_dim,
#             time_feature_dim=time_feature_dim,
#             prediction_length=dataset.prediction_length,
#         ).to(device)
#         retrain_history, _, retrain_improved = _fit(
#             model=model,
#             train_target=dataset.train_target,
#             train_start=dataset.train_start,
#             validation_windows=(),
#             dataset=dataset,
#             config=config,
#             device=device,
#             checkpoint_path=checkpoint_path,
#             phase="retrain_full",
#             max_epochs=max(
#                 best_epoch,
#                 int(config["training"].get("minimum_epochs", 2)),
#             ),
#         )
#         complete_history.extend(retrain_history)
#         improved = improved or retrain_improved

#     save_history_csv(history_path, complete_history)
#     return TrainingResult(
#         model=model,
#         history=complete_history,
#         best_epoch=best_epoch,
#         checkpoint_path=checkpoint_path,
#         improved=improved,
#     )


# def load_trained_model(
#     path: Path,
#     device: torch.device,
# ) -> tuple[TimeGrad, dict[str, Any]]:
#     checkpoint = torch.load(path, map_location=device, weights_only=False)
#     model = TimeGrad(TimeGradConfig.from_dict(checkpoint["model_config"]))
#     model.load_state_dict(checkpoint["model_state"])
#     model.to(device)
#     model.eval()
#     return model, checkpoint

"""Training and checkpointing for a single TimeGrad experiment."""

from __future__ import annotations

import csv
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler

from data import (
    DatasetBundle,
    ForecastWindow,
    SlidingWindowDataset,
    make_validation_windows,
    window_tensors,
)
from models import TimeGrad, TimeGradConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    model: TimeGrad
    history: list[dict[str, float | int | str]]
    best_epoch: int
    checkpoint_path: Path
    improved: bool


def build_model(
    config: Mapping[str, Any],
    target_dim: int,
    time_feature_dim: int,
    prediction_length: int,
) -> TimeGrad:
    model_values = deepcopy(dict(config["model"]))
    model_values.update(
        {
            "target_dim": target_dim,
            "time_feature_dim": time_feature_dim,
            "prediction_length": prediction_length,
        }
    )
    return TimeGrad(TimeGradConfig.from_dict(model_values))


def _make_train_loader(
    dataset: SlidingWindowDataset,
    batch_size: int,
    steps_per_epoch: int,
    seed: int,
    num_workers: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=batch_size * steps_per_epoch,
        generator=generator,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def _validation_loss(
    model: TimeGrad,
    windows: Sequence[ForecastWindow],
    freq: str,
    device: torch.device,
    seed: int,
) -> float:
    if not windows:
        return float("nan")
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    losses = []
    with torch.no_grad():
        for window in windows:
            tensors = window_tensors(window, freq, model.config.context_length)
            history = tensors["history"].to(device)
            future = tensors["target"].to(device)
            features = torch.cat(
                (
                    tensors["context_time_features"],
                    tensors["future_time_features"],
                ),
                dim=1,
            ).to(device)
            loss = model.training_loss(
                history=history,
                future=future,
                time_features=features,
                generator=generator,
            )
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _save_checkpoint(
    path: Path,
    model: TimeGrad,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: Sequence[Mapping[str, Any]],
    dataset: DatasetBundle,
    phase: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": model.config.to_dict(),
            "epoch": epoch,
            "phase": phase,
            "history": list(history),
            "dataset": {
                "name": dataset.name,
                "benchmark": dataset.benchmark,
                "freq": dataset.freq,
                "prediction_length": dataset.prediction_length,
                "target_dim": dataset.target_dim,
            },
        },
        temporary,
    )
    temporary.replace(path)


def _fit(
    model: TimeGrad,
    train_target: np.ndarray,
    train_start,
    validation_windows: Sequence[ForecastWindow],
    dataset: DatasetBundle,
    config: Mapping[str, Any],
    device: torch.device,
    checkpoint_path: Path,
    phase: str,
    max_epochs: int,
) -> tuple[list[dict[str, float | int | str]], int, bool]:
    training = config["training"]
    window_dataset = SlidingWindowDataset(
        target=train_target,
        start=train_start,
        freq=dataset.freq,
        history_length=model.config.history_length,
        context_length=model.config.context_length,
        prediction_length=model.config.prediction_length,
    )
    loader = _make_train_loader(
        dataset=window_dataset,
        batch_size=int(training["batch_size"]),
        steps_per_epoch=int(training["steps_per_epoch"]),
        seed=int(config["experiment"]["seed"]),
        num_workers=int(training.get("num_workers", 0)),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training.get("lr_plateau_factor", 0.5)),
        patience=int(training.get("lr_plateau_patience", 1)),
    )
    amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    gradient_clip = float(training.get("gradient_clip", 1.0))
    patience = int(training.get("early_stopping_patience", max_epochs))
    minimum_epochs = int(training.get("minimum_epochs", 2))
    minimum_delta = float(training.get("minimum_delta", 0.0))

    history: list[dict[str, float | int | str]] = []
    best_metric = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        gradient_norm = 0.0
        for batch in loader:
            history_tensor = batch["history"].to(device, non_blocking=True)
            future_tensor = batch["future"].to(device, non_blocking=True)
            features_tensor = batch["time_features"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                loss = model.training_loss(
                    history_tensor, future_tensor, features_tensor
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = float(clip_grad_norm_(model.parameters(), gradient_clip))
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().cpu())
            batches += 1

        train_loss = epoch_loss / max(batches, 1)
        validation_loss = _validation_loss(
            model=model,
            windows=validation_windows,
            freq=dataset.freq,
            device=device,
            seed=int(config["experiment"]["seed"]) + 10_000,
        )
        monitored = validation_loss if np.isfinite(validation_loss) else train_loss
        scheduler.step(monitored)
        row: dict[str, float | int | str] = {
            "phase": phase,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "gradient_norm": gradient_norm,
        }
        history.append(row)
        LOGGER.info(
            "%s epoch %d/%d - train_loss=%.6f val_loss=%s lr=%.3g",
            phase,
            epoch,
            max_epochs,
            train_loss,
            f"{validation_loss:.6f}" if np.isfinite(validation_loss) else "n/a",
            optimizer.param_groups[0]["lr"],
        )

        if monitored < best_metric - minimum_delta:
            best_metric = monitored
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            _save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                history,
                dataset,
                phase,
            )
        else:
            epochs_without_improvement += 1

        if epoch >= minimum_epochs and epochs_without_improvement >= patience:
            LOGGER.info("Early stopping %s at epoch %d", phase, epoch)
            break

    if best_state is None:
        raise RuntimeError("training completed without a valid checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    improved = len(history) >= 2 and float(history[-1]["train_loss"]) < float(
        history[0]["train_loss"]
    )
    return history, best_epoch, improved


def save_history_csv(
    path: Path, history: Sequence[Mapping[str, float | int | str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "phase",
        "epoch",
        "train_loss",
        "validation_loss",
        "learning_rate",
        "gradient_norm",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def train_experiment(
    dataset: DatasetBundle,
    config: Mapping[str, Any],
    time_feature_dim: int,
    device: torch.device,
    checkpoint_path: Path,
    history_path: Path,
) -> TrainingResult:
    validation_config = config.get("validation", {})
    validation_count = int(validation_config.get("windows", 0))
    validation_stride = validation_config.get("stride")

    train_target, validation_windows = make_validation_windows(
        dataset,
        count=validation_count,
        history_length=(
            int(config["model"]["context_length"])
            + max(int(value) for value in config["model"]["lags"])
        ),
        stride=(
            int(validation_stride)
            if validation_stride is not None
            else None
        ),
    )
    model = build_model(
        config=config,
        target_dim=dataset.target_dim,
        time_feature_dim=time_feature_dim,
        prediction_length=dataset.prediction_length,
    ).to(device)
    LOGGER.info(
        "TimeGrad parameters=%d, train_windows_prefix=%d, validation_windows=%d",
        model.parameter_count(),
        len(train_target),
        len(validation_windows),
    )

    history, best_epoch, improved = _fit(
        model=model,
        train_target=train_target,
        train_start=dataset.train_start,
        validation_windows=validation_windows,
        dataset=dataset,
        config=config,
        device=device,
        checkpoint_path=checkpoint_path,
        phase="train",
        max_epochs=int(config["training"]["epochs"]),
    )

    save_history_csv(history_path, history)
    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        checkpoint_path=checkpoint_path,
        improved=improved,
    )


def load_trained_model(
    path: Path,
    device: torch.device,
) -> tuple[TimeGrad, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = TimeGrad(TimeGradConfig.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint

