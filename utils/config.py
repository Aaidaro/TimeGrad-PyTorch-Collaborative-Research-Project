"""Configuration, paths, logging, and reproducibility helpers."""

from __future__ import annotations

import json
import logging.config
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if "extends" in config:
        base_path = (path.parent / config.pop("extends")).resolve()
        config = deep_merge(load_config(base_path), config)

    project_root = path.parent.parent
    config["_config_path"] = str(path)
    config["_project_root"] = str(project_root)
    return config


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must use dotted.path=value: {override}")
        dotted_key, raw_value = override.split("=", 1)
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as error:
            raise ValueError(f"invalid override value: {override}") from error
        cursor = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"override path is not a mapping: {dotted_key}")
        cursor[parts[-1]] = value
    return result


def resolve_project_path(config: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(os.path.expandvars(str(path)))
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def configure_logging(config: Mapping[str, Any]) -> None:
    logging_path = resolve_project_path(
        config, config.get("logging_config", "config/logging.yaml")
    )
    with logging_path.open("r", encoding="utf-8") as stream:
        logging_config = yaml.safe_load(stream)
    logging.config.dictConfig(logging_config)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)
