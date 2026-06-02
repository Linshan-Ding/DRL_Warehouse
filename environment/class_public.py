from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.json"
CONFIG_ENV_VAR = "DRL_WAREHOUSE_CONFIG"

REQUIRED_SECTIONS = (
    "warehouse",
    "robot",
    "picker",
    "order",
    "item",
    "ppo",
    "experiment",
    "paths",
)
_CONFIG_CACHE: dict[str, dict[str, Any]] = {}


def resolve_config_path(config_path: str | os.PathLike[str] | None = None) -> Path:
    raw_path = config_path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def deep_update(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _as_tuple(parameters: dict[str, Any], section: str, key: str) -> None:
    value = parameters.get(section, {}).get(key)
    if isinstance(value, list):
        parameters[section][key] = tuple(value)


def _normalize(parameters: dict[str, Any]) -> dict[str, Any]:
    _as_tuple(parameters, "warehouse", "depot_position")
    _as_tuple(parameters, "order", "poisson_parameter")
    _as_tuple(parameters, "order", "order_n_arrival")
    _as_tuple(parameters, "order", "order_n_items")
    return parameters


def validate_config(parameters: Mapping[str, Any]) -> None:
    missing = [section for section in REQUIRED_SECTIONS if section not in parameters]
    if missing:
        raise KeyError(f"Missing config sections: {', '.join(missing)}")

    warehouse = parameters["warehouse"]
    for key in ("shelf_capacity", "shelf_levels", "area_num", "aisle_num"):
        if int(warehouse[key]) <= 0:
            raise ValueError(f"warehouse.{key} must be positive")

    experiment = parameters["experiment"]
    if int(experiment["episodes"]) <= 0:
        raise ValueError("experiment.episodes must be positive")
    if int(experiment["max_days"]) <= 0:
        raise ValueError("experiment.max_days must be positive")


def load_config(
    config_path: str | os.PathLike[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = resolve_config_path(config_path)
    cache_key = str(path.resolve())
    if overrides is None and cache_key in _CONFIG_CACHE:
        return copy.deepcopy(_CONFIG_CACHE[cache_key])

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        parameters = json.load(f)

    if overrides:
        parameters = deep_update(parameters, dict(overrides))

    parameters = _normalize(parameters)
    validate_config(parameters)
    if overrides is None:
        _CONFIG_CACHE[cache_key] = copy.deepcopy(parameters)
    return parameters


class Config:
    def __init__(
        self,
        config_path: str | os.PathLike[str] | None = None,
        overrides: Mapping[str, Any] | None = None,
    ):
        self.config_path = resolve_config_path(config_path)
        self.parameters = load_config(self.config_path, overrides)

    def parameter(self) -> dict[str, Any]:
        return copy.deepcopy(self.parameters)
