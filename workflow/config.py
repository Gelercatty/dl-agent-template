"""Reviewable YAML experiments; JSON remains readable for existing job definitions."""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from .common import read_json, require


class ExperimentLoader(yaml.SafeLoader):
    def compose_node(self, parent, index):
        require(not self.check_event(yaml.AliasEvent), "YAML aliases/overlays are not supported")
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            require(type(key) is str, "Config keys must be strings")
            require(key not in result, f"Duplicate YAML key: {key}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _plain(value):
    if isinstance(value, dict):
        require(all(type(k) is str for k in value), "Config keys must be strings")
        for child in value.values():
            _plain(child)
    elif isinstance(value, list):
        for child in value:
            _plain(child)
    else:
        require(type(value) in (str, int, float, bool, type(None)), "Unsupported config value")
        if type(value) is float:
            require(math.isfinite(value), "Nonfinite config value")


def read_config(path: Path) -> dict:
    if path.suffix == ".json":
        value = read_json(path)
    else:
        require(path.suffix in {".yaml", ".yml"}, "Config must be YAML or legacy JSON")
        try:
            value = yaml.load(path.read_text(encoding="utf-8"), Loader=ExperimentLoader)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    require(isinstance(value, dict), "Config must be a mapping")
    _plain(value)
    return value


def max_steps(config: dict) -> int:
    return config["training"]["max_steps"] if config["schema"] == 2 else config["max_steps"]
