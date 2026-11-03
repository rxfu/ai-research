from pathlib import Path
from typing import Any

from core.utils.serialize import load_yaml, save_yaml


class ConfigManager:
    def __init__(
        self, config: dict[str, Any] | None = None, path: str | Path | None = None
    ) -> None:
        self.config = config or {}
        self.path = path

    def load(self, path: str | Path | None = None) -> None:
        path = path or self.path

        if path is None:
            raise ValueError("path 为 None，不合法")

        data = load_yaml(path)

        if not isinstance(data, dict):
            raise TypeError(f"配置 {path} 不是字典结构")

        self.config = data

    def save(self, path: str | Path | None = None) -> None:
        path = path or self.path

        if path is None:
            raise ValueError("path 为 None，不合法")

        save_yaml(path, self.config)

    def get(self, key: str, default: Any = None) -> Any:
        value = self.config

        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return default

            value = value[part]

        return value

    def set(self, key: str, value: Any) -> None:
        parts = key.split(".")
        current = self.config

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                raise TypeError(f"配置 {part} 不是字典结构")

            current = current[part]

        if parts[-1] not in current:
            raise KeyError(f"配置项不存在: {key}")

        current[parts[-1]] = value

    def validate(self) -> None:
        """在训练开始前检查基本配置。"""
        required = [
            "experiment",
            "data",
            "model",
            "training",
            "evaluation",
        ]
        config = self.config

        for key in required:
            if key not in config:
                raise ValueError(f"缺少配置字段: {key}")

        if config["data"]["num_classes"] < 2:
            raise ValueError("num_classes 必须大于 1")

        if config["data"]["batch_size"] < 1:
            raise ValueError("batch_size 必须大于 0")

        if config["training"]["epochs"] < 1:
            raise ValueError("epochs 必须大于 0")

        if config["training"]["accumulation_steps"] < 1:
            raise ValueError("accumulation_steps 必须大于 0")

        if config["training"]["learning_rate"] <= 0:
            raise ValueError("learning_rate 必须大于 0")

        if config["training"]["weight_decay"] <= 0:
            raise ValueError("weight_decay 不能小于 0")

        if config["training"]["patience"] < 1:
            raise ValueError("patience 必须大于 0")

        if config["training"]["monitor_mode"] not in {"min", "max"}:
            raise ValueError("monitor_mode 必须为 min 或 max")

        if config["model"]["input_dim"] != config["data"]["input_dim"]:
            raise ValueError("model.input_dim 必须与 data.input_dim 一致")

        if config["model"]["num_classes"] != config["data"]["num_classes"]:
            raise ValueError("model.num_classes 必须与 data.num_classes 一致")

    # def to_dict(self) -> dict[str, Any]:
    #
    #     return self.config.copy()
