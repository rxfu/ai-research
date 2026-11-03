from copy import deepcopy
from pathlib import Path
from typing import Any

from core.utils.serialize import load_yaml, save_yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置。"""
    config = load_yaml(path)
    # path = Path(path)

    # with path.open("r", encoding="utf-8") as f:
    #     config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"配置必须是字典结构: {path}")

    return config


def save_config(path: str | Path, config: dict) -> None:
    """保存实际运行使用的完整配置。"""
    save_yaml(path, config)
    # path = Path(path)
    # path.parent.mkdir(parents=True, exist_ok=True)
    #
    # with path.open("w", encoding="utf-8") as f:
    #     yaml.safe_dump(
    #         config,
    #         f,
    #         allow_unicode=True,
    #         sort_keys=False,
    #     )


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并字典，不修改原始配置。"""
    result = deepcopy(base)

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def set_dotted(config: dict, dotted_key: str, value: Any) -> None:
    """
    使用点分隔路径修改配置。

    示例：
        set_dotted(config, "model.hidden_dim", 256)
    """
    parts = dotted_key.split(".")
    node = config

    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise KeyError(f"配置路径不存在: {dotted_key}")

        node = node[part]

    if parts[-1] not in node:
        raise KeyError(f"配置项不存在: {dotted_key}")

    node[parts[-1]] = value


def validate_config(config: dict) -> None:
    """在训练开始前检查基本配置。"""
    required = [
        "experiment",
        "data",
        "model",
        "training",
        "evaluation",
    ]

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


# def validate_stage2_config(cfg: dict) -> None:
#     required_sections = [
#         "experiment",
#         "data",
#         "model",
#         "training",
#         "evaluation",
#     ]
#
#     for section in required_sections:
#         if section not in cfg:
#             raise ValueError(f"缺少配置节：{section}")
#
#     training = cfg["training"]
#     data = cfg["data"]
#
#     if training["epochs"] < 1:
#         raise ValueError("epochs 必须大于 0")
#
#     if training.get("accumulation_steps", 1) < 1:
#         raise ValueError("accumulation_steps 必须大于 0")
#
#     if data["batch_size"] < 1:
#         raise ValueError("batch_size 必须大于 0")
#
#     if training["monitor_mode"] not in {"min", "max"}:
#         raise ValueError("monitor_mode 必须为 min 或 max")
#
#     if training["learning_rate"] <= 0:
#         raise ValueError("learning_rate 必须大于 0")
#
#     if training.get("weight_decay", 0) < 0:
#         raise ValueError("weight_decay 不能小于 0")
#
#     if training.get("patience", 1) < 1:
#         raise ValueError("patience 必须大于 0")
