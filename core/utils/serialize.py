import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def save_yaml(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def load_yaml(path: str | Path) -> Any:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save_csv(path: str | Path, data: pd.DataFrame, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(path, index=index, encoding="utf-8-sig")


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")

    return pd.read_csv(path)
