from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

from core.utils.io import load_json, load_csv


def load_history(path: str | Path) -> dict[str, Any]:
    path = Path(path) / "history.json"

    return load_json(path)


def load_metrics(path: str | Path) -> dict[str, Any]:
    path = Path(path) / "metrics.json"

    return load_json(path)


def load_summary(path: str | Path) -> pd.DataFrame:
    path = Path(path) / "summary.csv"

    return load_csv(path)


def discover_exps(root: Path, pattern: str = "*/metrics.json") -> Dict[str, Path]:
    """扫描 root 下所有含 metrics.json 的实验目录。

    返回 {exp_name: exp_dir}
    """
    root = Path(root)
    result = {}
    for mf in root.glob(pattern):
        exp_dir = mf.parent
        result[exp_dir.name] = exp_dir
    return result


def load_multi_seed(exp_dirs: List[Path], metric: str) -> pd.DataFrame:
    """多 seed：读多个实验目录的 metrics，返回 DataFrame(seed, value)。"""
    rows = []
    for i, d in enumerate(exp_dirs):
        m = load_metrics(d)
        rows.append({"seed": i, "value": m.get(metric)})
    return pd.DataFrame(rows)
