import sys
from pathlib import Path

import matplotlib

# 判断是否为 Linux 系统
if sys.platform.startswith("linux"):
    # 必须在导入 pyplot 之前设置后端
    matplotlib.use("Agg")  # 无界面后端，服务器可用
    print("检测到 Linux 系统，已设置 Agg 后端（无界面模式）。")
else:
    print("检测到非 Linux 系统，将使用默认交互式后端。")

from matplotlib import pyplot as plt

# 论文常用配色（色盲友好）
PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


def configure_plot_style(
    font_family: str = "DejaVu Sans", font_size: int = 11, dpi: int = 300
) -> None:
    """设置全局风格。Windows 可传 'Microsoft YaHei'。"""
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            # "savefig.bbox": "tight",
            "font.family": font_family,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 1,
            "legend.fontsize": font_size - 1,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.autolayout": True,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(
    fig: plt.Figure,
    save_dir: str | Path,
    filename: str,
    formats: tuple[str, ...] = ("png", "pdf", "svg"),
    dpi: int = 300,
) -> list[Path]:
    """将图表保存为一种或多种格式。"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    paths = []

    for fmt in formats:
        path = save_dir / f"{filename}.{fmt}"
        fig.savefig(path, dpi=dpi, format=fmt)
        paths.append(path)

    return paths


def get_color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def show_figure() -> None:
    plt.show()
