import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from matplotlib import pyplot as plt

from core.visualization.style import configure_plot_style, save_figure, show_figure


class Plotter:
    """科研实验通用绘图器。

    支持：
    1. 训练曲线
    2. 多模型对比
    3. 消融实验
    4. 效率与序列长度分析
    5. 混淆矩阵
    6. 真实值与预测值对比
    """

    def __init__(
        self,
        save_dir: str | Path,
        dpi: int = 300,
        formats: tuple[str, ...] = ("png", "pdf"),
        font_family: str = "DejaVu Sans",
        show: bool | None = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.dpi = dpi
        self.formats = formats
        self.show = sys.platform == "win32" if show is None else show

        configure_plot_style(font_family=font_family, dpi=dpi)

    def _save(self, fig: plt.Figure, filename: str) -> list[Path]:
        """保存图表并关闭 Figure，避免长期实验中内存累积。"""
        try:
            paths = save_figure(
                fig=fig,
                save_dir=self.save_dir,
                filename=filename,
                formats=self.formats,
                dpi=self.dpi,
            )

            if self.show:
                show_figure()

            return paths
        finally:
            plt.close(fig)

    @staticmethod
    def _to_numpy(data) -> np.ndarray:
        """将 Tensor 或数组转换为 NumPy 数组。"""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()

        return np.asarray(data)

    def plot_dataset_samples(
        self,
        features,
        labels=None,
        feature_indices: tuple[int, int] = (0, 1),
        max_samples: int = 5000,
        filename: str = "dataset_samples",
    ) -> list[Path]:
        """绘制数据集样本分布。

        features: [N, D]
        labels: [N]，可选
        feature_indices: 用于绘图的两个特征维度
        """
        x = self._to_numpy(features)

        if x.ndim != 2:
            raise ValueError("features 的形状必须是 [N, D]。")

        if len(x) == 0:
            raise ValueError("features 不能为空。")

        i, j = feature_indices
        if not (0 <= i < x.shape[1] and 0 <= j < x.shape[1]):
            raise ValueError("feature_indices 超出特征数量范围。")

        if max_samples > 0 and len(x) > max_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(x), size=max_samples, replace=False)
            x = x[indices]
            if labels is not None:
                labels = self._to_numpy(labels)[indices]

        fig, ax = plt.subplots(figsize=(6, 5))

        if labels is None:
            ax.scatter(x[:, i], x[:, j], s=12, alpha=0.7)
        else:
            y = self._to_numpy(labels).reshape(-1)
            if len(y) != len(x):
                raise ValueError("labels 长度必须与 features 一致。")

            for label in np.unique(y):
                mask = y == label
                ax.scatter(
                    x[mask, i],
                    x[mask, j],
                    s=16,
                    alpha=0.7,
                    label=str(label),
                )
            ax.legend(title="Class")

        ax.set_xlabel(f"Feature {i}")
        ax.set_ylabel(f"Feature {j}")
        ax.set_title("Dataset Sample Distribution")

        fig.tight_layout()
        return self._save(fig, filename)

    def plot_class_distribution(
        self,
        labels,
        filename: str = "class_distribution",
    ) -> list[Path]:
        """绘制各类别样本数量。"""
        y = self._to_numpy(labels).reshape(-1)

        if len(y) == 0:
            raise ValueError("labels cannot be empty.")

        classes, counts = np.unique(y, return_counts=True)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(
            [str(c) for c in classes],
            counts,
        )

        ax.set_xlabel("Class")
        ax.set_ylabel("Number of Samples")
        ax.set_title("Class Distribution")

        for bar, count in zip(bars, counts):
            ax.annotate(
                str(int(count)),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
            )

        fig.tight_layout()
        return self._save(fig, filename)

    def plot_feature_distribution(
        self,
        features,
        feature_names: Sequence[str] | None = None,
        max_features: int = 12,
        bins: int = 30,
        filename: str = "feature_distribution",
    ) -> list[Path]:
        """分别绘制多个特征的直方图。"""
        x = self._to_numpy(features)

        if x.ndim != 2 or len(x) == 0:
            raise ValueError("features must have nonempty shape [N, D].")

        num_features = min(x.shape[1], max_features)

        if feature_names is not None and len(feature_names) != x.shape[1]:
            raise ValueError("feature_names length must match feature dimension.")

        cols = min(3, num_features)
        rows = (num_features + cols - 1) // cols

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(4 * cols, 3 * rows),
            squeeze=False,
        )

        for i, ax in enumerate(axes.flat):
            if i >= num_features:
                ax.set_visible(False)
                continue

            values = x[:, i]
            values = values[np.isfinite(values)]

            if len(values) == 0:
                ax.set_title(f"Feature {i}: no finite values")
                continue

            ax.hist(values, bins=bins, edgecolor="black", alpha=0.8)
            name = feature_names[i] if feature_names else f"Feature {i}"
            ax.set_title(name)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")

        fig.suptitle("Feature Distributions")
        fig.tight_layout()
        return self._save(fig, filename)

    def plot_correlation_matrix(
        self,
        features,
        feature_names: Sequence[str] | None = None,
        filename: str = "correlation_matrix",
    ) -> list[Path]:
        """绘制数值特征的 Pearson 相关系数矩阵。"""
        x = self._to_numpy(features)

        if x.ndim != 2 or len(x) < 2:
            raise ValueError("features must have shape [N, D], N >= 2.")

        if not np.isfinite(x).all():
            raise ValueError("features must contain finite values.")

        corr = np.corrcoef(x, rowvar=False)
        corr = np.atleast_2d(corr)

        num_features = x.shape[1]

        if feature_names is not None and len(feature_names) != num_features:
            raise ValueError("feature_names length must match feature dimension.")

        names = (
            list(feature_names)
            if feature_names is not None
            else [f"F{i}" for i in range(num_features)]
        )

        fig, ax = plt.subplots(
            figsize=(
                max(5, num_features * 0.55),
                max(4, num_features * 0.5),
            )
        )

        image = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
        fig.colorbar(image, ax=ax, label="Pearson Correlation")

        ax.set_xticks(np.arange(num_features))
        ax.set_yticks(np.arange(num_features))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        ax.set_title("Feature Correlation Matrix")

        fig.tight_layout()
        return self._save(fig, filename)

    def plot_sequence_samples(
        self,
        sequence,
        feature_indices: Sequence[int] | None = None,
        max_features: int = 4,
        filename: str = "sequence_samples",
    ) -> list[Path]:
        """绘制单个序列。

        sequence: [L] 或 [L, D]
        """
        x = self._to_numpy(sequence)

        if x.ndim == 1:
            x = x[:, None]

        if x.ndim != 2 or len(x) == 0:
            raise ValueError("sequence must have shape [L] or [L, D].")

        if not np.isfinite(x).all():
            raise ValueError("sequence must contain finite values.")

        indices = (
            list(feature_indices)
            if feature_indices is not None
            else list(range(min(x.shape[1], max_features)))
        )

        if not indices:
            raise ValueError("At least one feature must be selected.")

        if any(i < 0 or i >= x.shape[1] for i in indices):
            raise ValueError("feature_indices are out of range.")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        steps = np.arange(len(x))

        for i in indices:
            ax.plot(steps, x[:, i], label=f"Feature {i}")

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.set_title("Sequence Visualization")
        ax.legend()

        fig.tight_layout()
        return self._save(fig, filename)

    def plot_image_samples(
        self,
        images,
        labels=None,
        max_images: int = 16,
        filename: str = "image_samples",
    ) -> list[Path]:
        """绘制图像样本网格。

        images: [N, C, H, W] 或 [N, H, W, C]
        支持 1、3、4 通道。
        """
        x = self._to_numpy(images)

        if x.ndim != 4 or len(x) == 0:
            raise ValueError("images must have shape [N, C, H, W] or [N, H, W, C].")

        count = min(len(x), max_images)

        if labels is not None:
            y = self._to_numpy(labels).reshape(-1)
            if len(y) != len(x):
                raise ValueError("labels length must match images.")
        else:
            y = None

        cols = min(4, count)
        rows = (count + cols - 1) // cols

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(3 * cols, 3 * rows),
            squeeze=False,
        )

        for i, ax in enumerate(axes.flat):
            if i >= count:
                ax.set_visible(False)
                continue

            image = x[i]

            # CHW -> HWC
            if image.ndim == 3 and image.shape[0] in (1, 3, 4):
                image = np.transpose(image, (1, 2, 0))

            image = np.squeeze(image) if image.shape[-1:] == (1,) else image

            # 常见的 [-1, 1] 或 [0, 1] 图像数据处理
            image = image.astype(float)

            if not np.isfinite(image).all():
                raise ValueError("images must contain finite values.")

            if image.min() < 0:
                image = (image - image.min()) / (image.max() - image.min() + 1e-12)

            if image.max() > 1:
                image = np.clip(image / 255.0, 0, 1)

            if image.ndim == 2:
                ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(np.clip(image, 0, 1))

            if y is not None:
                ax.set_title(f"Label: {y[i]}")
            else:
                ax.set_title(f"Sample {i}")

            ax.axis("off")

        fig.suptitle("Image Dataset Samples")
        fig.tight_layout()
        return self._save(fig, filename)

    def plot_graph_structure(
        self,
        edge_index,
        num_nodes: int,
        node_labels=None,
        max_nodes: int = 200,
        seed: int = 42,
        filename: str = "graph_structure",
    ) -> list[Path]:
        """绘制单张图的拓扑结构。

        edge_index: [2, E]，节点编号从 0 开始。
        node_labels: 可选的节点类别，用于区分节点。
        """
        edges = self._to_numpy(edge_index)

        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E].")

        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive.")

        if edges.size:
            if not np.issubdtype(edges.dtype, np.integer):
                if not np.all(edges == np.floor(edges)):
                    raise ValueError("edge_index must contain integer indices.")
            edges = edges.astype(np.int64)

            if edges.min() < 0 or edges.max() >= num_nodes:
                raise ValueError("edge_index contains invalid node indices.")

        # 大图随机抽取节点进行可视化
        rng = np.random.default_rng(seed)
        if num_nodes > max_nodes:
            selected = np.sort(rng.choice(num_nodes, max_nodes, replace=False))
        else:
            selected = np.arange(num_nodes)

        selected_mask = np.zeros(num_nodes, dtype=bool)
        selected_mask[selected] = True

        # 保留抽样节点之间的边
        keep = (
            selected_mask[edges[0]] & selected_mask[edges[1]]
            if edges.shape[1] > 0
            else np.zeros(0, dtype=bool)
        )
        visible_edges = edges[:, keep]

        # 将原始节点编号映射到绘图中的局部编号
        local_index = np.full(num_nodes, -1, dtype=np.int64)
        local_index[selected] = np.arange(len(selected))

        fig, ax = plt.subplots(figsize=(7, 6))

        # 简单的圆形布局，无需额外安装图绘制依赖
        theta = np.linspace(0, 2 * np.pi, len(selected), endpoint=False)
        positions = np.column_stack([np.cos(theta), np.sin(theta)])

        for src, dst in visible_edges.T:
            i = local_index[src]
            j = local_index[dst]

            if i != j:
                ax.plot(
                    [positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    linewidth=0.5,
                    alpha=0.25,
                )

        if node_labels is not None:
            labels = self._to_numpy(node_labels).reshape(-1)

            if len(labels) != num_nodes:
                plt.close(fig)
                raise ValueError("node_labels length must equal num_nodes.")

            visible_labels = labels[selected]

            for label in np.unique(visible_labels):
                mask = visible_labels == label
                ax.scatter(
                    positions[mask, 0],
                    positions[mask, 1],
                    s=24,
                    label=str(label),
                )

            ax.legend(title="Node Label")
        else:
            ax.scatter(positions[:, 0], positions[:, 1], s=24)

        ax.set_title(f"Graph Structure ({len(selected)} of {num_nodes} nodes)")
        ax.axis("off")

        fig.tight_layout()
        return self._save(fig, filename)

    def plot_label_distribution(
        self,
        targets,
        bins: int = 30,
        filename: str = "target_distribution",
    ) -> list[Path]:
        """绘制连续目标值的分布。"""
        y = self._to_numpy(targets).reshape(-1)

        if len(y) == 0:
            raise ValueError("targets cannot be empty.")

        y = y[np.isfinite(y)]

        if len(y) == 0:
            raise ValueError("targets contain no finite values.")

        fig, ax = plt.subplots(figsize=(7, 4.5))

        ax.hist(y, bins=bins, edgecolor="black", alpha=0.8)
        ax.set_xlabel("Target Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Target Distribution")

        fig.tight_layout()
        return self._save(fig, filename)

    def plot_training_curves(
        self, history: Mapping[str, Sequence[float]], filename: str = "training_curves"
    ) -> list[Path]:
        """绘制训练指标曲线。

        示例 history:
        {
            "train_loss": [...],
            "val_loss": [...],
            "val_accuracy": [...],
        }
        """
        if not history:
            raise ValueError("history 不能为空")

        fig, ax = plt.subplots(figsize=(7, 4.5))

        for name, values in history.items():
            values = list(values)

            if not values:
                continue

            epochs = np.arange(1, len(values) + 1)
            ax.plot(epochs, values, label=name)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric value")
        ax.set_title("Training History")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        return self._save(fig, filename)

    def plot_model_comparison(self, filename: str = "model_comparison") -> list[Path]:
        pass

    def plot_ablation(self, filename: str = "ablation") -> list[Path]:
        pass

    def plot_sequence_scaling(self, filename: str = "sequence_scaling") -> list[Path]:
        pass

    def plot_confusion_matrix(self, filename: str = "confusion_matrix") -> list[Path]:
        pass

    def plot_predictions(self, filename: str = "predictions") -> list[Path]:
        pass

    def plot_residuals(self, filename: str = "residuals") -> list[Path]:
        pass
