from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


def data_to_device(data: Any, device: torch.device) -> Any:
    """递归地将批次中的 Tensor 移动到指定设备。"""
    if isinstance(data, torch.Tensor):
        return data.to(device)

    if isinstance(data, dict):
        return {key: data_to_device(value, device) for key, value in data.items()}

    if isinstance(data, tuple):
        return tuple(data_to_device(value, device) for value in data)

    if isinstance(data, list):
        return [data_to_device(value, device) for value in data]

    return data


def infer_batch_size(targets: Any) -> int:
    """从 targets 中获取当前批次的样本数。"""
    if isinstance(targets, torch.Tensor):
        if targets.ndim == 0:
            return 1
        return targets.shape[0]

    if isinstance(targets, dict):
        for value in targets.values():
            try:
                return infer_batch_size(value)
            except (TypeError, ValueError):
                continue

    if isinstance(targets, (tuple, list)) and targets:
        return infer_batch_size(targets[0])

    raise ValueError("无法从 targets 推断 batch size。")


def to_float(value: Any) -> float:
    """将单个指标转换为 Python float。"""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("每个指标必须是标量 Tensor。")
        return value.detach().item()

    return float(value)


class Trainer:
    """通用模型训练、验证与测试器。"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: Optional[str | torch.device] = None,
        metric_fn: Optional[Callable] = None,
        grad_clip_norm: Optional[float] = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.grad_clip_norm = grad_clip_norm

        self.history: list[dict[str, Any]] = []

    def run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        """执行一个训练 epoch，或完整的验证/测试流程。"""
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_samples = 0
        metric_sums: dict[str, float] = {}

        # 训练时启用梯度；验证与测试时禁用梯度。
        with torch.set_grad_enabled(training):
            for batch in loader:
                if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                    raise ValueError("每个 batch 应为 (inputs, targets)。")

                inputs, targets = batch
                inputs = data_to_device(inputs, self.device)
                targets = data_to_device(targets, self.device)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs.squeeze(1), targets)

                if loss.ndim != 0:
                    raise ValueError(
                        "loss_fn 应返回标量；" "如使用 reduction='none'，请先自行聚合。"
                    )

                if training:
                    loss.backward()

                    if self.grad_clip_norm is not None:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.grad_clip_norm,
                        )

                    self.optimizer.step()

                batch_size = infer_batch_size(targets)

                total_loss += loss.detach().item() * batch_size
                total_samples += batch_size

                if self.metric_fn is not None:
                    metrics = self.metric_fn(outputs.detach(), targets)

                    # 支持返回单个指标，也支持返回指标字典。
                    if not isinstance(metrics, dict):
                        metrics = {"metric": metrics}

                    for name, value in metrics.items():
                        metric_sums[name] = (
                            metric_sums.get(name, 0.0) + to_float(value) * batch_size
                        )

        if total_samples == 0:
            raise ValueError("DataLoader 没有产生任何样本。")

        results = {
            "loss": total_loss / total_samples,
        }

        for name, value in metric_sums.items():
            results[name] = value / total_samples

        return results

    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        """训练一个 epoch。"""
        return self.run_epoch(train_loader, training=True)

    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        """执行完整验证集评估。"""
        return self.run_epoch(val_loader, training=False)

    def test(self, test_loader: DataLoader) -> dict[str, float]:
        """执行完整测试集评估。"""
        return self.run_epoch(test_loader, training=False)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        checkpoint_path: Optional[str | Path] = None,
    ) -> list[dict[str, Any]]:
        """完整训练流程，并可保存验证集 loss 最优的模型。"""
        if epochs <= 0:
            raise ValueError("epochs 必须大于 0。")

        best_loss = float("inf")
        save_path = Path(checkpoint_path) if checkpoint_path else None

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)

            record: dict[str, Any] = {
                "epoch": epoch,
                "train": train_metrics,
            }

            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                record["val"] = val_metrics
                monitored_loss = val_metrics["loss"]
            else:
                monitored_loss = train_metrics["loss"]

            self.history.append(record)

            message = f"Epoch [{epoch}/{epochs}] train_loss={train_metrics['loss']:.4f}"

            if val_loader is not None:
                message += f" val_loss={record['val']['loss']:.4f}"

            print(message)

            if monitored_loss < best_loss:
                best_loss = monitored_loss

                if save_path is not None:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "best_loss": best_loss,
                            "history": self.history,
                        },
                        save_path,
                    )

        return self.history

    @torch.inference_mode()
    def evaluate(self, dataloader):

        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        all_outputs = []
        all_targets = []

        for inputs, targets in dataloader:

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)

            batch_size = targets.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())

        if total_samples == 0:
            raise ValueError("评估数据集为空。")

        outputs = torch.cat(all_outputs, dim=0)
        targets = torch.cat(all_targets, dim=0)

        predictions = outputs.argmax(dim=1)

        # 计算整个数据集的指标
        metrics = self.compute_metrics(
            predictions,
            targets,
        )

        metrics["loss"] = total_loss / total_samples

        return metrics, predictions, targets

    @staticmethod
    def compute_metrics(predictions, targets):

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        predictions = predictions.numpy()
        targets = targets.numpy()

        return {
            "accuracy": accuracy_score(targets, predictions),
            "precision": precision_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "recall": recall_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "f1": f1_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            ),
        }
