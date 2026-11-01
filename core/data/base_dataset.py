from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Any, List, TypeAlias

import torch
from torch.utils.data import Dataset

Inputs: TypeAlias = torch.Tensor | dict[str, Any]
Sample: TypeAlias = tuple[Inputs, torch.Tensor]


class BaseDataset(Dataset, ABC):
    VALID_SPLITS = ["train", "val", "test"]

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        joint_transform: Callable[..., Any] | None = None,
    ) -> None:
        self._validate_split(split)

        self.split = split
        self.root = Path(root) if root is not None else None
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        self.samples: List[Any] = []
        self.num_classes: int = 0

        self._setup()

        self._validate()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        if type(index) is not int:
            raise TypeError(
                f"索引必须是int类型，但是索引 {index} 类型是{type(index).__name__}"
            )

        if index < 0:
            index += len(self)

        if not 0 <= index < len(self):
            raise IndexError(f"索引 {index} 越界，数据集长度为 {len(self)}")

        inputs, target = self._load_sample(index)

        return self._apply_transform(inputs, target)

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _load_sample(self, index: int) -> Sample:
        raise NotImplementedError

    def _apply_transform(self, inputs: Any, target: Any) -> Sample:
        if self.joint_transform is not None:
            inputs, target = self.joint_transform(inputs, target)

        if self.transform is not None:
            inputs = self.transform(inputs)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return inputs, target

    def _validate_split(self, split: str | None) -> None:
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"无效的 split: {split!r}，必须是 {sorted(self.VALID_SPLITS)} 之一"
            )

    def _validate(self) -> None:
        if not self.samples:
            raise RuntimeError(
                f"{self.__class__.__name__}: split={self.split} 未加载任何样本，请检查 root={self.root}或_setup()函数"
            )

    def get_num_classes(self) -> int:
        return self.num_classes

    def summary(self) -> str:
        return f"{self.__class__.__name__}(split={self.split}, n={len(self)}, num_classes={self.num_classes})"
