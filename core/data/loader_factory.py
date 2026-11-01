from typing import Any, Callable, Optional

import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    Sampler,
)


class DataLoaderFactory:

    @staticmethod
    def build(
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        generator: Optional[torch.Generator] = None,
        collate_fn: Optional[Callable] = None,
        sampler: Optional[Sampler] = None,
        persistent_workers: Optional[bool] = None,
        prefetch_factor: Optional[int] = None,
        seed_worker: Optional[Callable] = None,
    ) -> DataLoader:

        if not isinstance(dataset, Dataset):
            raise TypeError(f"ds 必须是 Dataset，当前 {type(dataset)}")

        if batch_size <= 0:
            raise ValueError(f"batch_size 必须 > 0，当前 {batch_size}")

        if num_workers < 0:
            raise ValueError(f"num_workers 必须 >= 0，当前 {num_workers}")

        if sampler is not None and shuffle:
            raise ValueError("sampler and shuffle=True 不可同时使用")

        if persistent_workers and num_workers == 0:
            raise ValueError(
                f"persistent_workers 要求 num_workers > 0，当前 {num_workers}"
            )

        if prefetch_factor is not None:
            if num_workers == 0:
                raise ValueError(
                    f"prefetch_factor 要求 num_workers > 0，当前 {num_workers}"
                )
            if prefetch_factor <= 0:
                raise ValueError(f"prefetch_factor 必须 > 0，当前 {prefetch_factor}")

        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle if sampler is None else False,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": drop_last,
            "collate_fn": collate_fn,
            "sampler": sampler,
            "generator": generator,
        }

        if num_workers > 0:
            kwargs["worker_init_fn"] = seed_worker
            kwargs["persistent_workers"] = persistent_workers

            if prefetch_factor is not None:
                kwargs["prefetch_factor"] = prefetch_factor

        return DataLoader(**kwargs)

    @staticmethod
    def build_train(dataset: Dataset, **kwargs) -> DataLoader:
        return DataLoaderFactory.build(dataset, shuffle=True, **kwargs)

    @staticmethod
    def build_eval(dataset: Dataset, **kwargs) -> DataLoader:
        return DataLoaderFactory.build(
            dataset, shuffle=False, drop_last=False, **kwargs
        )
