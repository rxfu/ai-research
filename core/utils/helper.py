import time

import torch


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 CUDA，但当前 CUDA 不可用")

    if name == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("当前环境不支持 MPS")

    if name not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"不支持的设备: {name}")

    return torch.device(name)


def count_parameters(model: torch.nn.Module) -> dict:
    return {
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "parameters_trainable": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }


@torch.no_grad()
def measure_inference_latency(model, loader, device, warmup_batches=2):
    """
    测量批次前向传播耗时，并换算为每个样本的平均耗时。

    该指标不是完整的端到端服务延迟。
    """
    model.eval()
    timings = []

    for index, (x, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        if index >= warmup_batches:
            timings.append(elapsed / max(1, x.size(0)))

        if index >= warmup_batches + 10:
            break

    average = sum(timings) / len(timings) if timings else None

    return {
        "inference_seconds_per_sample": average,
        "inference_samples_per_second": (
            1.0 / average if average is not None and average > 0 else None
        ),
    }
