"""Select cuLA's inference checkpoint ABI for KDA prefill."""

import inspect
from collections.abc import Callable
from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def get_cula_checkpoint_kernel(
    device: torch.device,
    dtype: torch.dtype,
    head_dim: int,
    state_dtype: torch.dtype,
    lower_bound: float | None,
    page_size: int,
) -> Callable | None:
    if (
        device.type != "cuda"
        or torch.version.hip is not None
        or dtype != torch.bfloat16
        or head_dim != 128
        or state_dtype != torch.float32
        or lower_bound is None
        or not -5 <= lower_bound < 0
        or page_size <= 0
        or page_size % 64 != 0
        or torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3))
    ):
        return None

    try:
        from cula.kda import chunk_kda
    except (ImportError, OSError):
        return None
    try:
        params = inspect.signature(chunk_kda).parameters
    except (TypeError, ValueError):
        return None
    # Older releases accept arbitrary kwargs without implementing checkpoints.
    required = (
        "checkpoint_states",
        "checkpoint_interval",
        "cu_seqlens_cpu",
        "use_beta_sigmoid_in_kernel",
        "use_intracard_cp",
    )
    keyword_kinds = (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
    if all(name in params and params[name].kind in keyword_kinds for name in required):
        return chunk_kda
    return None
