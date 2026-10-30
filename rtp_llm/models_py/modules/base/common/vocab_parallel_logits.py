from collections.abc import Callable
from typing import Optional

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather


def gather_vocab_parallel_logits(
    local: torch.Tensor,
    tp_size: int,
    *,
    try_gather: Optional[Callable[[torch.Tensor], Optional[torch.Tensor]]] = None,
) -> torch.Tensor:
    """Gather vocabulary shards in rank order and convert once to sampler FP32.

    A prepared transport may return the final tensor or None to use the shared
    TP collective. Communication retains the local head's original dtype.
    """
    if (
        local.ndim != 2
        or tp_size <= 0
        or local.dtype
        not in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        )
    ):
        raise ValueError(
            "vocabulary logits require floating-point [rows, shard] and positive TP size"
        )
    rows, cols = local.shape
    if tp_size == 1:
        return local.float().contiguous()
    if try_gather is not None:
        logits = try_gather(local)
        if logits is not None:
            if (
                logits.device != local.device
                or logits.dtype != torch.float32
                or logits.shape != (rows, cols * tp_size)
                or not logits.is_contiguous()
            ):
                raise RuntimeError(
                    "vocabulary transport returned invalid gathered logits"
                )
            return logits
    gathered = all_gather(local.contiguous(), group=Group.TP)
    if gathered.shape != (tp_size * rows, cols):
        raise RuntimeError(
            "vocabulary TP collective returned invalid rank-major logits"
        )
    # One strided copy reorders ranks and converts BF16/FP16 to FP32, without
    # allocating an intermediate contiguous tensor in the transport dtype.
    logits = torch.empty(
        (rows, tp_size, cols), device=local.device, dtype=torch.float32
    )
    logits.copy_(gathered.reshape(tp_size, rows, cols).permute(1, 0, 2))
    return logits.reshape(rows, tp_size * cols)
