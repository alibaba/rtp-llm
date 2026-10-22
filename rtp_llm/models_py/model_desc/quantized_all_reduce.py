"""INT8 tensor-parallel reduction shared by the Qwen decoder layers."""

from typing import Callable, Optional

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, _get_group, all_reduce
from rtp_llm.models_py.kernels.cuda.fast_bf16_int8 import support


def _quantized_all_reduce(
    hidden_states: torch.Tensor,
    quant: Optional[Callable[..., None]],
    dequant: Optional[Callable[..., None]],
    *,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Gather INT8 codes with the selected scale dtype and reduce in rank order."""
    group_size = 16
    if (
        quant is None
        or dequant is None
        or not support(hidden_states, group_size)
        or hidden_states.numel() == 0
    ):
        return all_reduce(hidden_states, group=Group.TP)
    pg = _get_group(Group.TP)
    tp_size = pg.size()
    if tp_size == 1:
        return hidden_states
    m, k = hidden_states.shape
    q = torch.empty_like(hidden_states, dtype=torch.int8)
    s = torch.empty(
        (m, k // group_size), dtype=scale_dtype, device=hidden_states.device
    )
    q_all = torch.empty((tp_size * m, k), dtype=torch.int8, device=hidden_states.device)
    s_all = torch.empty(
        (tp_size * m, k // group_size),
        dtype=scale_dtype,
        device=hidden_states.device,
    )
    quant(hidden_states, q, s, group_size=group_size)
    torch.distributed.all_gather_into_tensor(q_all, q, group=pg)
    torch.distributed.all_gather_into_tensor(s_all, s, group=pg)
    dequant(
        q_all.view(tp_size, m, k),
        s_all.view(tp_size, m, k // group_size),
        hidden_states,
        group_size=group_size,
    )
    return hidden_states
