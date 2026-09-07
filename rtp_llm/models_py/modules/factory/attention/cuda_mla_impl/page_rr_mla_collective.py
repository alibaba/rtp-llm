"""Merge Page-RR partial attention back into the existing attention-head layout."""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather_into,
    all_reduce,
    get_process_group,
    reduce_scatter,
)


@triton.jit
def _weight_partial_output(
    partial,
    lses,
    weighted,
    tokens,
    heads: tl.constexpr,
    dim: tl.constexpr,
    cp_size: tl.constexpr,
    cp_rank: tl.constexpr,
    block: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    maximum = tl.full((), -float("inf"), tl.float32)
    for rank in tl.static_range(cp_size):
        lse = tl.load(lses + (rank * tokens + token) * heads + head)
        maximum = tl.maximum(maximum, lse, propagate_nan=tl.PropagateNan.ALL)
    has_keys = maximum != -float("inf")
    denominator = tl.full((), 0.0, tl.float32)
    for rank in tl.static_range(cp_size):
        lse = tl.load(lses + (rank * tokens + token) * heads + head)
        denominator += tl.where(has_keys, tl.exp2(lse - maximum), 0.0)
    local_lse = tl.load(lses + (cp_rank * tokens + token) * heads + head)
    scale = tl.where(has_keys, tl.exp2(local_lse - maximum) / denominator, 0.0)
    d = tl.arange(0, block)
    value = tl.load(partial + (token * heads + head) * dim + d, d < dim, other=0).to(
        tl.float32
    )
    # Empty shards have no meaningful O. Do not hide NaN/Inf from nonempty ones.
    value = tl.where(local_lse == -float("inf"), 0.0, value)
    # Head-major output is already the contiguous reduce-scatter input and
    # the batched V-projection input; neither consumer needs a reorder copy.
    tl.store(weighted + (head * tokens + token) * dim + d, value * scale, d < dim)


def merge_page_rr_attention(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    *,
    replicated_heads: bool,
) -> torch.Tensor:
    """Merge normalized O[T,H,L] and base-2 LSE[T,H] over the original TP group.

    Return FP32 [local_heads,T,L]. With head TP, H is all gathered heads and
    reduce-scatter restores each rank's head range. With compute CP, every rank
    already has all heads and retains the entire all-reduced output instead.
    """
    group = get_process_group(Group.TP)
    cp_size = torch.distributed.get_world_size(group)
    cp_rank = torch.distributed.get_rank(group)
    tokens, heads, dim = partial_output.shape
    if (
        not partial_output.is_contiguous()
        or not partial_lse.is_contiguous()
        or partial_lse.shape != (tokens, heads)
        or partial_lse.dtype != torch.float32
        or partial_lse.device != partial_output.device
    ):
        raise ValueError("Page-RR merge requires contiguous O[T,H,L] and FP32 LSE[T,H]")
    if not replicated_heads and heads % cp_size:
        raise ValueError(
            "Page-RR gathered heads must be divisible by the TP group size"
        )
    lses = torch.empty(
        (cp_size * tokens, heads), dtype=torch.float32, device=partial_lse.device
    )
    all_gather_into(partial_lse, lses, Group.TP)
    weighted = torch.empty(
        (heads, tokens, dim), dtype=torch.float32, device=partial_output.device
    )
    _weight_partial_output[(tokens, heads)](
        partial_output,
        lses,
        weighted,
        tokens,
        heads=heads,
        dim=dim,
        cp_size=cp_size,
        cp_rank=cp_rank,
        block=triton.next_power_of_2(dim),
        num_warps=8,
    )
    if replicated_heads:
        return all_reduce(weighted, Group.TP)
    return reduce_scatter(weighted, Group.TP)
