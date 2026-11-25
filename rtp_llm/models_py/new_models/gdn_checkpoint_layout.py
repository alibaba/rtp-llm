"""Shared checkpoint row ordering for Qwen gated delta projections."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from rtp_llm.ops import LinearAttentionConfig


def reorder_ba(ts: List[torch.Tensor], la_cfg: LinearAttentionConfig):
    """Reorder ba weight: [head_num_k, group_v*2, hidden] -> [b_all, a_all, hidden].T"""
    if len(ts) != 1 or ts[0].dim() != 2:
        raise ValueError("in_proj_ba transform expects one two-dimensional tensor")
    t = ts[0]
    hidden_size = t.shape[-1]
    head_num_k = la_cfg.linear_num_key_heads
    head_num_v = la_cfg.linear_num_value_heads
    if head_num_v % head_num_k != 0:
        raise ValueError(
            f"linear value heads {head_num_v} must divide into key heads {head_num_k}"
        )
    group_v = head_num_v // head_num_k
    if t.shape[0] != head_num_v * 2:
        raise ValueError(f"in_proj_ba rows must be {head_num_v * 2}, got {t.shape[0]}")
    t = t.reshape(head_num_k, group_v * 2, t.shape[-1])
    b, a = t.split([group_v, group_v], dim=1)
    return torch.cat([b.reshape(-1, hidden_size), a.reshape(-1, hidden_size)], dim=0)


def reorder_qkvz(ts: List[torch.Tensor], la_cfg: LinearAttentionConfig):
    """Reorder qkvz weight: split q,k,v,z per-head then concatenate."""
    if len(ts) != 1 or ts[0].dim() not in (1, 2):
        raise ValueError("in_proj_qkvz transform expects one row-oriented tensor")
    t = ts[0]
    if t.dim() == 1:
        t = t.unsqueeze(1)
    head_num_k = la_cfg.linear_num_key_heads
    head_num_v = la_cfg.linear_num_value_heads
    head_k_dim = la_cfg.linear_key_head_dim
    head_v_dim = la_cfg.linear_value_head_dim
    if head_num_v % head_num_k != 0:
        raise ValueError(
            f"linear value heads {head_num_v} must divide into key heads {head_num_k}"
        )
    group_v = head_num_v // head_num_k
    dim0, dim1 = t.shape
    qkvz_size = head_num_k * head_k_dim * 2 + head_num_v * head_v_dim * 2
    if dim0 == qkvz_size:
        t = t.reshape(head_num_k, head_k_dim * 2 + head_v_dim * group_v * 2, dim1)
        q, k, v, z = torch.split(
            t,
            [head_k_dim, head_k_dim, head_v_dim * group_v, head_v_dim * group_v],
            dim=1,
        )
        return torch.cat(
            [
                q.reshape(-1, dim1),
                k.reshape(-1, dim1),
                v.reshape(-1, dim1),
                z.reshape(-1, dim1),
            ],
            dim=0,
        )
    raise ValueError(f"in_proj_qkvz rows must be {qkvz_size}, got {dim0}")


def reorder_qkvz_scale(
    tensor: torch.Tensor,
    la_cfg: LinearAttentionConfig,
    block_n: int,
) -> torch.Tensor:
    """Reorder a fused per-block QKVZ scale grid without guessing layout."""
    head_k_dim = la_cfg.linear_key_head_dim
    head_v_dim = la_cfg.linear_value_head_dim
    head_num_k = la_cfg.linear_num_key_heads
    head_num_v = la_cfg.linear_num_value_heads
    if head_num_v % head_num_k != 0:
        raise ValueError("linear value heads must be divisible by key heads")
    if head_k_dim % block_n != 0 or head_v_dim % block_n != 0:
        raise ValueError(
            "Qwen3-Next FP8 QKVZ head dimensions must align to the output "
            f"block size {block_n}: key={head_k_dim}, value={head_v_dim}"
        )
    group_v = head_num_v // head_num_k
    per_group = (head_k_dim * 2 + head_v_dim * group_v * 2) // block_n
    expected_rows = head_num_k * per_group
    if tensor.dim() != 2 or tensor.shape[0] != expected_rows:
        raise ValueError(
            f"QKVZ scale grid must have {expected_rows} rows, got {tuple(tensor.shape)}"
        )
    reshaped = tensor.reshape(head_num_k, per_group, tensor.shape[1])
    q_rows = head_k_dim // block_n
    value_rows = head_v_dim * group_v // block_n
    q, k, v, z = torch.split(reshaped, [q_rows, q_rows, value_rows, value_rows], dim=1)
    return torch.cat(
        [
            q.reshape(-1, tensor.shape[1]),
            k.reshape(-1, tensor.shape[1]),
            v.reshape(-1, tensor.shape[1]),
            z.reshape(-1, tensor.shape[1]),
        ],
        dim=0,
    ).contiguous()
