"""Marlin W16A16 MoE weight pack utilities (DCU/lightop kernel layout).

The packed layout matches vllm-dcu's `fused_experts_impl_w16a16_marlin`:
- w13 (gate+up) packs from [E, 2N, K] to [E, K/16, 2N*16]
- w2  (down)    packs from [E, K, N]  to [E, N/16, K*16]

Reference: vllm-dcu/model_executor/layers/fused_moe/marlin_quant.py
"""

from typing import Optional

import numpy as np
import torch


def _get_weight_perm() -> torch.Tensor:
    """Per-warp 16x32 permutation (kept identical to vllm-dcu marlin_quant.py)."""
    perm = []
    for i in range(64):
        for col in range(2):
            cur_col = (i % 16) * 2 + col
            for row in range(4):
                cur_row = (i // 16) * 4 + row
                perm.append(cur_row * 32 + cur_col)
    return torch.from_numpy(np.array(perm))


_WEIGHT_PERM_CACHE: Optional[torch.Tensor] = None


def _weight_perm() -> torch.Tensor:
    global _WEIGHT_PERM_CACHE
    if _WEIGHT_PERM_CACHE is None:
        _WEIGHT_PERM_CACHE = _get_weight_perm()
    return _WEIGHT_PERM_CACHE


def _pack_single_2d(w: torch.Tensor) -> torch.Tensor:
    """Pack one expert from [out, in] -> [in/16, out*16] in marlin layout."""
    # Match vllm-dcu: input [size_n, size_k], operate on transposed [size_k, size_n].
    w_kn = w.t().contiguous()
    size_k, size_n = w_kn.shape
    k_tile, n_tile = 16, 32
    assert size_k % k_tile == 0, f"K={size_k} not divisible by {k_tile}"
    assert size_n % n_tile == 0, f"N={size_n} not divisible by {n_tile}"

    q = w_kn.reshape(size_k // k_tile, k_tile, size_n // n_tile, n_tile)
    q = q.permute(0, 2, 1, 3)
    q = q.reshape(size_k // k_tile, size_n * k_tile)

    perm = _weight_perm().to(q.device)
    q = q.reshape(-1, perm.numel())[:, perm].reshape(size_k // k_tile, size_n * k_tile)
    return q.contiguous()


def pack_marlin_w16a16_per_expert(weight: torch.Tensor) -> torch.Tensor:
    """Pack [E, out, in] -> [E, in/16, out*16] expert-by-expert.

    Each expert is processed independently to minimise peak memory.
    """
    assert weight.dim() == 3, f"expected 3D MoE weight, got {weight.shape}"
    assert weight.dtype in (torch.float16, torch.bfloat16), (
        f"marlin W16A16 requires fp16/bf16, got {weight.dtype}"
    )

    print(f"marlin pack tensor with shape: {weight.shape}")
    num_experts = weight.size(0)
    first = _pack_single_2d(weight[0])
    out = first.new_empty((num_experts,) + first.shape)
    out[0].copy_(first)
    del first

    for i in range(1, num_experts):
        out[i].copy_(_pack_single_2d(weight[i]))
    return out


def shapes_match_marlin_packed(w1: torch.Tensor, w2: torch.Tensor) -> bool:
    """Return True iff (w1, w2) shapes already look like marlin packed MoE weights.

    Mirrors vllm-dcu's `_is_marlin_w16a16_packed` so the runtime auto-dispatches.
    """
    if w1.dim() != 3 or w2.dim() != 3 or w1.size(0) != w2.size(0):
        return False
    k_div16 = w1.size(1)
    if k_div16 <= 0 or w1.size(2) % 16 != 0:
        return False
    twoN = w1.size(2) // 16
    if twoN % 2 != 0:
        return False
    N = twoN // 2
    if w2.size(2) % 16 != 0:
        return False
    K = k_div16 * 16
    if w2.size(2) // 16 != K:
        return False
    if w2.size(1) * 16 != N:
        return False
    return True
