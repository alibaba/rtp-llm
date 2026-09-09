"""Fused symmetric-memory AllGather/GEMM for Kimi K3 Prefill."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.collective_torch import Group, get_process_group
from rtp_llm.models_py.distributed.symm_mem import (
    fused_all_gather_fp8_linear,
    fused_all_gather_matmul,
    reserve_fused_all_gather_matmul_workspace,
)
from rtp_llm.models_py.modules.factory.linear.quantized_activation import (
    QuantizedActivation,
)
from rtp_llm.models_py.modules.kimi_k3._collective_gemm import collective_gemm_state_key


@dataclass
class _AllGatherGemmState:
    fp8: bool
    group: dist.ProcessGroup
    device: torch.device
    world_size: int
    max_m: int
    k: int
    dtype: torch.dtype
    workspace_bytes: int


_STATES: dict[tuple[dist.ProcessGroup, int, bool], _AllGatherGemmState] = {}


def configure_all_gather_gemm(group, device, *, max_m, k, dtype, fp8=False) -> bool:
    """Reserve the fused operator workspace, independently of request length."""
    if fp8:
        import torch.distributed._symmetric_memory as symm

        if not callable(getattr(symm, "_pipelined_multi_all_gather_and_consume", None)):
            raise RuntimeError("installed PyTorch lacks the FP8 AG consumer pipeline")
    key = (*collective_gemm_state_key(group, device), fp8)
    device = torch.device("cuda", key[1])
    existing = _STATES.get(key)
    if existing is not None:
        if (max_m, k, dtype) != (existing.max_m, existing.k, existing.dtype):
            raise RuntimeError(
                "K3 AllGather/GEMM was already configured with a different shape"
            )
        return True
    world_size = int(group.size())
    if dtype != torch.bfloat16:
        raise TypeError(f"K3 fused AllGather/GEMM requires BF16 input, got {dtype}")
    if max_m <= 0 or max_m % world_size:
        raise ValueError(
            f"AllGather/GEMM max_m must be positive and divisible by TP{world_size}, got {max_m}"
        )
    if k <= 0:
        raise ValueError(f"AllGather/GEMM K must be positive, got {k}")
    local_m = max_m // world_size
    workspace_bytes = local_m * k * torch.empty((), dtype=dtype).element_size()
    if fp8:
        workspace_bytes = (
            local_m * k + ((k + 511) // 512) * ((local_m + 3) // 4 * 4) * 4
        )
    if world_size > 1:
        reserve_fused_all_gather_matmul_workspace(group, workspace_bytes)
    _STATES[key] = _AllGatherGemmState(
        fp8, group, device, world_size, max_m, k, dtype, workspace_bytes
    )
    logging.info(
        "[K3_ALL_GATHER_GEMM] fused TP%d max_m=%d k=%d fp8=%s workspace=%.3f GiB",
        world_size,
        max_m,
        k,
        fp8,
        workspace_bytes / (1 << 30),
    )
    return True


def all_gather_gemm(
    local_input, weights: Sequence, *, logical_m: int, group: Group = Group.TP
) -> list[torch.Tensor]:
    """Gather and project through the fused operator for every nonempty TP shard."""
    if logical_m < 0:
        raise ValueError(f"logical_m must be non-negative, got {logical_m}")
    if not weights:
        raise ValueError("AG requires at least one projection")
    tensor_weights = [isinstance(w, torch.Tensor) for w in weights]
    if any(tensor_weights) and not all(tensor_weights):
        raise TypeError("AG projections must use a consistent precision policy")
    if not any(tensor_weights) and not isinstance(local_input, QuantizedActivation):
        values, scales = weights[0].quantize_input(local_input)
        m, k = values.shape
        if not weights[0].scale_ue8m0:
            raise ValueError("K3 FP8 pair AG requires UE8M0 activation scales")
        aligned_m = (m + 3) // 4 * 4
        wire = scales.as_strided(((k + 511) // 512, aligned_m), (aligned_m, 1))
        local_input = QuantizedActivation(values, wire)
    if isinstance(local_input, QuantizedActivation):
        return _all_gather_quantized(
            local_input, weights, logical_m=logical_m, group=group
        )
    process_group = get_process_group(group)
    world_size = int(process_group.size())
    if local_input.ndim != 2:
        raise ValueError("AllGather/GEMM input must have [local_M, K] shape")
    physical_m = int(local_input.shape[0]) * world_size
    if logical_m > physical_m:
        raise ValueError(f"logical_m={logical_m} exceeds physical_m={physical_m}")
    # TP1 has no communication; empty shards must not enter a collective kernel.
    if world_size == 1 or physical_m == 0:
        return [torch.matmul(local_input, w)[:logical_m] for w in weights]
    state = _STATES.get(
        (*collective_gemm_state_key(process_group, local_input.device), False)
    )
    if state is None:
        raise RuntimeError("AllGather/GEMM must be initialized before execution")
    if local_input.device != state.device or int(local_input.shape[1]) != state.k:
        raise ValueError("AllGather/GEMM input does not match the configured workspace")
    if physical_m > state.max_m:
        raise RuntimeError(f"AllGather/GEMM M={physical_m} exceeds max_m={state.max_m}")
    if local_input.dtype != state.dtype or not local_input.is_contiguous():
        raise TypeError("AllGather/GEMM input must be contiguous BF16")
    with torch.profiler.record_function("RTP::kimi_k3.all_gather_gemm.fused"):
        _, outputs = fused_all_gather_matmul(
            local_input, weights, process_group, return_gathered=False
        )
    return [
        out if out.shape[0] == logical_m else out.narrow(0, 0, logical_m)
        for out in outputs
    ]


def _all_gather_quantized(local_input, projections, *, logical_m, group):
    process_group = get_process_group(group)
    size = int(process_group.size())
    m, k = local_input.shape
    if logical_m > m * size:
        raise ValueError("logical rows exceed FP8 AG capacity")
    if not projections or any(
        isinstance(p, torch.Tensor) or not p.scale_ue8m0 or p.K != k
        for p in projections
    ):
        raise ValueError("FP8 AG requires compatible UE8M0 projections")
    if size == 1 or m == 0:
        return [
            p.forward_quantized(local_input.values, local_input.scales)[:logical_m]
            for p in projections
        ]
    state = _STATES.get(
        (*collective_gemm_state_key(process_group, local_input.device), True)
    )
    if state is None:
        raise RuntimeError("FP8 AG must be initialized before execution")
    if m * size > state.max_m or k != state.k or local_input.device != state.device:
        raise ValueError("FP8 AG exceeds configured workspace")
    with torch.profiler.record_function("RTP::kimi_k3.all_gather_gemm.fp8_fused"):
        outputs = fused_all_gather_fp8_linear(local_input, projections, process_group)
    return [out[:logical_m] for out in outputs]


__all__ = ["all_gather_gemm", "configure_all_gather_gemm"]
