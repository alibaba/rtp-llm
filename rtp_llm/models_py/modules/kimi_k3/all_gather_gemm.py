"""AG/GEMM dispatch by per-rank input rows on the supported TP8 setup.

Prefill: M < 128 staging, 128 <= M < 4096 direct, M >= 4096 overlap.
Decode: staging at every configured size. BF16 and FP8 share this policy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.collective_torch import Group, get_process_group
from rtp_llm.models_py.distributed.custom_all_gather import (
    STAGING_MAX_LOCAL_M,
    CustomAllGather,
    create_custom_all_gather,
)
from rtp_llm.models_py.distributed.symm_mem import (
    fused_all_gather_fp8_linear,
    fused_all_gather_matmul,
    reserve_fused_all_gather_matmul_workspace,
)
from rtp_llm.models_py.modules.factory.linear.quantized_activation import (
    QuantizedActivation,
)
from rtp_llm.models_py.modules.kimi_k3._collective_gemm import collective_gemm_state_key

# Per-rank physical input rows, including sequence-parallel padding.
_OVERLAP_MIN_LOCAL_M = 4096


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
    use_fused: bool = True
    custom: CustomAllGather | None = None
    overlap_min_local_m: int = _OVERLAP_MIN_LOCAL_M


_STATES: dict[tuple[dist.ProcessGroup, int, bool], _AllGatherGemmState] = {}


def configure_all_gather_gemm(
    group, device, *, max_m, k, dtype, fp8=False, use_fused=True
) -> bool:
    """Initialize all resources before capture; use_fused permits prefill overlap.

    max_m is global output capacity; dispatch uses per-rank physical input rows.
    Custom AG always uses staging for decode; prefill selects by input rows.
    Torch fused prefill or NCCL decode remains the initialization fallback.
    """
    if fp8 and use_fused:
        import torch.distributed._symmetric_memory as symm

        if not callable(getattr(symm, "_pipelined_multi_all_gather_and_consume", None)):
            raise RuntimeError("installed PyTorch lacks the FP8 AG consumer pipeline")
    key = (*collective_gemm_state_key(group, device), fp8)
    device = torch.device("cuda", key[1])
    existing = _STATES.get(key)
    if existing is not None:
        if (max_m, k, dtype, use_fused) != (
            existing.max_m,
            existing.k,
            existing.dtype,
            existing.use_fused
        ):
            raise RuntimeError(
                "K3 AllGather/GEMM was already configured with a different shape or backend"
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
    if world_size > 1 and use_fused:
        reserve_fused_all_gather_matmul_workspace(group, workspace_bytes)
    if not use_fused:
        workspace_bytes = 0
    custom_max_m = (
        min(max_m, (_OVERLAP_MIN_LOCAL_M - 1) * world_size) if use_fused else max_m
    )
    custom = create_custom_all_gather(
        group, device, max_m=custom_max_m, k=k, fp8=fp8, staging_only=not use_fused
    )
    _STATES[key] = _AllGatherGemmState(
        fp8,
        group,
        device,
        world_size,
        max_m,
        k,
        dtype,
        workspace_bytes,
        use_fused,
        custom,
    )
    logging.info(
        "[K3_ALL_GATHER_GEMM] %s TP%d max_global_rows=%d max_local_rows=%d "
        "k=%d fp8=%s overlap_workspace=%.3f GiB",
        (
            ("custom+overlap" if use_fused else "custom")
            if custom is not None
            else ("fused" if use_fused else "nccl")
        ),
        world_size,
        max_m,
        local_m,
        k,
        fp8,
        workspace_bytes / (1 << 30),
    )
    if custom is not None:
        logging.info(
            "[K3_ALL_GATHER_GEMM] per-rank input rows: %s",
            (
                f"prefill staging M<{STAGING_MAX_LOCAL_M}; "
                f"direct {STAGING_MAX_LOCAL_M}<=M<{_OVERLAP_MIN_LOCAL_M}; "
                f"overlap M>={_OVERLAP_MIN_LOCAL_M}"
                if use_fused
                else "decode staging for all configured M"
            ),
        )
    return True


def all_gather_gemm(
    local_input, weights: Sequence, *, logical_m: int, group: Group = Group.TP
) -> list[torch.Tensor]:
    """Gather and project with the backend selected during initialization."""
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
    local_m = int(local_input.shape[0])
    physical_m = local_m * world_size
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
    if state.custom is not None and (
        not state.use_fused or local_m < state.overlap_min_local_m
    ):
        staging = not state.use_fused or local_m < STAGING_MAX_LOCAL_M
        name = "staging" if staging else "direct"
        with torch.profiler.record_function(
            f"RTP::kimi_k3.all_gather_gemm.bf16_{name}"
        ):
            gathered = state.custom.all_gather(local_input, staging=staging)
            return [torch.matmul(gathered, w)[:logical_m] for w in weights]
    if not state.use_fused:
        with torch.profiler.record_function("RTP::kimi_k3.all_gather_gemm.bf16_nccl"):
            gathered = local_input.new_empty((physical_m, state.k))
            dist.all_gather_into_tensor(gathered, local_input, group=process_group)
            return [torch.matmul(gathered, w)[:logical_m] for w in weights]
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
    if state.custom is not None and (
        not state.use_fused or m < state.overlap_min_local_m
    ):
        staging = not state.use_fused or m < STAGING_MAX_LOCAL_M
        name = "staging" if staging else "direct"
        with torch.profiler.record_function(f"RTP::kimi_k3.all_gather_gemm.fp8_{name}"):
            values, scales = state.custom.all_gather_fp8(
                local_input.values, local_input.scale_wire, staging=staging
            )
            return [
                p.forward_quantized(values, scales)[:logical_m] for p in projections
            ]
    if not state.use_fused:
        with torch.profiler.record_function("RTP::kimi_k3.all_gather_gemm.fp8_nccl"):
            values = local_input.values.new_empty((size * m, k))
            groups, aligned_m = local_input.scale_wire.shape
            wire = local_input.scale_wire.new_empty((size * groups, aligned_m))
            dist.all_gather_into_tensor(
                values.view(torch.uint8),
                local_input.values.view(torch.uint8),
                group=process_group
            )
            dist.all_gather_into_tensor(wire, local_input.scale_wire, group=process_group)
            # Each rank pads its scale rows independently. Remove that padding
            # before joining rank-local rows, then restore the GEMM's global alignment.
            scales = wire.new_zeros((groups, (size * m + 3) // 4 * 4))
            scales[:, : size * m].copy_(
                wire.view(size, groups, aligned_m)[:, :, :m]
                .permute(1, 0, 2)
                .reshape(groups, size * m)
            )
            return [
                p.forward_quantized(values, scales.T[: size * m])[:logical_m]
                for p in projections
            ]
    with torch.profiler.record_function("RTP::kimi_k3.all_gather_gemm.fp8_fused"):
        outputs = fused_all_gather_fp8_linear(local_input, projections, process_group)
    return [out[:logical_m] for out in outputs]


__all__ = ["all_gather_gemm", "configure_all_gather_gemm"]
