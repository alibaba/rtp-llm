"""Torch symmetric-memory AllGather/GEMM for Kimi K3 Prefill."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather_into,
    get_process_group,
)
from rtp_llm.models_py.distributed.symm_mem import (
    fused_all_gather_matmul,
    reserve_fused_all_gather_matmul_workspace,
)
from rtp_llm.models_py.modules.kimi_k3._collective_gemm import (
    DEFAULT_COLLECTIVE_GEMM_MIN_M,
    collective_gemm_state_key,
    should_use_collective_gemm,
)

DEFAULT_ALL_GATHER_GEMM_MIN_TOKENS = DEFAULT_COLLECTIVE_GEMM_MIN_M


@dataclass
class _AllGatherGemmState:
    enabled: bool
    group: dist.ProcessGroup
    device: torch.device
    world_size: int
    max_m: int
    k: int
    dtype: torch.dtype
    workspace_bytes: int


_STATES: dict[tuple[dist.ProcessGroup, int], _AllGatherGemmState] = {}


def should_use_all_gather_gemm(
    physical_m: int,
    *,
    min_m: int = DEFAULT_ALL_GATHER_GEMM_MIN_TOKENS,
) -> bool:
    """Choose fused AllGather/GEMM from the current physical global M."""

    return should_use_collective_gemm(physical_m, min_m=min_m)


def configure_all_gather_gemm(
    group: dist.ProcessGroup,
    device: torch.device,
    *,
    enabled: bool,
    max_m: int,
    k: int,
    dtype: torch.dtype,
) -> bool:
    """Reserve one process-lifetime Torch AG-GEMM workspace per group/device."""

    device = torch.device(device)
    key = collective_gemm_state_key(group, device)
    device = torch.device("cuda", key[1])
    existing = _STATES.get(key)
    if existing is not None:
        requested_shape = (max_m, k, dtype)
        existing_shape = (existing.max_m, existing.k, existing.dtype)
        if existing_shape != requested_shape:
            raise RuntimeError(
                "K3 AllGather/GEMM was already configured with a different "
                f"shape: existing={existing_shape}, requested={requested_shape}"
            )
        return existing.enabled

    world_size = int(group.size())
    use_fused = enabled and should_use_all_gather_gemm(max_m)
    workspace_bytes = 0
    if use_fused:
        if dtype != torch.bfloat16:
            raise TypeError(
                "K3 fused AllGather/GEMM requires BF16 input, " f"got {dtype}"
            )
        if max_m <= 0 or max_m % world_size:
            raise ValueError(
                "AllGather/GEMM max_m must be positive and divisible by "
                f"TP{world_size}, got {max_m}"
            )
        if k <= 0:
            raise ValueError(f"AllGather/GEMM K must be positive, got {k}")
        itemsize = torch.empty((), dtype=dtype).element_size()
        workspace_bytes = max_m // world_size * k * itemsize
        reserve_fused_all_gather_matmul_workspace(group, workspace_bytes)
        logging.info(
            "[K3_ALL_GATHER_GEMM] enabled Torch symmetric-memory AG-GEMM: "
            "TP%d max_m=%d k=%d runtime_min_m=%d workspace=%.3f GiB",
            world_size,
            max_m,
            k,
            DEFAULT_ALL_GATHER_GEMM_MIN_TOKENS,
            workspace_bytes / (1 << 30),
        )
    else:
        logging.info("[K3_ALL_GATHER_GEMM] using NCCL AllGather + Torch GEMM")

    _STATES[key] = _AllGatherGemmState(
        enabled=use_fused,
        group=group,
        device=device,
        world_size=world_size,
        max_m=max_m,
        k=k,
        dtype=dtype,
        workspace_bytes=workspace_bytes,
    )
    return use_fused


def all_gather_gemm(
    local_input: torch.Tensor,
    weights: Sequence[torch.Tensor],
    *,
    logical_m: int,
    group: Group = Group.TP,
) -> list[torch.Tensor]:
    """All-gather equal token shards, project, and trim padding rows."""

    if logical_m < 0:
        raise ValueError(f"logical_m must be non-negative, got {logical_m}")
    process_group = get_process_group(group)
    world_size = int(process_group.size())
    physical_m = int(local_input.shape[0]) * world_size
    if logical_m > physical_m:
        raise ValueError(f"logical_m={logical_m} exceeds physical_m={physical_m}")

    state = None
    if local_input.is_cuda:
        state = _STATES.get(
            collective_gemm_state_key(process_group, local_input.device)
        )
    use_fused = (
        state is not None and state.enabled and should_use_all_gather_gemm(physical_m)
    )
    if use_fused:
        assert state is not None
        if local_input.device != state.device:
            raise ValueError(
                f"AllGather/GEMM input device {local_input.device} != "
                f"workspace {state.device}"
            )
        if physical_m > state.max_m:
            raise RuntimeError(
                f"AllGather/GEMM M={physical_m} exceeds max_m={state.max_m}"
            )
        if local_input.ndim != 2 or int(local_input.shape[1]) != state.k:
            raise ValueError(
                "AllGather/GEMM input must have configured [local_M, K] "
                f"shape with K={state.k}, got {tuple(local_input.shape)}"
            )
        if local_input.dtype != state.dtype or not local_input.is_contiguous():
            raise TypeError(
                "AllGather/GEMM input must be contiguous "
                f"{state.dtype}, got dtype={local_input.dtype} "
                f"contiguous={local_input.is_contiguous()}"
            )
        with torch.profiler.record_function("RTP::kimi_k3.all_gather_gemm.fused"):
            _, outputs = fused_all_gather_matmul(
                local_input,
                weights,
                process_group,
                return_gathered=False,
            )
    else:
        with torch.profiler.record_function("RTP::kimi_k3.all_gather_gemm.all_gather"):
            gathered = all_gather_into(
                local_input,
                local_input.new_empty((physical_m, *local_input.shape[1:])),
                group,
            )
            gathered = gathered.narrow(0, 0, logical_m)
        with torch.profiler.record_function("RTP::kimi_k3.all_gather_gemm.gemm"):
            outputs = [torch.matmul(gathered, weight) for weight in weights]

    trimmed = []
    for output in outputs:
        if output.shape[0] < logical_m:
            raise ValueError(
                "projection output has fewer rows than logical_m: "
                f"output={tuple(output.shape)}, logical_m={logical_m}"
            )
        trimmed.append(
            output if output.shape[0] == logical_m else output.narrow(0, 0, logical_m)
        )
    return trimmed


__all__ = [
    "DEFAULT_ALL_GATHER_GEMM_MIN_TOKENS",
    "all_gather_gemm",
    "configure_all_gather_gemm",
    "should_use_all_gather_gemm",
]
