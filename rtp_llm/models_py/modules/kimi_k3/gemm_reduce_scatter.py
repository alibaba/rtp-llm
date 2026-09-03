"""DeepGEMM BF16 GEMM/ReduceScatter for Kimi K3 Prefill."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.kimi_k3._collective_gemm import (
    DEFAULT_COLLECTIVE_GEMM_MIN_M,
    collective_gemm_state_key,
    should_use_collective_gemm,
)

DEFAULT_GEMM_REDUCE_SCATTER_MIN_TOKENS = DEFAULT_COLLECTIVE_GEMM_MIN_M

_BACKEND_ENV = "KIMI_K3_GEMM_REDUCE_SCATTER_BACKEND"
_SUPPORTED_WORLD_SIZES = (2, 4, 8)


@dataclass
class _GemmReduceScatterState:
    enabled: bool
    group: dist.ProcessGroup
    device: torch.device
    world_size: int
    max_m: int
    n: int
    deep_gemm: Optional[Any] = None
    workspace: Optional[Any] = None


_STATES: dict[tuple[dist.ProcessGroup, int], _GemmReduceScatterState] = {}


def gemm_reduce_scatter_backend() -> str:
    """Return the process-lifetime backend selected for K3 o_proj + RS."""

    backend = os.environ.get(_BACKEND_ENV, "deepgemm").strip().lower()
    if backend not in ("auto", "deepgemm", "nccl", "off"):
        raise ValueError(
            f"{_BACKEND_ENV} must be auto, deepgemm, nccl, or off; " f"got {backend!r}"
        )
    return backend


def should_use_gemm_reduce_scatter(
    physical_m: int,
    *,
    min_m: int = DEFAULT_GEMM_REDUCE_SCATTER_MIN_TOKENS,
) -> bool:
    """Choose fused GEMM/RS from the current Prefill kernel's physical M."""

    return should_use_collective_gemm(physical_m, min_m=min_m)


def configure_gemm_reduce_scatter(
    group: dist.ProcessGroup,
    device: torch.device,
    *,
    enabled: bool,
    max_m: int,
    n: int,
) -> bool:
    """Create one process-lifetime DeepGEMM workspace per TP group/device."""

    device = torch.device(device)
    key = collective_gemm_state_key(group, device)
    device = torch.device("cuda", key[1])
    existing = _STATES.get(key)
    if existing is not None:
        if existing.max_m != max_m or existing.n != n:
            raise RuntimeError(
                "K3 GEMM/RS was already configured with a different shape: "
                f"existing=(max_m={existing.max_m}, n={existing.n}), "
                f"requested=(max_m={max_m}, n={n})"
            )
        return existing.enabled

    backend = gemm_reduce_scatter_backend()
    requested = (
        enabled
        and should_use_gemm_reduce_scatter(max_m)
        and backend not in ("nccl", "off")
    )
    deep_gemm = None
    local_ready = requested
    failure_reason = ""
    if requested:
        try:
            import deep_gemm as imported_deep_gemm

            deep_gemm = imported_deep_gemm
            required = ("GemmRSBuffer", "bf16_gemm_rs_nn")
            missing = [name for name in required if not hasattr(deep_gemm, name)]
            if missing:
                local_ready = False
                failure_reason = f"DeepGEMM is missing {missing}"
            else:
                capability = torch.cuda.get_device_capability(device)
                if capability not in ((10, 0), (10, 3)):
                    local_ready = False
                    failure_reason = (
                        "DeepGEMM BF16 GEMM/RS requires SM100/SM103, got "
                        f"SM{capability[0]}{capability[1]}"
                    )
        except Exception as exc:  # pragma: no cover - deployment dependent
            local_ready = False
            failure_reason = f"failed to import DeepGEMM: {exc}"

    world_size = int(group.size())
    group_ready = False
    if requested:
        readiness = torch.tensor([int(local_ready)], dtype=torch.int32, device=device)
        readiness_by_rank = readiness.new_empty(world_size)
        dist.all_gather_into_tensor(readiness_by_rank, readiness, group=group)
        group_ready = bool(readiness_by_rank.min().item())
        if not group_ready:
            message = failure_reason or (
                "at least one TP rank cannot use DeepGEMM GEMM/RS"
            )
            if backend == "deepgemm":
                raise RuntimeError(message)
            logging.warning(
                "[K3_GEMM_REDUCE_SCATTER] falling back to NCCL: %s",
                message,
            )

    use_deepgemm = requested and group_ready
    workspace = None
    if use_deepgemm:
        if world_size not in _SUPPORTED_WORLD_SIZES:
            raise RuntimeError(
                "DeepGEMM GEMM/RS supports "
                f"TP{_SUPPORTED_WORLD_SIZES}, got TP{world_size}"
            )
        if max_m <= 0 or max_m % world_size:
            raise ValueError(
                f"GEMM/RS max_m must be positive and divisible by TP{world_size}, "
                f"got {max_m}"
            )
        assert deep_gemm is not None
        workspace = deep_gemm.GemmRSBuffer(
            group,
            max_m=max_m,
            n=n,
            device=device,
        )
        logging.info(
            "[K3_GEMM_REDUCE_SCATTER] enabled BF16 o_proj+RS: "
            "TP%d max_m=%d n=%d "
            "runtime_min_m=%d workspace=%.3f GiB deep_gemm=%s",
            world_size,
            max_m,
            n,
            DEFAULT_GEMM_REDUCE_SCATTER_MIN_TOKENS,
            workspace.num_bytes / (1 << 30),
            getattr(deep_gemm, "__file__", "unknown"),
        )
    else:
        logging.info("[K3_GEMM_REDUCE_SCATTER] using Torch GEMM + NCCL ReduceScatter")

    _STATES[key] = _GemmReduceScatterState(
        enabled=use_deepgemm,
        group=group,
        device=device,
        world_size=world_size,
        max_m=max_m,
        n=n,
        deep_gemm=deep_gemm,
        workspace=workspace,
    )
    return use_deepgemm


def gemm_reduce_scatter(
    x: torch.Tensor,
    weight: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Project row-parallel heads and return the local token shard.

    ``weight`` is the loader's canonical contiguous ``[K, N]`` tensor shared
    by both implementations. Runtime M selects DeepGEMM or Torch/NCCL inside
    this operator; modeling never branches on the selected implementation.
    """

    if not x.is_cuda or x.ndim != 2 or x.dtype != torch.bfloat16:
        raise TypeError(
            "K3 GEMM/RS input must be CUDA BF16 [M,K], got "
            f"shape={tuple(x.shape)} dtype={x.dtype} device={x.device}"
        )
    state = _STATES.get(collective_gemm_state_key(group, x.device))
    if state is None:
        raise RuntimeError("K3 GEMM/RS must be configured before model execution")
    if x.device != state.device:
        raise ValueError(
            f"K3 GEMM/RS input device {x.device} != workspace {state.device}"
        )
    if (
        weight.ndim != 2
        or weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or not weight.is_contiguous()
    ):
        raise TypeError(
            "K3 GEMM/RS weight must be contiguous CUDA BF16 [K,N], got "
            f"shape={tuple(weight.shape)} dtype={weight.dtype} "
            f"device={weight.device} contiguous={weight.is_contiguous()}"
        )
    expected_weight_shape = (int(x.shape[1]), state.n)
    if tuple(weight.shape) != expected_weight_shape:
        raise ValueError(
            f"K3 o_proj weight must be {expected_weight_shape}, "
            f"got {tuple(weight.shape)}"
        )
    if weight.device != x.device:
        raise ValueError(f"K3 o_proj weight device {weight.device} != input {x.device}")

    physical_m = int(x.shape[0])
    if physical_m % state.world_size:
        raise ValueError(
            f"GEMM/RS physical M={physical_m} must be divisible by "
            f"TP{state.world_size}; pad once at the model boundary"
        )
    if physical_m > state.max_m:
        raise RuntimeError(
            f"K3 GEMM/RS M={physical_m} exceeds configured max_m={state.max_m}"
        )

    implementations = (_torch_gemm_reduce_scatter, _deepgemm_reduce_scatter)
    implementation = implementations[
        int(state.enabled and should_use_gemm_reduce_scatter(physical_m))
    ]
    return implementation(x, weight, state, physical_m)


def _torch_gemm_reduce_scatter(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: _GemmReduceScatterState,
    physical_m: int,
) -> torch.Tensor:
    with torch.profiler.record_function("RTP::kimi_k3.gemm_reduce_scatter.torch"):
        partial = torch.mm(x, weight)
        if state.world_size == 1:
            return partial
        output = partial.new_empty((physical_m // state.world_size, state.n))
        dist.reduce_scatter_tensor(
            output,
            partial.contiguous(),
            op=dist.ReduceOp.SUM,
            group=state.group,
        )
        return output


def _deepgemm_reduce_scatter(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: _GemmReduceScatterState,
    physical_m: int,
) -> torch.Tensor:
    if physical_m != x.shape[0]:
        padded = x.new_zeros((physical_m, x.shape[1]))
        padded.narrow(0, 0, x.shape[0]).copy_(x)
        x = padded
    elif not x.is_contiguous():
        x = x.contiguous()

    output = x.new_empty((physical_m // state.world_size, state.n))
    assert state.deep_gemm is not None and state.workspace is not None
    with torch.profiler.record_function("RTP::kimi_k3.gemm_reduce_scatter.fused"):
        state.deep_gemm.bf16_gemm_rs_nn(
            x,
            weight,
            output,
            state.workspace,
            compiled_dims="nk",
        )
        return output


__all__ = [
    "DEFAULT_GEMM_REDUCE_SCATTER_MIN_TOKENS",
    "configure_gemm_reduce_scatter",
    "gemm_reduce_scatter",
    "gemm_reduce_scatter_backend",
    "should_use_gemm_reduce_scatter",
]
