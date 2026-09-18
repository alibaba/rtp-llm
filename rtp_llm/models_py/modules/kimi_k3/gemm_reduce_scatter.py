"""Fused GEMM/RS for Prefill M >= 512; NCCL for smaller M, Decode and TP16."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.factory.linear.quantized_activation import (
    QuantizedActivation,
)
from rtp_llm.models_py.modules.kimi_k3._collective_gemm import collective_gemm_state_key

_SUPPORTED_WORLD_SIZES = (2, 4, 8)
_MIN_FUSED_M = 512


@dataclass
class _GemmReduceScatterState:
    group: dist.ProcessGroup
    device: torch.device
    world_size: int
    max_m: int
    n: int
    deep_gemm: Optional[Any] = None
    workspace: Optional[Any] = None
    fp8: bool = False
    use_fused: bool = True


_STATES: dict[tuple[dist.ProcessGroup, int], _GemmReduceScatterState] = {}


def _validate_fp8_backend(deep_gemm: Any) -> None:
    if not callable(getattr(deep_gemm, "fp8_gemm_rs_nt", None)):
        raise RuntimeError("installed DeepGEMM lacks the public fp8_gemm_rs_nt API")


def configure_gemm_reduce_scatter(
    group, device, *, max_m, n, fp8=False, use_fused=True
) -> bool:
    """Enable Prefill fusion eligibility; execution also requires M >= 512.

    Decode callers pass use_fused=False. TP16 always uses NCCL.
    """
    use_fused = use_fused and int(group.size()) != 16
    key = collective_gemm_state_key(group, device)
    device = torch.device("cuda", key[1])
    existing = _STATES.get(key)
    if existing is not None:
        if (
            existing.max_m != max_m
            or existing.n != n
            or existing.use_fused != use_fused
        ):
            raise RuntimeError(
                "K3 GEMM/RS was already configured with a different shape or backend"
            )
        if fp8 and existing.use_fused:
            _validate_fp8_backend(existing.deep_gemm)
        existing.fp8 = existing.fp8 or fp8
        return True
    world_size = int(group.size())
    if world_size not in _SUPPORTED_WORLD_SIZES and world_size != 16:
        raise RuntimeError(
            f"GEMM/RS supports TP{(*_SUPPORTED_WORLD_SIZES, 16)}, got TP{world_size}"
        )
    if max_m <= 0 or max_m % world_size or n <= 0:
        raise ValueError(
            "GEMM/RS capacity must be positive with max_m divisible by TP size"
        )
    if not use_fused:
        # No fused workspace: both GEMM dtypes share this NCCL state.
        _STATES[key] = _GemmReduceScatterState(
            group, device, world_size, max_m, n, fp8=True, use_fused=False
        )
        logging.info(
            "[K3_GEMM_REDUCE_SCATTER] nccl TP%d max_m=%d n=%d", world_size, max_m, n
        )
        return True
    deep_gemm = None
    failure_reason = ""
    try:
        import deep_gemm as imported_deep_gemm

        deep_gemm = imported_deep_gemm
        missing = [
            name
            for name in ("GemmRSBuffer", "bf16_gemm_rs_nn")
            if not hasattr(deep_gemm, name)
        ]
        if missing:
            failure_reason = f"DeepGEMM is missing {missing}"
        if fp8:
            _validate_fp8_backend(deep_gemm)
        capability = torch.cuda.get_device_capability(device)
        if capability not in ((10, 0), (10, 3)):
            failure_reason = f"DeepGEMM GEMM/RS requires SM100/SM103, got {capability}"
    except Exception as exc:
        failure_reason = f"failed to import DeepGEMM: {exc}"
    readiness = torch.tensor(
        [int(not failure_reason)], dtype=torch.int32, device=device
    )
    dist.all_reduce(readiness, op=dist.ReduceOp.MIN, group=group)
    if not bool(readiness.item()):
        raise RuntimeError(
            failure_reason or "at least one TP rank cannot use DeepGEMM GEMM/RS"
        )
    workspace = deep_gemm.GemmRSBuffer(group, max_m=max_m, n=n, device=device)
    _STATES[key] = _GemmReduceScatterState(
        group, device, world_size, max_m, n, deep_gemm, workspace, fp8=fp8
    )
    logging.info(
        "[K3_GEMM_REDUCE_SCATTER] fused TP%d max_m=%d n=%d workspace=%.3f GiB",
        world_size,
        max_m,
        n,
        workspace.num_bytes / (1 << 30),
    )
    return True


def gemm_reduce_scatter(
    x: torch.Tensor,
    weight: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    pad_rows: bool,
) -> torch.Tensor:
    """Project with TP1 identity or the configured collective backend."""
    if not x.is_cuda:
        raise TypeError("K3 GEMM/RS requires CUDA input")
    world_size = int(group.size())
    if world_size == 1:
        if isinstance(x, QuantizedActivation):
            if isinstance(weight, torch.Tensor):
                raise TypeError("quantized TP1 GEMM requires a quantized projection")
            return weight.forward_quantized(x.values, x.scales)
        return (
            weight(x)
            if not isinstance(weight, torch.Tensor)
            else torch.matmul(x, weight)
        )
    state = _STATES.get(collective_gemm_state_key(group, x.device))
    if state is None:
        raise RuntimeError("GEMM/RS must be initialized before execution")
    if x.ndim != 2 or (
        x.dtype != torch.bfloat16 and not isinstance(x, QuantizedActivation)
    ):
        raise TypeError(
            "K3 GEMM/RS input must be CUDA BF16 [M,K], got "
            f"shape={tuple(x.shape)} dtype={x.dtype} device={x.device}"
        )
    if x.device != state.device:
        raise ValueError(
            f"K3 GEMM/RS input device {x.device} != workspace {state.device}"
        )

    m = int(x.shape[0])
    # Decide from the input M, before any TP padding (511 -> 512 is not fused).
    use_fused = state.use_fused and state.world_size != 16 and m >= _MIN_FUSED_M
    physical_m = (
        ((m + state.world_size - 1) // state.world_size) * state.world_size
        if pad_rows
        else m
    )
    if physical_m % state.world_size:
        raise ValueError(
            f"K3 GEMM/RS M={physical_m} must be divisible by TP{state.world_size}"
        )
    if not isinstance(weight, torch.Tensor):
        # The public API has no bias epilogue or disable_ue8m0_cast option.
        # K3's Blackwell weights use packed UE8M0; preserve other projections'
        # existing semantics via their own forward method and NCCL.
        if (
            not use_fused
            or not weight.scale_ue8m0
            or getattr(weight, "bias", None) is not None
        ):
            return _fp8_nccl_gemm_reduce_scatter(x, weight, state, physical_m)
        return _fp8_fused_gemm_reduce_scatter(x, weight, state, physical_m)
    if (
        weight.ndim != 2
        or weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or not weight.is_contiguous()
    ):
        raise TypeError(
            "K3 GEMM/ReduceScatter weight must be contiguous CUDA BF16 "
            f"[K,N], got shape={tuple(weight.shape)} dtype={weight.dtype} "
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
    if physical_m > state.max_m:
        raise RuntimeError(
            f"K3 GEMM/RS M={physical_m} exceeds configured max_m={state.max_m}"
        )
    if physical_m != m:
        padded_x = x.new_zeros((physical_m, x.shape[1]))
        padded_x.narrow(0, 0, m).copy_(x)
        x = padded_x
    elif not x.is_contiguous():
        x = x.contiguous()

    output = x.new_empty((physical_m // state.world_size, state.n))
    if physical_m == 0:
        return output
    if not use_fused:
        with torch.profiler.record_function(
            "RTP::kimi_k3.gemm_reduce_scatter.bf16_nccl"
        ):
            partial = torch.mm(x, weight)
            dist.reduce_scatter_tensor(
                output, partial, op=dist.ReduceOp.SUM, group=state.group
            )
        return output
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


__all__ = ["configure_gemm_reduce_scatter", "gemm_reduce_scatter"]


def _fp8_fused_gemm_reduce_scatter(x, projection, state, physical_m):
    """Run the public FP8 GEMM/RS API using the shared BF16/FP8 workspace."""
    if not state.fp8 or projection.K != x.shape[1] or projection.N != state.n:
        raise ValueError("FP8 RS projection does not match the configured workspace")
    if physical_m > state.max_m:
        raise RuntimeError("FP8 RS exceeds the configured token capacity")
    if isinstance(x, QuantizedActivation):
        x = x.pad_rows(physical_m)
    elif physical_m != x.shape[0]:
        padded = x.new_zeros((physical_m, x.shape[1]))
        padded[: x.shape[0]].copy_(x)
        x = padded
    else:
        x = x.contiguous()
    rows = physical_m // state.world_size
    output = torch.empty((rows, state.n), dtype=torch.bfloat16, device=x.device)
    if physical_m == 0:
        return output
    assert state.deep_gemm is not None and state.workspace is not None
    with torch.profiler.record_function("RTP::kimi_k3.gemm_reduce_scatter.fp8_fused"):
        # quantize_input preserves an existing QuantizedActivation's packed
        # UE8M0 values/scales; BF16 inputs use the projection's quantizer.
        state.deep_gemm.fp8_gemm_rs_nt(
            projection.quantize_input(x),
            (projection.weight, projection.weight_scales),
            output,
            state.workspace,
            compiled_dims="nk",
        )
    return output


def _fp8_nccl_gemm_reduce_scatter(x, projection, state, physical_m):
    """Compute one FP8 GEMM, then sum/scatter its BF16 output with NCCL."""
    if not state.fp8 or projection.K != x.shape[1] or projection.N != state.n:
        raise ValueError("FP8 RS projection does not match the configured workspace")
    if physical_m > state.max_m:
        raise RuntimeError("FP8 RS exceeds the configured token capacity")
    output = torch.empty(
        (physical_m // state.world_size, state.n), dtype=torch.bfloat16, device=x.device
    )
    if physical_m == 0:
        return output
    if isinstance(x, QuantizedActivation):
        x = x.pad_rows(physical_m)
    elif physical_m != x.shape[0]:
        padded = x.new_zeros((physical_m, x.shape[1]))
        padded[: x.shape[0]].copy_(x)
        x = padded
    else:
        x = x.contiguous()
    with torch.profiler.record_function("RTP::kimi_k3.gemm_reduce_scatter.fp8_nccl"):
        if isinstance(x, QuantizedActivation):
            partial = projection.forward_quantized(x.values, x.scales)
        else:
            partial = projection(x)
        dist.reduce_scatter_tensor(
            output, partial, op=dist.ReduceOp.SUM, group=state.group
        )
    return output
