"""DeepGEMM BF16 and FP8 GEMM/ReduceScatter for Kimi K3 Prefill."""

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


@dataclass
class _GemmReduceScatterState:
    group: dist.ProcessGroup
    device: torch.device
    world_size: int
    max_m: int
    n: int
    deep_gemm: Optional[Any] = None
    workspace: Optional[Any] = None


_STATES: dict[tuple[dist.ProcessGroup, int], _GemmReduceScatterState] = {}


def _validate_fp8_workspace(deep_gemm: Any, workspace: Any) -> None:
    required = ("_mapping_handle", "_launch_lock", "_last_stream", "_barrier")
    if (
        getattr(workspace, "_data_offset_bytes", None) != 128
        or any(not hasattr(workspace, key) for key in required)
        or not hasattr(getattr(deep_gemm, "_C", None), "bf16_gemm_rs_reduce")
    ):
        raise RuntimeError("installed DeepGEMM lacks the FP8 peer-output RS ABI")


def configure_gemm_reduce_scatter(group, device, *, max_m, n, fp8=False) -> bool:
    """Create the fused DeepGEMM workspace; unavailable backends fail at startup."""
    key = collective_gemm_state_key(group, device)
    device = torch.device("cuda", key[1])
    existing = _STATES.get(key)
    if existing is not None:
        if existing.max_m != max_m or existing.n != n:
            raise RuntimeError(
                "K3 GEMM/RS was already configured with a different shape"
            )
        if fp8:
            _validate_fp8_workspace(existing.deep_gemm, existing.workspace)
        return True
    world_size = int(group.size())
    if world_size not in _SUPPORTED_WORLD_SIZES:
        raise RuntimeError(
            f"DeepGEMM GEMM/RS supports TP{_SUPPORTED_WORLD_SIZES}, got TP{world_size}"
        )
    if max_m <= 0 or max_m % world_size or n <= 0:
        raise ValueError(
            "GEMM/RS capacity must be positive with max_m divisible by TP size"
        )
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
    if fp8:
        _validate_fp8_workspace(deep_gemm, workspace)
    _STATES[key] = _GemmReduceScatterState(
        group, device, world_size, max_m, n, deep_gemm, workspace
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
    """Run fused GEMM/RS for every Prefill size, including padded small M."""
    if not x.is_cuda:
        raise TypeError("K3 GEMM/RS requires CUDA input")
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
        return _fp8_remote_gemm_reduce_scatter(x, weight, state, physical_m)
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


def _fp8_remote_gemm_reduce_scatter(x, projection, state, physical_m):
    """Write FP8 GEMM BF16 outputs directly to destination-owned source slots.

    Destination-sized GEMMs reuse DeepGEMM's existing TMA output descriptor.
    No full local partial or post-GEMM peer copy is materialized. The published
    BF16 source-slot layout and both barriers match GemmRSBuffer's ABI.
    """
    if projection.K != x.shape[1] or projection.N != state.n:
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
    workspace = state.workspace
    rows = physical_m // state.world_size
    output = torch.empty((rows, state.n), dtype=torch.bfloat16, device=x.device)
    if physical_m == 0:
        return output
    if workspace is None or workspace._mapping_handle is None:
        raise RuntimeError("FP8 RS requires a live DeepGEMM symmetric workspace")
    if workspace._data_offset_bytes != 128:
        raise RuntimeError("unsupported DeepGEMM GEMM/RS workspace ABI")
    slot_offset = workspace._data_offset_bytes // 2 + workspace.rank * rows * state.n
    # Materialize all peer views before the first collective write.
    peers = [
        workspace._mapping_handle.get_buffer(
            dst, (rows, state.n), torch.bfloat16, storage_offset=slot_offset
        )
        for dst in range(state.world_size)
    ]
    with workspace._launch_lock, torch.cuda.device(workspace.device):
        stream = torch.cuda.current_stream(workspace.device)
        if workspace._last_stream is not None and workspace._last_stream != stream:
            stream.wait_stream(workspace._last_stream)
        with torch.profiler.record_function(
            "RTP::kimi_k3.gemm_reduce_scatter.fp8_remote"
        ):
            for step in range(state.world_size):
                dst = (workspace.rank + step) % state.world_size
                if isinstance(x, QuantizedActivation):
                    shard = x.narrow_rows(dst * rows, rows)
                    projection.forward_quantized(
                        shard.values, shard.scales, out=peers[dst]
                    )
                else:
                    projection(x.narrow(0, dst * rows, rows), out=peers[dst])
            workspace._barrier(0)
            state.deep_gemm._C.bf16_gemm_rs_reduce(
                output, workspace.buffer, state.world_size, physical_m, state.n
            )
            workspace._barrier(1)
            workspace._last_stream = stream
    return output
