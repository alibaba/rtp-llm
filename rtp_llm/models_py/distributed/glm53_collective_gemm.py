"""GLM Prefill BF16 collective-GEMM, adapted from origin/feat/k3_dev.

The persistent RS workspace and PyTorch AG pipeline belong to one TP group.
Decode DP and small/ragged fallback paths retain the ordinary collectives.
"""

import os
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist


@dataclass
class _State:
    group: Any
    max_tokens: int
    hidden: int
    all_gather: bool
    deep_gemm: Any = None
    workspace: Any = None


_STATE: Optional[_State] = None
_MIN_TOKENS = 32768


def configure_glm53_collective_gemm(group, hidden: int) -> None:
    """Initialize collectively after model/distributed initialization, before forward."""
    global _STATE
    ag = os.environ.get("GLM53_PREFILL_AG_GEMM", "0") == "1"
    rs = os.environ.get("GLM53_PREFILL_GEMM_RS", "0") == "1"
    if not ag and not rs:
        return
    max_tokens = int(os.environ.get("GLM53_COLLECTIVE_MAX_TOKENS", "1048576"))
    if max_tokens <= 0 or max_tokens % group.size():
        raise ValueError(
            "GLM collective token capacity must be positive and TP divisible"
        )
    if group.size() not in (2, 4, 8) or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        raise RuntimeError("GLM collective-GEMM requires SM100/103 and TP2/4/8")
    if _STATE is not None:
        if (
            _STATE.group is not group
            or _STATE.max_tokens != max_tokens
            or _STATE.hidden != hidden
            or _STATE.all_gather != ag
            or (_STATE.workspace is not None) != rs
        ):
            raise RuntimeError(
                "GLM collective-GEMM was already initialized differently"
            )
        return
    import torch.distributed._symmetric_memory as symm_mem

    symm_mem.enable_symm_mem_for_group(group.group_name)
    deep_gemm = workspace = None
    if rs:
        import deep_gemm

        if not hasattr(deep_gemm, "GemmRSBuffer") or not hasattr(
            deep_gemm, "bf16_gemm_rs_nn"
        ):
            raise RuntimeError("GLM GEMM/RS requires the K3 DeepGEMM GEMM/RS build")
        workspace = deep_gemm.GemmRSBuffer(
            group, max_m=max_tokens, n=hidden, device=torch.cuda.current_device()
        )
    _STATE = _State(group, max_tokens, hidden, ag, deep_gemm, workspace)


def canonical_bf16_weight(projection):
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import (
        CudaF16Linear,
    )

    if (
        not isinstance(projection, CudaF16Linear)
        or projection.bias is not None
        or projection.weight.dtype != torch.bfloat16
    ):
        return None
    weight = projection.weight.T
    return weight if weight.ndim == 2 and weight.is_contiguous() else None


def can_fuse_input(projections, logical_tokens: int) -> bool:
    return (
        _STATE is not None
        and _STATE.all_gather
        and logical_tokens >= _MIN_TOKENS
        and all(canonical_bf16_weight(p) is not None for p in projections)
    )


def all_gather_projections(local_x, projections, logical_tokens: int):
    if not can_fuse_input(projections, logical_tokens):
        raise RuntimeError("GLM AG-GEMM called without a compatible initialized path")
    state = _STATE
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("GLM Prefill AG-GEMM does not support CUDA Graph")
    physical = local_x.shape[0] * state.group.size()
    if (
        local_x.ndim != 2
        or local_x.dtype != torch.bfloat16
        or not local_x.is_contiguous()
        or physical > state.max_tokens
        or not 0 <= physical - logical_tokens < state.group.size()
    ):
        raise ValueError("GLM AG-GEMM input does not match its contiguous token layout")
    weights = [canonical_bf16_weight(p) for p in projections]
    if any(
        w.shape[0] != local_x.shape[1] or w.device != local_x.device for w in weights
    ):
        raise ValueError("GLM AG-GEMM projections do not match the local input")
    with torch.profiler.record_function("RTP::glm53.prefill.all_gather_gemm"):
        _, outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            local_x, weights, 0, state.group.group_name, return_A=False
        )
    return tuple(output[:logical_tokens] for output in outputs)


def project_reduce_scatter(x, projection):
    """Return None for ordinary GEMM/RS; never fall back after collective launch."""
    state = _STATE
    if state is None or state.workspace is None or x.shape[0] < _MIN_TOKENS:
        return None
    weight = canonical_bf16_weight(projection)
    if weight is None:
        return None
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("GLM Prefill GEMM/RS does not support CUDA Graph")
    world = state.group.size()
    logical = x.shape[0]
    physical = (logical + world - 1) // world * world
    if physical > state.max_tokens:
        raise ValueError("GLM GEMM/RS exceeds its configured token capacity")
    if (
        x.ndim != 2
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or weight.shape != (x.shape[1], state.hidden)
        or weight.device != x.device
    ):
        raise ValueError("GLM GEMM/RS input/projection shape or device mismatch")
    if physical != logical:
        padded = x.new_zeros((physical, x.shape[1]))
        padded[:logical].copy_(x)
        x = padded
    else:
        x = x.contiguous()
    out = x.new_empty((physical // world, state.hidden))
    with torch.profiler.record_function("RTP::glm53.prefill.gemm_reduce_scatter"):
        state.deep_gemm.bf16_gemm_rs_nn(
            x, weight, out, state.workspace, compiled_dims="nk"
        )
    return out
