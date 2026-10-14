"""Model-independent TP workspaces, agreed and allocated before graph capture.

Calls and graph replays sharing a workspace must be serialized on the model
compute stream. AG results alias scratch storage; public tensor AG copies the
result, while AG/projection consumes it immediately on that stream.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.custom_all_gather import create_custom_all_gather
from rtp_llm.models_py.distributed.push_reduce_scatter import create_push_reduce_scatter

_PROFILE_COLLECTIVES = os.environ.get("RTP_LLM_PROFILE_MODEL_MODULES", "0") == "1"
_NO_PROFILE_SCOPE = nullcontext()


def _profile_scope(name):
    return (
        torch.profiler.record_function(name)
        if _PROFILE_COLLECTIVES
        else _NO_PROFILE_SCOPE
    )


def _enabled(name: str) -> bool:
    value = os.environ.get(name, "auto").lower()
    if value not in ("0", "1", "auto"):
        raise ValueError(f"{name} must be 0, 1 or auto, got {value!r}")
    return value != "0"


def _key(group, device, width):
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    return group, index, int(width)


@dataclass
class TpCollectiveWorkspace:
    group: dist.ProcessGroup
    device: torch.device
    width: int
    max_m: int
    world_size: int
    push_rs: object = None
    bf16_ag: object = None
    fp8_ag: object = None
    fp8_ag_configured: bool = False

    def contains(self, global_rows: int) -> bool:
        return 0 < global_rows <= self.max_m


_WORKSPACES: dict[tuple, TpCollectiveWorkspace] = {}
_RETIRED_WORKSPACES: list[TpCollectiveWorkspace] = []


def get_tp_collective_workspace(group, device, width):
    if torch.device(device).type != "cuda":
        return None
    return _WORKSPACES.get(_key(group, device, width))


def clear_tp_collective_workspaces():
    """Release mappings only after inference/graphs have stopped using them."""
    _WORKSPACES.clear()
    _RETIRED_WORKSPACES.clear()


def configure_tp_collectives(
    group, device, *, max_m: int, width: int, fp8: bool = False
):
    """Reserve small-token AG/RS capacity for any sequence-parallel model.

    Default auto enables available kernels. Unsupported hardware/topology keeps
    the existing backend collectively; no topology or allocation happens later.
    A larger second model gets new storage; old graph mappings remain alive.
    """
    device = torch.device(device)
    world_size = group.size()
    if world_size == 1 or device.type != "cuda":
        return None
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "TP collectives must be initialized before CUDA Graph capture"
        )
    key = _key(group, device, width)
    existing = _WORKSPACES.get(key)
    # Exchange the entire plan before any rank can skip a workspace rendezvous.
    failure = ""
    try:
        if max_m <= 0 or max_m % world_size or width <= 0:
            raise ValueError(
                "TP collective capacity must be positive and divisible by TP"
            )
        enable_rs = _enabled("RTP_LLM_CUSTOM_REDUCE_SCATTER")
        enable_ag = _enabled("RTP_LLM_CUSTOM_ALL_GATHER")
        enable_fp8_ag = fp8 and _enabled("RTP_LLM_FP8_AG_GEMM")
    except ValueError as exc:
        failure = str(exc)
        enable_rs = enable_ag = enable_fp8_ag = False
    plan = (
        max_m,
        width,
        enable_rs,
        enable_ag,
        enable_fp8_ag,
        (
            None
            if existing is None
            else (
                existing.max_m,
                existing.push_rs is not None,
                existing.bf16_ag is not None,
                existing.fp8_ag is not None,
                existing.fp8_ag_configured,
            )
        ),
        failure,
    )
    plans = [None] * world_size
    dist.all_gather_object(plans, plan, group=group)
    if any(p != plan for p in plans) or any(p[-1] for p in plans):
        raise ValueError(f"TP ranks disagree on collective initialization: {plans}")
    if existing is None or existing.max_m < max_m:
        if existing is not None:
            _RETIRED_WORKSPACES.append(existing)
        existing = TpCollectiveWorkspace(
            group, torch.device("cuda", key[1]), width, max_m, world_size
        )
        if enable_rs:
            existing.push_rs = create_push_reduce_scatter(
                group, existing.device, max_m=max_m, n=width
            )
        if enable_ag:
            existing.bf16_ag = create_custom_all_gather(
                group,
                existing.device,
                max_m=max_m,
                k=width,
                fp8=False,
                staging_only=True,
            )
        _WORKSPACES[key] = existing
    if enable_fp8_ag and not existing.fp8_ag_configured:
        existing.fp8_ag = create_custom_all_gather(
            group,
            existing.device,
            max_m=existing.max_m,
            k=width,
            fp8=True,
            staging_only=True,
        )
        existing.fp8_ag_configured = True
    logging.info(
        "[TP_COLLECTIVES] TP%d width=%d max_m=%d BF16_AG=%s FP8_AG=%s RS=%s",
        world_size,
        width,
        existing.max_m,
        "staging" if existing.bf16_ag is not None else "existing",
        "staging" if existing.fp8_ag is not None else "existing",
        "push" if existing.push_rs is not None else "nccl",
    )
    return existing


def try_reduce_scatter(input_tensor, output, group) -> bool:
    if input_tensor.ndim != 2 or input_tensor.dtype != torch.bfloat16:
        return False
    state = get_tp_collective_workspace(
        group, input_tensor.device, input_tensor.shape[1]
    )
    if (
        state is None
        or state.push_rs is None
        or not state.contains(input_tensor.shape[0])
    ):
        return False
    source = input_tensor.contiguous()
    if source.data_ptr() % 16:
        source = source.clone()
    with _profile_scope("RTP::distributed.reduce_scatter.push"):
        state.push_rs.reduce_scatter(source, output)
    return True


def try_all_gather(input_tensor, group):
    if input_tensor.ndim != 2 or input_tensor.dtype != torch.bfloat16:
        return None
    state = get_tp_collective_workspace(
        group, input_tensor.device, input_tensor.shape[1]
    )
    if (
        state is None
        or state.bf16_ag is None
        or not state.contains(input_tensor.shape[0] * state.world_size)
    ):
        return None
    with _profile_scope("RTP::distributed.all_gather.bf16_staging"):
        # Preserve ordinary all_gather's ownership: a later gather may overwrite
        # scratch while the caller still retains this result as a residual.
        # The staging kernel writes directly to this owned output, avoiding a
        # second copy and launch after communication.
        output = torch.empty(
            (input_tensor.shape[0] * state.world_size, input_tensor.shape[1]),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        return state.bf16_ag.all_gather(
            input_tensor.contiguous(), staging=True, out=output
        )


def try_all_gather_projection(local_input, projection, group):
    """Quantize only at an existing grouped FP8 Linear boundary."""
    if not getattr(projection, "scale_ue8m0", False):
        return None
    state = get_tp_collective_workspace(group, local_input.device, projection.K)
    if (
        state is None
        or state.fp8_ag is None
        or not state.contains(local_input.shape[0] * state.world_size)
    ):
        return None
    from rtp_llm.models_py.distributed.fp8_collective_projection import Fp8Activation

    if isinstance(local_input, Fp8Activation):
        values, wire = local_input.values, local_input.scale_wire
    else:
        if local_input.ndim != 2 or local_input.dtype != torch.bfloat16:
            return None
        values, scales = projection.quantize_input(local_input.contiguous())
        local_m, k = values.shape
        aligned_m = (local_m + 3) // 4 * 4
        wire = scales.as_strided(((k + 511) // 512, aligned_m), (aligned_m, 1))
    with _profile_scope("RTP::distributed.all_gather.fp8_staging"):
        values, scales = state.fp8_ag.all_gather_fp8(values, wire, staging=True)
        return projection.forward_quantized(values, scales)
