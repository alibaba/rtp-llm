"""GLM Prefill BF16 collective-GEMM, adapted from origin/feat/k3_dev.

The RS workspace and AG pipelines belong to one TP group. Ordinary non-Graph
Prefill also owns the small BF16 symmetric-memory AG path. Other execution
modes and unsupported payloads retain their existing collective dispatch.
"""

import os
import socket
import threading
from dataclasses import dataclass
from pathlib import Path
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
    gemm_reduce_scatter: bool = False
    stable_reduce_scatter: bool = False
    peer_pointers: Optional[torch.Tensor] = None
    data_offset_bytes: int = 0
    reduce_partials: Any = None
    initialization: tuple = ()
    native_all_gather: Any = None


_STATE: Optional[_State] = None
_MIN_TOKENS = 32768


class _NativeAllGather:
    """One model-owned buffer; its next writer waits for the previous copy-out.

    Only configured non-Graph Prefill uses this owner. The lock orders host
    submissions; the event also orders different CUDA streams after the lock
    has been released. Outputs belong to the caller, never to the symm buffer.
    """

    def __init__(self, communicator):
        self.communicator = communicator
        self.lock = threading.Lock()
        self.completion = torch.cuda.Event()
        self.failed = False
        self.completion.record(torch.cuda.current_stream(communicator.device))

    def gather(self, tensor):
        with self.lock, torch.cuda.device(self.communicator.device):
            if self.failed:
                raise RuntimeError("GLM native AllGather owner previously failed")
            stream = torch.cuda.current_stream(self.communicator.device)
            try:
                stream.wait_event(self.completion)
                tensor.record_stream(stream)
                source = tensor
                if tensor.data_ptr() % 16:
                    # Storage offsets can differ across ranks. Stage locally
                    # instead of choosing a different collective on this rank.
                    source = tensor.clone()
                    source.record_stream(stream)
                output = self.communicator.all_gather(source)
                if output is None:
                    raise RuntimeError("GLM native AllGather unexpectedly declined")
                # all_gather includes out.copy_ from the reusable symm buffer.
                self.completion.record(stream)
            except BaseException:
                # A partially submitted collective is not a local fallback.
                self.failed = True
                raise
            return output.view(
                self.communicator.world_size * tensor.shape[0], *tensor.shape[1:]
            )


def _initialization_key(parallelism, enable_cuda_graph):
    return (
        *(
            int(getattr(parallelism, field))
            for field in (
                "world_size",
                "world_rank",
                "local_world_size",
                "local_rank",
                "tp_size",
                "tp_rank",
                "dp_size",
                "pp_size",
            )
        ),
        str(parallelism.role_type).rsplit(".", 1)[-1],
        bool(enable_cuda_graph),
    )


def _native_ag_contract(group, parallelism, enable_cuda_graph):
    """Publish a local startup verdict, including the actual TP membership.

    Failures here are voted down before anyone constructs the communicator.
    Rendezvous exceptions from the communicator itself deliberately propagate.
    """
    from rtp_llm.models_py.distributed.rank_layout import Group, RankLayout
    from rtp_llm.models_py.distributed.symm_mem import torch_symm_mem_available

    try:
        world = dist.get_world_size(group)
        member = dist.get_rank(group)
        ranks = [dist.get_global_rank(group, rank) for rank in range(world)]
        expected = RankLayout.from_parallelism_config(parallelism).group_of(
            Group.TP, parallelism.world_rank
        )
        local_world = int(parallelism.local_world_size)
        role = str(parallelism.role_type).rsplit(".", 1)[-1]
        # The boot identity prevents repeated/container-local hostnames from
        # being mistaken for CUDA-IPC locality. An unavailable identity declines.
        node = [
            socket.gethostname(),
            Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            os.getuid(),
        ]
        device = torch.cuda.current_device()
        admitted = (
            torch_symm_mem_available
            and bool(node[1])
            and role == "PREFILL"
            and not enable_cuda_graph
            and world in (2, 4, 8)
            and world == parallelism.tp_size
            and parallelism.world_size == dist.get_world_size()
            and ranks == expected
            and local_world > 0
            and min(ranks) // local_world == max(ranks) // local_world
            and ranks[member] == parallelism.world_rank == dist.get_rank()
            and member == parallelism.tp_rank
            and device == parallelism.local_rank == parallelism.world_rank % local_world
        )
        return {
            "admitted": bool(admitted),
            "shared": [
                ranks,
                expected,
                int(parallelism.world_size),
                local_world,
                int(parallelism.dp_size),
                int(parallelism.pp_size),
                role,
                bool(enable_cuda_graph),
                node,
            ],
            "device": device,
        }
    except (OSError, ValueError, RuntimeError, AttributeError, IndexError):
        return {"admitted": False}


def _configure_native_all_gather(group, parallelism, enable_cuda_graph):
    from rtp_llm.models_py.distributed.collective_utils import exchange_group_state
    from rtp_llm.models_py.distributed.symm_mem import TorchSymmMemCommunicator

    local = _native_ag_contract(group, parallelism, enable_cuda_graph)
    peers = exchange_group_state(group, local, "glm53_native_all_gather")
    if not (
        all(peer.get("admitted", False) for peer in peers)
        and all(peer["shared"] == local["shared"] for peer in peers)
        and len({peer["device"] for peer in peers}) == len(peers)
    ):
        return None
    communicator = TorchSymmMemCommunicator(group, local["device"])
    if communicator.disabled:
        return None
    return _NativeAllGather(communicator)


def configure_glm53_collective_gemm(
    group, hidden: int, parallelism, *, enable_cuda_graph: bool
) -> None:
    """Initialize collectively after model/distributed initialization, before forward."""
    global _STATE
    ag = os.environ.get("GLM53_PREFILL_AG_GEMM", "0") == "1"
    rs = os.environ.get("GLM53_PREFILL_GEMM_RS", "0") == "1"
    stable_rs = os.environ.get("GLM53_PREFILL_STABLE_RS", "0") == "1"
    if not ag and not rs and not stable_rs:
        return
    initialization = _initialization_key(parallelism, enable_cuda_graph)
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
            or _STATE.gemm_reduce_scatter != rs
            or _STATE.stable_reduce_scatter != stable_rs
            or _STATE.initialization != initialization
        ):
            raise RuntimeError(
                "GLM collective-GEMM was already initialized differently"
            )
        return
    import torch.distributed._symmetric_memory as symm_mem

    symm_mem.enable_symm_mem_for_group(group.group_name)
    deep_gemm = workspace = None
    peer_pointers = None
    data_offset_bytes = 0
    reduce_partials = None
    if rs or stable_rs:
        import deep_gemm

        if not hasattr(deep_gemm, "GemmRSBuffer") or not hasattr(
            deep_gemm, "bf16_gemm_rs_nn"
        ):
            raise RuntimeError("GLM GEMM/RS requires the K3 DeepGEMM GEMM/RS build")
        workspace = deep_gemm.GemmRSBuffer(
            group, max_m=max_tokens, n=hidden, device=torch.cuda.current_device()
        )
        if stable_rs:
            # Both supported workspace ABIs store the rank-major BF16 payload
            # after the control region. Do not depend on a private Python field
            # removed in DeepGEMM 2.8.2.
            data_offset_bytes = (
                workspace.num_bytes - max_tokens * hidden * torch.bfloat16.itemsize
            )
            if data_offset_bytes < 128 or data_offset_bytes % 128:
                raise RuntimeError("Unsupported GLM stable RS workspace layout")
            reduce_partials = getattr(deep_gemm._C, "gemm_rs_reduce", None)
            if reduce_partials is None:
                reduce_partials = getattr(deep_gemm._C, "bf16_gemm_rs_reduce", None)
            if not callable(reduce_partials):
                raise RuntimeError("DeepGEMM is missing the GEMM/RS reduction binding")
            peer_pointers = torch.tensor(
                workspace.buffer_ptrs, dtype=torch.uint64, device=workspace.device
            )
            # The first model forward may use another stream. Reuse the same
            # handoff contract as GEMM/RS for pointer-table initialization.
            stream = torch.cuda.current_stream(workspace.device)
            completion = getattr(workspace, "_completion_event", None)
            if completion is not None:
                completion.record(stream)
            else:
                workspace._last_stream = stream
    # Preserve existing DeepGEMM workspace initialization and group order.
    # This optional communicator is owned by this GLM Prefill state, never by
    # the permanently disabled global AllReduce/symm communicator.
    native_all_gather = (
        _configure_native_all_gather(group, parallelism, enable_cuda_graph)
        if ag
        else None
    )
    _STATE = _State(
        group,
        max_tokens,
        hidden,
        ag,
        deep_gemm,
        workspace,
        gemm_reduce_scatter=rs,
        stable_reduce_scatter=stable_rs,
        peer_pointers=peer_pointers,
        data_offset_bytes=data_offset_bytes,
        reduce_partials=reduce_partials,
        initialization=initialization,
        native_all_gather=native_all_gather,
    )


def all_gather(tensor: torch.Tensor, group, *, role: Optional[str] = None):
    """GLM-only small BF16 AG; preserve ordinary and CP typed fallbacks.

    Admission is fixed collectively at model initialization with CUDA Graph
    disabled. Only the existing ordinary-Prefill metadata activates it; Decode,
    target verification and calls outside a GLM forward retain their original
    path. A manual capture in an admitted non-Graph model is an error, not an
    attempt to choose a different route on this rank. Mixed-rank capture is
    outside this contract.
    """
    from rtp_llm.models_py.distributed import collective_torch
    from rtp_llm.models_py.modules.dsv4.forward_metadata import metadata_cache

    state = _STATE
    native = state.native_all_gather if state is not None else None
    if (
        native is not None
        and group == collective_torch.Group.TP
        and metadata_cache() is not None
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "GLM native AllGather requires configured non-Graph Prefill"
            )
        communicator = native.communicator
        if (
            tensor.ndim > 0
            and tensor.numel() > 0
            and tensor.device == communicator.device
            # The op may select 16-byte vectors from the gathered output.
            # Each rank's shard must match that alignment. The owner stages
            # misaligned pointers without changing the collective route.
            and tensor.numel() * tensor.element_size() % 16 == 0
            and communicator.should_torch_symm_mem_allgather(tensor)
        ):
            with torch.profiler.record_function("RTP::glm53.prefill.native_all_gather"):
                return native.gather(tensor)
    if role is not None:
        # CP roles already have a separate pynccl/typed-payload dispatcher.
        # Keep that exact fallback when the native BF16 route declines.
        from rtp_llm.models_py.distributed.pynccl_cp import all_gather as cp_all_gather

        return cp_all_gather(tensor, group=group, role=role)
    return collective_torch.all_gather(tensor, group)


def all_gather_trim(tensor: torch.Tensor, logical_tokens: int, group):
    gathered = all_gather(tensor, group)
    if not 0 <= logical_tokens <= gathered.shape[0]:
        raise ValueError("logical token count exceeds the gathered rows")
    return gathered.narrow(0, 0, logical_tokens)


def reduce_scatter_glm53(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce TP partials in a destination-independent FP32 addition order.

    BF16 NCCL reduction can change with the destination rank. In GLM sequence
    parallel this makes the same request depend on its batch position. Keep
    BF16 transport/storage, then use GEMM/RS's fixed source-order FP32 reducer.
    Only dedicated GLM Prefill SP callers select this function.
    """
    state = _STATE
    if state is None or not state.stable_reduce_scatter:
        from rtp_llm.models_py.distributed.collective_torch import (
            Group,
            reduce_scatter_padded,
        )

        return reduce_scatter_padded(tensor, Group.TP)
    workspace = state.workspace
    world = state.group.size()
    if workspace.buffer is None:
        raise RuntimeError("GLM stable RS workspace has been destroyed")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("GLM Prefill stable RS does not support CUDA Graph")
    if (
        tensor.ndim != 2
        or tensor.dtype != torch.bfloat16
        or tensor.device != workspace.device
        or tensor.shape[1] != state.hidden
    ):
        raise ValueError("GLM stable RS expects BF16 [tokens, hidden] on its device")
    logical = tensor.shape[0]
    physical = (logical + world - 1) // world * world
    if physical > state.max_tokens:
        raise ValueError("GLM stable RS exceeds its configured token capacity")
    workspace_begin = workspace.buffer.data_ptr()
    if workspace_begin <= tensor.data_ptr() < workspace_begin + workspace.num_bytes:
        raise ValueError("GLM stable RS input must not alias its workspace")
    output = tensor.new_empty((physical // world, state.hidden))
    if physical == 0:
        return output
    if physical != logical:
        source = tensor.new_zeros((physical, state.hidden))
        source[:logical].copy_(tensor)
    else:
        source = tensor.contiguous()
    from rtp_llm.models_py.triton_kernels.common.glm53_reduce_scatter import (
        publish_glm53_partials,
    )

    local_numel = output.numel()
    block = 8192
    # Round up separately for every peer: an odd local row count has a tail
    # on each peer, not just at the end of the combined input allocation.
    grid = (world * ((local_numel + block - 1) // block),)
    with torch.profiler.record_function("RTP::glm53.prefill.stable_reduce_scatter"):
        with workspace._launch_lock, torch.cuda.device(workspace.device):
            stream = torch.cuda.current_stream(workspace.device)
            completion = getattr(workspace, "_completion_event", None)
            if completion is not None:
                stream.wait_event(completion)
            elif (
                workspace._last_stream is not None and workspace._last_stream != stream
            ):
                stream.wait_stream(workspace._last_stream)
            source.record_stream(stream)
            publish_glm53_partials[grid](
                source,
                state.peer_pointers,
                local_numel,
                world,
                workspace.rank,
                state.data_offset_bytes,
                block,
            )
            # These release/acquire-system barriers publish remote stores and
            # prevent the next GEMM/RS or RS from overwriting unconsumed partials.
            workspace._barrier(0)
            state.reduce_partials(
                output, workspace.buffer, world, physical, state.hidden
            )
            workspace._barrier(1)
            if completion is not None:
                completion.record(stream)
            else:
                workspace._last_stream = stream
    return output


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


def _all_gather_weights(local_x, projections, logical_tokens: int):
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
    return weights


def all_gather_projections(local_x, projections, logical_tokens: int):
    weights = _all_gather_weights(local_x, projections, logical_tokens)
    state = _STATE
    with torch.profiler.record_function("RTP::glm53.prefill.all_gather_gemm"):
        _, outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            local_x, weights, 0, state.group.group_name, return_A=False
        )
    return tuple(output[:logical_tokens] for output in outputs)


def packed_kda_projections(x, packed_weight, widths):
    """Project gathered tokens once; consumers keep views of the packed layout."""
    with torch.profiler.record_function("RTP::glm53.prefill.packed_gemm"):
        packed = torch.mm(x, packed_weight)
    return packed[:, : sum(widths)].split(widths, dim=-1)


def all_gather_packed_kda_projections(
    local_x, projections, packed_weight, widths, logical_tokens
):
    """One packed GEMM per AG chunk; return row-strided views without repacking."""
    _all_gather_weights(local_x, projections, logical_tokens)
    state = _STATE
    with torch.profiler.record_function("RTP::glm53.prefill.packed_ag_gemm"):
        _, outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            local_x, [packed_weight], 0, state.group.group_name, return_A=False
        )
    packed = outputs[0][:logical_tokens, : sum(widths)]
    return packed.split(widths, dim=-1)


def all_gather_kda_projections(local_x, projections, logical_tokens: int):
    """Compute replicated low-rank weights only for local tokens.

    QKV/beta retain fused AG/GEMM. Separate gathers produce contiguous low-rank
    outputs directly, avoiding packing and unpacking copies on the large path.
    """
    if len(projections) != 4:
        raise ValueError("Local KDA requires QKV, beta, f_a and g_a projections")
    _all_gather_weights(local_x, projections, logical_tokens)
    qkv_proj, beta_proj, f_a_proj, g_a_proj = projections
    with torch.profiler.record_function("RTP::glm53.prefill.local_low_rank"):
        local_low = (f_a_proj(local_x), g_a_proj(local_x))
        physical = local_x.shape[0] * _STATE.group.size()
        low = []
        for local in local_low:
            gathered = local.new_empty((physical, local.shape[1]))
            dist.all_gather_into_tensor(gathered, local, group=_STATE.group)
            low.append(gathered[:logical_tokens])
    projected_qkv, beta_input = all_gather_projections(
        local_x, (qkv_proj, beta_proj), logical_tokens
    )
    return projected_qkv, beta_input, *low


def project_reduce_scatter(x, projection):
    """Return None for ordinary GEMM/RS; never fall back after collective launch."""
    state = _STATE
    if state is None or not state.gemm_reduce_scatter or x.shape[0] < _MIN_TOKENS:
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
