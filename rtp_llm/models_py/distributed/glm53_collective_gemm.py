"""GLM Prefill BF16 collective-GEMM, adapted from origin/feat/k3_dev.

The RS workspace and AG pipelines belong to one TP group. Ordinary non-Graph
Prefill also owns the small BF16 symmetric-memory AG path. Other execution
modes and unsupported payloads retain their existing collective dispatch.
"""

import logging
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
    peer_pointers: Optional[torch.Tensor] = None
    data_offset_bytes: int = 0
    reduce_partials: Any = None
    initialization: tuple = ()
    native_all_gather: Any = None


_STATE: Optional[_State] = None
_MIN_TOKENS = 32768
_DEFAULT_MAX_TOKENS = 1 << 20
_ORDERED_RS_BUFFER_BYTES = 64 << 20


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
        if not torch_symm_mem_available or dist.get_backend(group) != "nccl":
            return {"admitted": False}
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
            bool(node[1])
            and torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))
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
            and all(
                torch.cuda.can_device_access_peer(device, rank % local_world)
                for rank in ranks
                if rank != parallelism.world_rank
            )
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


def _native_group_agrees(local, peers):
    return (
        all(peer.get("admitted", False) for peer in peers)
        and all(peer["shared"] == local["shared"] for peer in peers)
        and len({peer["device"] for peer in peers}) == len(peers)
    )


def _configure_native_all_gather(group, device):
    from rtp_llm.models_py.distributed.symm_mem import TorchSymmMemCommunicator

    communicator = TorchSymmMemCommunicator(group, device)
    if communicator.disabled:
        return None
    return _NativeAllGather(communicator)


def _local_collective_features(device, max_tokens, hidden):
    try:
        import torch.distributed._symmetric_memory as symm_mem
    except (ImportError, OSError):
        return None, None, None, 0
    ag = (
        symm_mem
        if callable(getattr(symm_mem, "enable_symm_mem_for_group", None))
        and callable(getattr(torch.ops.symm_mem, "fused_all_gather_matmul", None))
        else None
    )
    if hidden % 8:
        return ag, None, None, 0
    try:
        import deep_gemm

        native = getattr(deep_gemm, "_C", None)
        reduce = getattr(native, "gemm_rs_reduce", None) or getattr(
            native, "bf16_gemm_rs_reduce", None
        )
        size = getattr(deep_gemm, "get_bf16_gemm_rs_workspace_size", None) or getattr(
            native, "get_bf16_gemm_rs_workspace_size", None
        )
        if not all(
            callable(op)
            for op in (
                getattr(deep_gemm, "GemmRSBuffer", None),
                getattr(deep_gemm, "bf16_gemm_rs_nn", None),
                size,
                reduce,
            )
        ):
            return ag, None, None, 0
        required = size(max_tokens, hidden)
        offset = required - max_tokens * hidden * torch.bfloat16.itemsize
        if (
            offset < 128
            or offset % 128
            or required > torch.cuda.mem_get_info(device)[0]
        ):
            return ag, None, None, 0
    except (ImportError, OSError, RuntimeError):
        return ag, None, None, 0
    return ag, deep_gemm, reduce, required


def configure_glm53_collective_gemm(
    group,
    hidden: int,
    parallelism,
    *,
    enable_cuda_graph: bool,
    max_batch_tokens: int = 0
) -> None:
    """Agree on optional collectives on every TP rank before allocating buffers."""
    from rtp_llm.models_py.distributed.collective_utils import exchange_group_state

    global _STATE
    world = group.size()
    requested = max_batch_tokens or _DEFAULT_MAX_TOKENS
    max_tokens = (min(requested, _DEFAULT_MAX_TOKENS) + world - 1) // world * world
    initialization = _initialization_key(parallelism, enable_cuda_graph)
    eager_prefill = initialization[-2:] == ("PREFILL", False)
    local = _native_ag_contract(group, parallelism, enable_cuda_graph)
    configured = _STATE is not None
    existing_matches = not configured or (
        _STATE.group is group
        and _STATE.max_tokens == max_tokens
        and _STATE.hidden == hidden
        and _STATE.initialization == initialization
    )
    ag = deep_gemm = reduce = None
    required = 0
    if local["admitted"] and not configured and max_tokens > 0 and hidden > 0:
        ag, deep_gemm, reduce, required = _local_collective_features(
            local["device"], max_tokens, hidden
        )
    local.update(
        eager_prefill=eager_prefill,
        shape=[hidden, max_tokens],
        configured=configured,
        matches=existing_matches,
        ag=ag is not None,
        rs_bytes=required,
    )
    peers = exchange_group_state(group, local, "glm53_prefill_collectives")
    if any(peer["configured"] != configured or not peer["matches"] for peer in peers):
        raise RuntimeError("GLM collective-GEMM was already initialized differently")
    if configured or not all(peer["eager_prefill"] for peer in peers):
        return
    if (
        hidden <= 0
        or max_tokens <= 0
        or any(peer["shape"] != local["shape"] for peer in peers)
    ):
        raise ValueError(
            "GLM TP ranks must agree on positive hidden size and token capacity"
        )
    native_group = _native_group_agrees(local, peers)
    use_ag = native_group and all(peer["ag"] for peer in peers)
    use_rs = (
        native_group
        and required > 0
        and all(peer["rs_bytes"] == required for peer in peers)
    )
    workspace = peer_pointers = None
    data_offset_bytes = 0
    if use_ag:
        ag.enable_symm_mem_for_group(group.group_name)
    if use_rs:
        workspace = deep_gemm.GemmRSBuffer(
            group, max_m=max_tokens, n=hidden, device=local["device"]
        )
        if workspace.num_bytes != required:
            raise RuntimeError("GLM GEMM/RS workspace size changed after TP agreement")
        data_offset_bytes = required - max_tokens * hidden * torch.bfloat16.itemsize
        peer_pointers = torch.tensor(
            workspace.buffer_ptrs, dtype=torch.uint64, device=workspace.device
        )
        stream = torch.cuda.current_stream(workspace.device)
        completion = getattr(workspace, "_completion_event", None)
        if completion is not None:
            completion.record(stream)
        else:
            workspace._last_stream = stream
    native_all_gather = (
        _configure_native_all_gather(group, local["device"]) if use_ag else None
    )
    _STATE = _State(
        group,
        max_tokens,
        hidden,
        use_ag,
        deep_gemm if use_rs else None,
        workspace,
        peer_pointers=peer_pointers,
        data_offset_bytes=data_offset_bytes,
        reduce_partials=reduce if use_rs else None,
        initialization=initialization,
        native_all_gather=native_all_gather,
    )
    logging.info(
        "GLM Prefill TP%d tokens=%d AG/GEMM=%s GEMM/RS=%s native AG=%s",
        world,
        max_tokens,
        use_ag,
        use_rs,
        native_all_gather is not None,
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


def _ordered_reduce_scatter(tensor: torch.Tensor, group) -> torch.Tensor:
    """Exchange equal BF16 token shards, then add local partials in source-rank order."""
    rows, hidden = tensor.shape
    world = group.size()
    shard_rows = (rows + world - 1) // world
    output = tensor.new_empty((shard_rows, hidden))
    if not shard_rows:
        return output
    # Keep the two BF16 transport buffers and one FP32 accumulator bounded.
    bytes_per_row = hidden * (
        2 * world * tensor.element_size() + torch.float32.itemsize
    )
    chunk_rows = min(shard_rows, max(1, _ORDERED_RS_BUFFER_BYTES // bytes_per_row))
    send_buffer = tensor.new_empty((world * chunk_rows, hidden))
    recv_buffer = torch.empty_like(send_buffer)
    sum_buffer = torch.empty(
        (chunk_rows, hidden), device=tensor.device, dtype=torch.float32
    )
    for offset in range(0, shard_rows, chunk_rows):
        count = min(chunk_rows, shard_rows - offset)
        send = send_buffer[: world * count].view(world, count, hidden)
        received = recv_buffer[: world * count].view_as(send)
        send.zero_()
        for dest in range(world):
            start = dest * shard_rows + offset
            valid = min(count, rows - start)
            if valid > 0:
                send[dest, :valid].copy_(tensor.narrow(0, start, valid))
        dist.all_to_all_single(received, send, group=group)
        accumulator = sum_buffer[:count].zero_()
        for partial in received.unbind(0):
            accumulator.add_(partial)
        output.narrow(0, offset, count).copy_(accumulator)
    return output


def reduce_scatter_glm53(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce TP partials in a destination-independent FP32 addition order.

    BF16 NCCL reduction can change with the destination rank. In GLM sequence
    parallel this makes the same request depend on its batch position. Keep
    BF16 transport/storage, then use GEMM/RS's fixed source-order FP32 reducer.
    Only dedicated GLM Prefill SP callers select this function.
    """
    state = _STATE
    if state is None:
        from rtp_llm.models_py.distributed.collective_torch import (
            Group,
            reduce_scatter_padded,
        )

        return reduce_scatter_padded(tensor, Group.TP)
    world = state.group.size()
    if (
        tensor.ndim != 2
        or tensor.dtype != torch.bfloat16
        or tensor.shape[1] != state.hidden
    ):
        raise ValueError("GLM stable RS expects BF16 [tokens, hidden]")
    if tensor.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("GLM Prefill stable RS does not support CUDA Graph")
    logical = tensor.shape[0]
    physical = (logical + world - 1) // world * world
    if state.reduce_partials is None or physical > state.max_tokens:
        return _ordered_reduce_scatter(tensor, state.group)
    workspace = state.workspace
    if workspace.buffer is None:
        raise RuntimeError("GLM stable RS workspace has been destroyed")
    if tensor.device != workspace.device:
        raise ValueError("GLM stable RS input must be on the workspace device")
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
        and _MIN_TOKENS <= logical_tokens <= _STATE.max_tokens
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
        return None
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
