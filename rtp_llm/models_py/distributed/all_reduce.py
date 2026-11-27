"""Select the reduction policy shared by all tensor-parallel ranks."""

import logging
import socket
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Optional

import torch
import torch.distributed as dist

from rtp_llm.config.collective_config import CollectiveConfig
from rtp_llm.models_py.distributed.collective_utils import exchange_group_state

_ALL_REDUCE_WORKSPACE_BYTES = 16 * 1024 * 1024
_SMALL_BYTES = 32 * 2560 * 2
_PREFILL_ROWS = (4093, 4094, 4095, 4096, 8192, 12288, 16384)


def _all_reduce_chunks(source: torch.Tensor, dest: torch.Tensor, limit: int):
    """Slice matching logical rows without flattening a strided tensor into a full copy."""
    if source.numel() <= limit:
        yield source, dest
        return
    row_size = source.numel() // source.size(0)
    if row_size > limit:
        for row in range(source.size(0)):
            yield from _all_reduce_chunks(source[row], dest[row], limit)
    else:
        rows = limit // row_size
        for start in range(0, source.size(0), rows):
            yield source[start : start + rows], dest[start : start + rows]


def all_reduce_batch_invariant(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    sum_gathered_ranks: Callable[[torch.Tensor, torch.Tensor, int], None],
    inplace: bool = True,
) -> torch.Tensor:
    """Sum like-shaped BF16 tensors in TP group-rank order and round once per element."""
    if tensor.dtype != torch.bfloat16 or tensor.layout != torch.strided:
        raise ValueError("batch-invariant all-reduce requires a strided BF16 tensor")
    if group is None:
        raise ValueError("batch-invariant all-reduce requires a TP process group")
    world_size = dist.get_world_size(group)
    if world_size < 1:
        raise ValueError(
            "batch-invariant all-reduce requires membership in the TP group"
        )
    if world_size == 1 or tensor.numel() == 0:
        return tensor if inplace else tensor.clone()

    if (
        inplace
        and not tensor.is_contiguous()
        and torch._debug_has_internal_overlap(tensor) == 1
    ):
        raise ValueError(
            "in-place batch-invariant all-reduce requires non-overlapping elements"
        )
    output = tensor if inplace else torch.empty_like(tensor)
    # Reserve space for one rank-major gather and at most one local staging
    # buffer. Chunk boundaries are independent of strides on individual ranks.
    capacity = max(
        1, _ALL_REDUCE_WORKSPACE_BYTES // ((world_size + 1) * tensor.element_size())
    )
    capacity = min(capacity, tensor.numel())
    gathered = tensor.new_empty(world_size * capacity)
    staging = None
    if not (tensor.is_contiguous() and output.is_contiguous()):
        staging = tensor.new_empty(capacity)

    with torch.cuda.device_of(tensor):
        for source, dest in _all_reduce_chunks(tensor, output, capacity):
            count = source.numel()
            if source.is_contiguous():
                send = source.view(-1)
            else:
                send = staging[:count]
                send.view_as(source).copy_(source)
            # Synchronous c10d semantics order NCCL and the caller's current
            # CUDA stream in both directions, including workspace reuse.
            dist.all_gather_into_tensor(
                gathered[: world_size * count], send, group=group
            )
            result = dest.view(-1) if dest.is_contiguous() else staging[:count]
            sum_gathered_ranks(gathered, result, world_size)
            if not dest.is_contiguous():
                dest.copy_(result.view_as(dest))
    return output


class IpcAllReduce:
    def __init__(self, group, device, options):
        from .flashinfer_ipc_all_reduce import FlashInferIpcAllReduce

        self.group = group
        self.device = device
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.disabled = False
        self.cpp_small_enabled = True
        self.allow_ipc_capture = False
        self._prefill_policy = options.large_prefill
        self._prefill_workspace = None
        self._prefill_attempted = False
        self._workspace = FlashInferIpcAllReduce(group, device, max_bytes=_SMALL_BYTES)

    @property
    def large_prefill(self):
        return self._prefill_workspace is not None

    def prepare_large_prefill(self, model_configs, engine_config):
        if self._prefill_attempted or self._prefill_policy is False:
            return
        self._prefill_attempted = True
        max_bytes = 0
        try:
            runtime = engine_config.runtime_config
            max_tokens = runtime.fifo_scheduler_config.max_batch_tokens_size
            supported = any(
                config.hidden_size == 2560 and config.compute_dtype == torch.bfloat16
                for config in model_configs
            )
            rows = max(
                (rows for rows in _PREFILL_ROWS if rows <= max_tokens), default=0
            )
            max_bytes = rows * 2560 * 2 if supported else 0
            required = 2 * max_bytes + 8 * 1024**2 + self._workspace.ops.meta_size()
            free, total = torch.cuda.mem_get_info(self.device)
            # Keep the runtime floor used by MemoryEvaluationHelper, plus any
            # explicitly sized KV cache. Automatic KV sizing sees this actual
            # allocation through cudaMemGetInfo; do not deduct it a second time.
            runtime_bytes = max(
                runtime.reserve_runtime_mem_mb * 1024**2,
                2048 * 1024**2,
                int(total * 0.05),
            )
            kv_bytes = max(0, engine_config.kv_cache_config.kv_cache_mem_mb) * 1024**2
            admitted = (
                max_bytes > 0
                and required <= total // 100
                and free >= required + runtime_bytes + kv_bytes
            )
        except Exception:
            admitted = False
        peers = exchange_group_state(
            self.group, [max_bytes, admitted], "prefill_budget"
        )
        if not all(peer == [max_bytes, True] for peer in peers):
            logging.info("Large-prefill IPC startup admission declined: %s", peers)
            return
        from .flashinfer_ipc_all_reduce import (
            FlashInferIpcAllReduce,
            IpcInitializationError,
        )

        try:
            workspace = FlashInferIpcAllReduce(
                self.group, self.device, max_bytes=max_bytes
            )
        except IpcInitializationError as error:
            logging.warning("Large-prefill IPC unavailable; using NCCL: %s", error)
            return
        workspace.large_prefill = True
        self._prefill_workspace = workspace
        logging.info("Large-prefill IPC prepared: capacity_bytes=%s", max_bytes)

    def should_use(self, tensor):
        if (
            self.disabled
            or not tensor.is_cuda
            or tensor.device != self.device
            or tensor.dtype != torch.bfloat16
            or tensor.ndim != 2
            or tensor.shape[1] != 2560
            or not tensor.is_contiguous()
        ):
            return False
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and not self.allow_ipc_capture:
            return False
        workspace = self._workspace_for(tensor)
        return workspace is not None and workspace.is_buffer_size_sufficient(
            self.world_size, tensor.shape[0], tensor.shape[1], tensor.dtype
        )

    def _workspace_for(self, tensor):
        if tensor.shape[0] <= 32:
            return self._workspace
        if not torch.cuda.is_current_stream_capturing():
            return self._prefill_workspace
        return None

    def should_use_cpp_small(self, tensor):
        return (
            self.cpp_small_enabled
            and tensor.ndim == 2
            and 1 <= tensor.shape[0] <= 32
            and self.should_use(tensor)
        )

    def should_use_registered_graph_input(self, tensor):
        return (
            self.allow_ipc_capture
            and self._workspace.registered_graph_inputs
            and torch.cuda.is_current_stream_capturing()
            and self.should_use(tensor)
        )

    def all_reduce(self, tensor):
        return self._workspace_for(tensor).all_reduce(tensor)

    def all_reduce_into(self, tensor, output):
        return self._workspace_for(tensor).all_reduce(tensor, output)

    def finish_graph_capture(self):
        self._workspace.finish_graph_capture()

    def destroy(self):
        if not self.disabled:
            self._workspace.destroy()
            if self._prefill_workspace is not None:
                self._prefill_workspace.destroy()
        self.disabled = True


_communicator: Optional[IpcAllReduce] = None
_canonical_p2p = None
_batch_invariant_sum = None
_group = None
_options = None
_capture_active = False


def prepare_collective_workspaces(model_configs, engine_config):
    """Prepare persistent storage after weights, before warmup and KV sizing."""
    if _communicator is not None:
        _communicator.prepare_large_prefill(model_configs, engine_config)


def _capability_error(device, options):
    try:
        if (
            not torch.cuda.is_available()
            or torch.version.hip is not None
            or torch.cuda.get_device_capability(device) != (12, 0)
        ):
            return "requires NVIDIA SM120"
        size = 2 if options.all_reduce_backend == "sm120_ipc" else 8
        if torch.cuda.device_count() < size:
            return f"requires at least {size} visible GPUs"
        if not all(
            peer == device.index
            or torch.cuda.can_device_access_peer(device.index, peer)
            for peer in range(size)
        ):
            return "direct peer access unavailable"
        if options.all_reduce_backend == "sm120_ipc":
            # Read the effective allocator state, including runtime settings.
            # Legacy CUDA IPC cannot export expandable (VMM) graph segments.
            settings = torch.cuda.memory._snapshot()["allocator_settings"]
            if settings["expandable_segments"]:
                return "IPC graph registration does not support expandable_segments"
            from flashinfer.comm import cuda_ipc, vllm_ar

            if not hasattr(cuda_ipc, "cudart") or not hasattr(
                vllm_ar, "init_custom_ar"
            ):
                return "FlashInfer IPC API unavailable"
    except Exception as error:
        return str(error)
    return None


def init_all_reduce(group, device, parallelism, options: CollectiveConfig):
    """Negotiate one backend before entering optional GPU setup on any rank."""
    global _communicator, _canonical_p2p, _batch_invariant_sum, _group, _options
    error = None
    batch_invariant_sum = None
    try:
        options.validate_topology(parallelism)
    except ValueError as exception:
        error = str(exception)
    if error is None and options.batch_invariant and _group is None:
        try:
            from rtp_llm.models_py.triton_kernels.peer_all_reduce import (
                sum_gathered_ranks,
            )

            batch_invariant_sum = partial(
                all_reduce_batch_invariant, sum_gathered_ranks=sum_gathered_ranks
            )
        except Exception as exception:
            # A rank missing the selected backend must still join policy agreement.
            error = str(exception)
    state = [options.all_reduce_backend, options.large_prefill, error, _group is group]
    if options.batch_invariant:
        # Keep the ordinary four-field handshake compatible with existing ranks.
        state.extend([True, _group is not None, _options == options])
    peers = exchange_group_state(group, state, "policy")
    if any(peer[2] is not None for peer in peers):
        raise ValueError(f"Invalid AllReduce configuration: {peers}")
    if any(peer != state for peer in peers):
        raise ValueError(
            "AllReduce policy or initialization state differs across TP ranks"
        )
    if _group is not None:
        if _group is not group or _options != options:
            raise RuntimeError(
                "Destroy the distributed environment before changing policy"
            )
        return _communicator

    if batch_invariant_sum is not None:
        _batch_invariant_sum = batch_invariant_sum
        _group, _options = group, options
        logging.info("AllReduce policy: TP BF16 SUM uses batch-invariant reduction")
        return None

    candidate = options
    if options.all_reduce_backend == "auto":
        # Only TP2 IPC is automatically admitted. TP8 P2P remains an explicit
        # experiment; unsupported DP/PP/CP topologies keep their current backend.
        candidate = replace(options, all_reduce_backend="sm120_ipc", large_prefill=None)
        try:
            candidate.validate_topology(parallelism)
        except ValueError:
            candidate = CollectiveConfig("nccl")
        else:
            candidate = replace(candidate, large_prefill=options.large_prefill)
    communicator = None
    canonical_p2p = None
    fallback_reason = None
    if options.all_reduce_backend != "nccl":
        reason = (
            None
            if candidate.all_reduce_backend == "nccl"
            else _capability_error(device, candidate)
        )
        state = [candidate.all_reduce_backend, socket.gethostname(), reason]
        peers = exchange_group_state(group, state, "capabilities")
        if any(
            peer[0] == "nccl"
            or peer[0] != state[0]
            or peer[1] != state[1]
            or peer[2] is not None
            for peer in peers
        ):
            fallback_reason = f"unsupported topology or capabilities: {peers}"
            if options.all_reduce_backend != "auto":
                logging.warning(
                    "AllReduce %s unavailable; all TP ranks use NCCL: %s",
                    options.all_reduce_backend,
                    fallback_reason,
                )
        elif candidate.all_reduce_backend == "sm120_p2p":
            from .canonical_p2p import init_canonical_p2p

            canonical_p2p = init_canonical_p2p(group, device, requested=True)
            if canonical_p2p is None:
                fallback_reason = "P2P initialization unavailable"
        else:
            from .flashinfer_ipc_all_reduce import IpcInitializationError

            try:
                communicator = IpcAllReduce(group, device, candidate)
            except IpcInitializationError as error:
                fallback_reason = str(error)
                logging.warning(
                    "IPC initialization unavailable; all TP ranks use NCCL: %s", error
                )

    # Only a completed setup or an agreed NCCL fallback is idempotent. Leave
    # the state untouched when negotiation or optional construction raises.
    _communicator, _canonical_p2p = communicator, canonical_p2p
    _group, _options = group, options
    logging.info(
        "AllReduce policy: requested_backend=%s, effective_backend=%s, "
        "large_prefill=%s, fallback_reason=%s",
        options.all_reduce_backend,
        (
            candidate.all_reduce_backend
            if communicator is not None or canonical_p2p is not None
            else "nccl"
        ),
        options.large_prefill,
        fallback_reason,
    )
    return _communicator


def begin_graph_capture():
    global _capture_active
    if _capture_active:
        raise RuntimeError("Nested collective graph capture is unsupported")
    _capture_active = True
    if _communicator is not None:
        _communicator.allow_ipc_capture = True


def abort_graph_capture():
    global _capture_active
    _capture_active = False
    if _communicator is not None:
        _communicator.allow_ipc_capture = False


def end_graph_capture():
    # Reset local admission even if pointer registration fails. A failed
    # captured graph must never replay or fall back to eager execution.
    abort_graph_capture()
    if _communicator is not None:
        _communicator.finish_graph_capture()


def get_all_reduce():
    return _communicator


def get_canonical_p2p():
    return _canonical_p2p


def get_batch_invariant_sum(tensor):
    if _batch_invariant_sum is not None and tensor.dtype == torch.bfloat16:
        return _batch_invariant_sum
    return None


def destroy_all_reduce():
    global _communicator, _canonical_p2p, _batch_invariant_sum, _group, _options
    try:
        if _communicator is not None:
            _communicator.destroy()
        if _canonical_p2p is not None:
            _canonical_p2p.close()
    finally:
        abort_graph_capture()
        _communicator = _canonical_p2p = _batch_invariant_sum = _group = _options = None
