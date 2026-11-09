"""Capability-selected SM120 IPC/P2P behind the generic collective API."""

import logging
import socket
from dataclasses import replace
from typing import Optional

import torch
import torch.distributed as dist

from rtp_llm.config.qwen4_collective_config import Qwen4CollectiveOptions
from rtp_llm.models_py.distributed.collective_utils import exchange_group_state

_SMALL_BYTES = 32 * 2560 * 2
_PREFILL_BYTES = 16384 * 2560 * 2
_PREFILL_ROWS = (4093, 4094, 4095, 4096, 8192, 12288, 16384)


class Qwen4IpcAllReduce:
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
        # Decode graph pointers always belong to this fixed small workspace.
        if options.large_prefill is True:
            self.prepare_large_prefill()

    @property
    def large_prefill(self):
        return self._prefill_workspace is not None

    def prepare_large_prefill(self):
        if (
            self._prefill_attempted
            or self._prefill_policy is False
            or _workspaces_frozen
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        self._prefill_attempted = True
        # Two IPC buffers plus rank metadata. Leave at least 1 GiB for the
        # engine, and cap this optional workspace at 1% of device capacity.
        required = 2 * _PREFILL_BYTES + 8 * 1024 * 1024
        try:
            free, total = torch.cuda.mem_get_info(self.device)
            admitted = required <= total // 100 and free >= required + 1024**3
        except Exception:
            admitted = False
        if not all(exchange_group_state(self.group, admitted, "prefill_budget")):
            return
        from .flashinfer_ipc_all_reduce import (
            FlashInferIpcAllReduce,
            IpcInitializationError,
        )

        try:
            workspace = FlashInferIpcAllReduce(
                self.group, self.device, max_bytes=_PREFILL_BYTES
            )
        except IpcInitializationError as error:
            logging.warning("Large-prefill IPC unavailable; using NCCL: %s", error)
            return
        workspace.large_prefill = True
        self._prefill_workspace = workspace

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
        if tensor.shape[0] in _PREFILL_ROWS and not capturing:
            self.prepare_large_prefill()
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


_communicator: Optional[Qwen4IpcAllReduce] = None
_canonical_p2p = None
_group = None
_options = None
_configured_options = Qwen4CollectiveOptions()
_workspaces_frozen = False
_capture_active = False


def configure_collectives(py_env_configs):
    """Install parsed overrides in each spawned rank, without model-name gates."""
    global _configured_options, _workspaces_frozen
    options = Qwen4CollectiveOptions(
        backend=py_env_configs.qwen4_all_reduce_backend,
        large_prefill=py_env_configs.qwen4_all_reduce_large_prefill,
    )
    options.validate_topology(py_env_configs.parallelism_config)
    if _group is not None and options != _options:
        raise RuntimeError("Destroy the distributed environment before changing policy")
    _configured_options = options
    _workspaces_frozen = False


def freeze_workspaces():
    """Serving may only use workspaces prepared during startup/warmup."""
    global _workspaces_frozen
    _workspaces_frozen = True


def _capability_error(device, options):
    try:
        if (
            not torch.cuda.is_available()
            or torch.version.hip is not None
            or torch.cuda.get_device_capability(device) != (12, 0)
        ):
            return "requires NVIDIA SM120"
        size = 2 if options.backend == "sm120_ipc" else 8
        if torch.cuda.device_count() < size:
            return f"requires at least {size} visible GPUs"
        if not all(
            peer == device.index
            or torch.cuda.can_device_access_peer(device.index, peer)
            for peer in range(size)
        ):
            return "direct peer access unavailable"
        if options.backend == "sm120_ipc":
            from flashinfer.comm import cuda_ipc, vllm_ar

            if not hasattr(cuda_ipc, "cudart") or not hasattr(
                vllm_ar, "init_custom_ar"
            ):
                return "FlashInfer IPC API unavailable"
    except Exception as error:
        return str(error)
    return None


def init_qwen4_allreduce(group, device, parallelism, options=None):
    """Negotiate one backend before entering optional GPU setup on any rank."""
    global _communicator, _canonical_p2p, _group, _options
    options = options or _configured_options
    error = None
    try:
        options.validate_topology(parallelism)
    except ValueError as exception:
        error = str(exception)
    state = [options.backend, options.large_prefill, error, _group is group]
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

    candidate = options
    if options.backend == "auto":
        # Only TP2 IPC is automatically admitted. TP8 P2P remains an explicit
        # experiment; unsupported DP/PP/CP topologies keep their current backend.
        candidate = replace(options, backend="sm120_ipc", large_prefill=None)
        try:
            candidate.validate_topology(parallelism)
        except ValueError:
            candidate = Qwen4CollectiveOptions("nccl")
        else:
            candidate = replace(candidate, large_prefill=options.large_prefill)
    _group, _options = group, options
    if options.backend == "nccl":
        return None

    reason = (
        None if candidate.backend == "nccl" else _capability_error(device, candidate)
    )
    state = [candidate.backend, socket.gethostname(), reason]
    peers = exchange_group_state(group, state, "capabilities")
    if any(
        peer[0] == "nccl"
        or peer[0] != state[0]
        or peer[1] != state[1]
        or peer[2] is not None
        for peer in peers
    ):
        logging.warning(
            "AllReduce %s unavailable; all TP ranks use NCCL: %s",
            candidate.backend,
            peers,
        )
        return None
    if candidate.backend == "sm120_p2p":
        from .canonical_p2p import init_canonical_p2p

        _canonical_p2p = init_canonical_p2p(group, device, requested=True)
    else:
        from .flashinfer_ipc_all_reduce import IpcInitializationError

        try:
            _communicator = Qwen4IpcAllReduce(group, device, candidate)
        except IpcInitializationError as error:
            logging.warning(
                "IPC initialization unavailable; all TP ranks use NCCL: %s", error
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


def get_qwen4_allreduce():
    return _communicator


def get_canonical_p2p():
    return _canonical_p2p


def destroy_qwen4_allreduce():
    global _communicator, _canonical_p2p, _group, _options
    global _workspaces_frozen
    try:
        if _communicator is not None:
            _communicator.destroy()
        if _canonical_p2p is not None:
            _canonical_p2p.close()
    finally:
        abort_graph_capture()
        _communicator = _canonical_p2p = _group = _options = None
        _workspaces_frozen = False
