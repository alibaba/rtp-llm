"""Qwen4-owned IPC/P2P dispatch; generic FlashInfer runtime stays disabled."""

import logging
from typing import Optional

import torch
import torch.distributed as dist
from rtp_llm.config.qwen4_collective_config import Qwen4CollectiveOptions


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
        self.large_prefill = options.large_prefill
        self._workspace = FlashInferIpcAllReduce(
            group,
            device,
            max_bytes=128 * 1024 * 1024 if self.large_prefill else 32 * 2560 * 2,
        )
        self._workspace.large_prefill = self.large_prefill

    def should_use(self, tensor):
        return (
            not self.disabled
            and tensor.is_cuda
            and tensor.device == self.device
            and tensor.dtype == torch.bfloat16
            and tensor.ndim == 2
            and tensor.shape[1] == 2560
            and tensor.is_contiguous()
            and not (
                torch.cuda.is_current_stream_capturing() and not self.allow_ipc_capture
            )
            and self._workspace.is_buffer_size_sufficient(
                self.world_size, tensor.shape[0], tensor.shape[1], tensor.dtype
            )
        )

    def should_use_cpp_small(self, tensor):
        return (
            self.cpp_small_enabled
            and tensor.ndim == 2
            and 1 <= tensor.shape[0] <= 32
            and tensor.shape[1] == 2560
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
        return self._workspace.all_reduce(tensor)

    def all_reduce_into(self, tensor, output):
        return self._workspace.all_reduce(tensor, output)

    def finish_graph_capture(self):
        self._workspace.finish_graph_capture()

    def destroy(self):
        if not self.disabled:
            self._workspace.destroy()
        self.disabled = True


_communicator: Optional[Qwen4IpcAllReduce] = None
_canonical_p2p = None
_group = None
_options = None


def _capability_error(device, options):
    try:
        if (
            not torch.cuda.is_available()
            or torch.version.hip is not None
            or torch.cuda.get_device_capability(device) != (12, 0)
        ):
            return "requires NVIDIA SM120"
        size = 2 if options.backend == "sm120_ipc" else 8
        if torch.cuda.device_count() != size:
            return f"requires exactly {size} visible GPUs"
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
    """Agree on policy before any rank enters an optional backend constructor."""
    global _communicator, _canonical_p2p, _group, _options
    options = options or Qwen4CollectiveOptions()
    error = None
    try:
        options.validate_topology(parallelism)
    except ValueError as exception:
        error = str(exception)
    state = (options, error, _group is group)
    peers = [None] * dist.get_world_size(group)
    dist.all_gather_object(peers, state, group=group)
    if any(peer[1] is not None for peer in peers):
        raise ValueError(f"Invalid Qwen4 AllReduce configuration: {peers}")
    if any(peer != state for peer in peers):
        raise ValueError(
            "Qwen4 AllReduce policy or initialization state differs across TP ranks"
        )
    if _group is not None:
        if _group is not group or _options != options:
            raise RuntimeError(
                "Destroy the distributed environment before changing Qwen4 AllReduce policy"
            )
        return _communicator
    _group, _options = group, options
    if options.backend == "nccl":
        return None

    reason = _capability_error(device, options)
    dist.all_gather_object(peers, reason, group=group)
    if any(peer is not None for peer in peers):
        logging.warning(
            "Qwen4 %s unavailable; all TP ranks use NCCL: %s", options.backend, peers
        )
        return None
    if options.backend == "sm120_p2p":
        from .canonical_p2p import init_canonical_p2p

        _canonical_p2p = init_canonical_p2p(group, device, requested=True)
    else:
        # The IPC constructor votes on local allocation and registration failures
        # before raising on every rank. Collective failures propagate unchanged.
        from .flashinfer_ipc_all_reduce import IpcInitializationError

        try:
            _communicator = Qwen4IpcAllReduce(group, device, options)
        except IpcInitializationError as error:
            logging.warning(
                "Qwen4 IPC initialization unavailable; all TP ranks use NCCL: %s", error
            )
    return _communicator


def get_qwen4_allreduce():
    return _communicator


def get_canonical_p2p():
    return _canonical_p2p


def destroy_qwen4_allreduce():
    global _communicator, _canonical_p2p, _group, _options
    try:
        if _communicator is not None:
            _communicator.destroy()
        if _canonical_p2p is not None:
            _canonical_p2p.close()
    finally:
        _communicator = _canonical_p2p = _group = _options = None
