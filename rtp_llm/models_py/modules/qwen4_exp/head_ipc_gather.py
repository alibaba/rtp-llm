"""Opt-in TP2 Head gather and draft argmax using P2P CUDA IPC.

The native gather kernel preserves BF16 bits while expanding to FP32. The native
argmax kernel reduces each rank's BF16 shard locally and exchanges only 8 bytes
per row across NVLink in a 2-slot ping-pong buffer (eager and CUDA-graph safe).
"""

import ctypes
import threading

import torch
import torch.distributed as dist


class Qwen4HeadIpcGather:
    max_rows = 32

    def __init__(self, group, device, *, max_rows: int = 32):
        from flashinfer.comm import cuda_ipc

        from rtp_llm.ops.compute_ops import rtp_llm_ops

        self.group = group
        self.device = torch.device(device)
        self.rank = dist.get_rank(group)
        self.ops = rtp_llm_ops
        self.max_rows = int(max_rows)
        self._last_stream = None
        self._argmax_last_stream = None
        self._lock = threading.Lock()
        self.closed = False
        assert dist.get_world_size(group) == 2
        with torch.cuda.device(self.device):
            self.signals = cuda_ipc.create_shared_buffer(
                self.ops.head_ipc_meta_size(), group
            )
            self.buffers = cuda_ipc.create_shared_buffer(
                self.max_rows * 124160 * 2, group
            )
            self.argmax_signals = cuda_ipc.create_shared_buffer(
                self.ops.head_ipc_argmax_meta_size(), group
            )
            self.argmax_buffers = cuda_ipc.create_shared_buffer(
                self.ops.head_ipc_argmax_buffer_size(), group
            )
            cuda_ipc.cudart.cudaMemset(
                ctypes.c_void_p(self.signals[self.rank]),
                0,
                self.ops.head_ipc_meta_size(),
            )
            cuda_ipc.cudart.cudaMemset(
                ctypes.c_void_p(self.argmax_signals[self.rank]),
                0,
                self.ops.head_ipc_argmax_meta_size(),
            )
            torch.cuda.synchronize(self.device)
            dist.barrier(group=group)

    def try_gather(self, local):
        if not (
            not self.closed
            and isinstance(local, torch.Tensor)
            and local.is_cuda
            and local.device == self.device
            and local.dtype == torch.bfloat16
            and local.ndim == 2
            and 1 <= local.shape[0] <= self.max_rows
            and local.shape[1] == 124160
            and local.is_contiguous()
            and local.data_ptr() % 16 == 0
            and not torch.cuda.is_current_stream_capturing()
        ):
            return None
        stream = torch.cuda.current_stream(self.device)
        with self._lock:
            if self._last_stream is not None and self._last_stream != stream:
                stream.wait_stream(self._last_stream)
            output = self.ops.head_ipc_gather(
                local, self.buffers, self.signals, self.rank
            )
            self._last_stream = stream
        return output

    def try_argmax(self, local, valid_vocab_size: int = 0):
        argmax_buffers = getattr(self, "argmax_buffers", None)
        argmax_signals = getattr(self, "argmax_signals", None)
        if not (
            not self.closed
            and argmax_buffers
            and argmax_signals
            and isinstance(local, torch.Tensor)
            and local.is_cuda
            and local.device == self.device
            and local.dtype == torch.bfloat16
            and local.ndim == 2
            and 1 <= local.shape[0] <= self.max_rows
            and local.shape[1] == 124160
            and local.is_contiguous()
            and local.data_ptr() % 16 == 0
        ):
            return None
        capturing = torch.cuda.is_current_stream_capturing()
        stream = torch.cuda.current_stream(self.device)
        with self._lock:
            if not capturing:
                last = getattr(self, "_argmax_last_stream", None)
                if last is not None and last != stream:
                    stream.wait_stream(last)
            output = self.ops.head_ipc_argmax(
                local,
                argmax_buffers,
                argmax_signals,
                self.rank,
                int(valid_vocab_size),
            )
            if not capturing:
                self._argmax_last_stream = stream
        return output

    def close(self):
        if self.closed:
            return
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            if dist.is_initialized():
                dist.barrier(group=self.group)
            # Close imported mappings as well as each locally owned allocation.
            if getattr(self, "buffers", None):
                self.ops.head_ipc_close(self.buffers, self.rank)
            if getattr(self, "signals", None):
                self.ops.head_ipc_close(self.signals, self.rank)
            if getattr(self, "argmax_buffers", None):
                self.ops.head_ipc_close(self.argmax_buffers, self.rank)
            if getattr(self, "argmax_signals", None):
                self.ops.head_ipc_close(self.argmax_signals, self.rank)
        self.buffers = self.signals = []
        self.argmax_buffers = self.argmax_signals = []
        self.closed = True


def prepare_head_ipc_gather(config, parallelism, head):
    role = str(getattr(parallelism, "role_type", "")).rsplit(".", 1)[-1]
    if not (
        role in ("DECODE", "PDFUSION")
        and (
            getattr(config, "_qwen4_full_fp8", False)
            or getattr(config, "_qwen4_full_fp8_nvfp4", False)
        )
        and parallelism.tp_size == 2
        and parallelism.dp_size == 1
        and parallelism.local_world_size == 2
        and head is not None
        and tuple(head.weight.shape) == (124160, 2560)
        and head.weight.dtype == torch.float8_e4m3fn
        and head.weight.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(head.weight.device) in ((9, 0), (12, 0))
        and torch.cuda.device_count() == 2
        and torch.cuda.can_device_access_peer(0, 1)
        and torch.cuda.can_device_access_peer(1, 0)
    ):
        return None
    from rtp_llm.models_py.distributed.collective_torch import Group, _get_group

    # Unsupported layouts return before any collective. An allocation/IPC error
    # is a startup failure; do not let one rank silently switch implementations.
    return Qwen4HeadIpcGather(_get_group(Group.TP), head.weight.device, max_rows=256)
