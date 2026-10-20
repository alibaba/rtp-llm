"""Opt-in TP2 Head gather using the measured vLLM IPC synchronization protocol.

The native kernel preserves BF16 bits while expanding to FP32. One dedicated
staging allocation is serialized across eager stream handoffs. CUDA graph
capture retains NCCL; captured local Head projection is unaffected.
"""

import ctypes
import os
import threading

import torch
import torch.distributed as dist


class Qwen4HeadIpcGather:
    def __init__(self, group, device):
        from flashinfer.comm import cuda_ipc

        from rtp_llm.ops.compute_ops import rtp_llm_ops

        self.group = group
        self.device = torch.device(device)
        self.rank = dist.get_rank(group)
        self.ops = rtp_llm_ops
        self._last_stream = None
        self._lock = threading.Lock()
        self.closed = False
        assert dist.get_world_size(group) == 2
        with torch.cuda.device(self.device):
            self.signals = cuda_ipc.create_shared_buffer(
                self.ops.head_ipc_meta_size(), group
            )
            self.buffers = cuda_ipc.create_shared_buffer(32 * 124160 * 2, group)
            cuda_ipc.cudart.cudaMemset(
                ctypes.c_void_p(self.signals[self.rank]),
                0,
                self.ops.head_ipc_meta_size(),
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
            and 1 <= local.shape[0] <= 32
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

    def close(self):
        if self.closed:
            return
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            if dist.is_initialized():
                dist.barrier(group=self.group)
            # Close imported mappings as well as each locally owned allocation.
            self.ops.head_ipc_close(self.buffers, self.rank)
            self.ops.head_ipc_close(self.signals, self.rank)
        self.buffers = self.signals = []
        self.closed = True


def prepare_head_ipc_gather(config, parallelism, head):
    if os.environ.get("QWEN4_HEAD_IPC_GATHER", "0") != "1":
        return None
    role = str(getattr(parallelism, "role_type", "")).rsplit(".", 1)[-1]
    if not (
        role == "DECODE"
        and getattr(config, "_qwen4_full_fp8_nvfp4", False)
        and parallelism.tp_size == 2
        and parallelism.dp_size == 1
        and parallelism.local_world_size == 2
        and head is not None
        and tuple(head.weight.shape) == (124160, 2560)
        and head.weight.dtype == torch.float8_e4m3fn
        and head.weight.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(head.weight.device) == (12, 0)
        and torch.cuda.device_count() == 2
        and torch.cuda.can_device_access_peer(0, 1)
        and torch.cuda.can_device_access_peer(1, 0)
    ):
        return None
    from rtp_llm.models_py.distributed.collective_torch import Group, _get_group

    # Unsupported layouts return before any collective. An allocation/IPC error
    # is a startup failure; do not let one rank silently switch implementations.
    return Qwen4HeadIpcGather(_get_group(Group.TP), head.weight.device)
