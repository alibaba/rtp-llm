"""TP2 BF16 sums using TensorRT and vLLM PCIe kernels on SM120."""

import functools
import hashlib
import os
from pathlib import Path

import torch
import torch.distributed as dist

_communicator = None


@functools.lru_cache(maxsize=1)
def _module():
    from flashinfer.jit import env, gen_jit_spec

    source = Path(__file__).with_name("sm120_ipc_allreduce.cu").resolve()
    # FlashInfer retains Ninja files across temporary runfiles and redeploys.
    cuda_cflags = ["-gencode=arch=compute_120a,code=sm_120a"]
    source_id = hashlib.sha256(
        str(source).encode()
        + b"\0"
        + source.read_bytes()
        + b"\0"
        + "\0".join(cuda_cflags).encode()
    ).hexdigest()[:16]
    return gen_jit_spec(
        f"rtp_sm120_ipc_allreduce_v1_{source_id}",
        [source],
        extra_include_paths=[str(env.FLASHINFER_CSRC_DIR)],
        extra_cuda_cflags=cuda_cflags,
    ).build_and_load()


@functools.lru_cache(maxsize=1)
def _trt_module():
    from flashinfer.jit import env, gen_jit_spec

    return gen_jit_spec(
        "rtp_sm120_trt_allreduce_v1",
        [
            env.FLASHINFER_CSRC_DIR / n
            for n in (
                "trtllm_allreduce.cu",
                "trtllm_allreduce_fusion.cu",
                "trtllm_moe_allreduce_fusion.cu",
            )
        ],
        extra_cuda_cflags=["-gencode=arch=compute_120a,code=sm_120a"],
    ).build_and_load()


class Sm120IpcAllReduce:
    """One serialized CUDA stream and two local peers per communicator.

    IPC buffers and peer pointers stay fixed across all captured graphs. Copying
    each input into staging also keeps graph-pool allocations private. Call
    close on both ranks after all graph replays have finished.
    """

    def __init__(self, group):
        from flashinfer.comm.cuda_ipc import create_shared_buffer

        self.group = group
        self.rank = dist.get_rank(group)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.max_bytes = 4096 * 5120 * 2
        self.module = _module()
        self.signals = create_shared_buffer(self.module.signal_size(), group)
        self.module.zero_signal(self.signals[self.rank])
        self.buffers = create_shared_buffer(self.max_bytes, group)
        self.pointers = torch.tensor(
            self.buffers + [0] * 6, device=self.device, dtype=torch.int64
        )
        # TensorRT's Lamport path avoids the staging copy for decode. Keep
        # vLLM's smaller workspace for the measured prefill range above it.
        self.trt_module = _trt_module()
        lamport_size = 2 * 512 * 5120 * 2

        def align(n):
            return (n + (1 << 21) - 1) // (1 << 21) * (1 << 21)

        self.trt_buffers = [
            create_shared_buffer(align(n), group)
            for n in (lamport_size, 2 * 256 * 4, lamport_size * 3)
        ]
        self.trt_module.trtllm_lamport_initialize(
            self.trt_buffers[2][self.rank], align(lamport_size * 3) // 2, torch.float16
        )
        self.trt_flags = torch.tensor(
            [0, 0, 0, lamport_size, 0], device=self.device, dtype=torch.int32
        )
        self.trt_pointers = torch.tensor(
            [p for pair in self.trt_buffers for p in pair]
            + [self.trt_flags.data_ptr()],
            device=self.device,
            dtype=torch.int64,
        )
        torch.cuda.synchronize()
        dist.barrier(group)
        # Exercise IPC reads/writes before graph capture, on this process group.
        for rows in (1, 513, 4096):
            probe = torch.full(
                (rows, 5120), self.rank + 1, device=self.device, dtype=torch.bfloat16
            )
            expected = probe.clone()
            dist.all_reduce(expected, group=group)
            actual = self.all_reduce(probe)
            torch.cuda.synchronize()
            if not torch.equal(actual, expected):
                raise RuntimeError("SM120 IPC allreduce preflight differs from NCCL")

    def is_supported(self, tensor):
        return (
            self.signals is not None
            and tensor.device == self.device
            and tensor.dtype == torch.bfloat16
            and tensor.ndim == 2
            and tensor.shape[1] == 5120
            and 0 < tensor.shape[0] <= 4096
            and tensor.is_contiguous()
        )

    def all_reduce(self, tensor, out=None):
        if out is None:
            out = torch.empty_like(tensor)
        if tensor.shape[0] <= 512:
            self.trt_module.trtllm_allreduce_fusion(
                tensor,
                2,
                self.rank,
                tensor.shape[0],
                5120,
                self.trt_pointers,
                False,
                True,  # use_oneshot (FlashInfer 0.6.9)
                True,  # trigger_completion_at_end
                False,
                0,
                out,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            return out
        self.module.registered_allreduce(
            tensor,
            out,
            self.pointers,
            self.buffers[self.rank],
            self.max_bytes,
            *self.signals,
            self.rank,
            36,
        )
        return out

    def close(self):
        from flashinfer.comm.cuda_ipc import free_shared_buffer

        if self.signals is None:
            return
        torch.cuda.synchronize()
        dist.barrier(self.group)
        for pointers in (self.buffers, self.signals, *self.trt_buffers):
            self.module.close_ipc_pointer(pointers[1 - self.rank])
        dist.barrier(self.group)
        for pointers in (self.buffers, self.signals, *self.trt_buffers):
            free_shared_buffer(pointers, self.group)
        self.signals = None


def prepare_communicator(parallelism, hidden_size, quant_config):
    """Initialize only the measured single-node SM120 mixed-quant model path."""
    global _communicator
    if (
        _communicator is not None
        or os.getenv("RTP_LLM_SM120_FUSION", "0") != "1"
        or not torch.cuda.is_available()
        or torch.version.hip is not None
        or hidden_size != 5120
        or parallelism.tp_size != 2
        or parallelism.world_size != 2
        or parallelism.local_world_size != 2
        or torch.cuda.device_count() != 2
        or any(torch.cuda.get_device_capability(i) != (12, 0) for i in range(2))
        or os.getenv("RTP_LLM_STREAM_ASYNC", "0") == "1"
        or os.getenv("RTP_LLM_MTP_ASYNC_DEVICE_STATE", "0") == "1"
        or quant_config is None
        or quant_config.get_method() != "modelopt_mixed"
    ):
        return
    if not all(torch.cuda.can_device_access_peer(i, 1 - i) for i in range(2)):
        return
    from rtp_llm.models_py.distributed.collective_torch import Group, _get_group

    _communicator = Sm120IpcAllReduce(_get_group(Group.TP))


def get_communicator():
    return _communicator


def destroy_communicator():
    global _communicator
    if _communicator is not None:
        _communicator.close()
        _communicator = None
