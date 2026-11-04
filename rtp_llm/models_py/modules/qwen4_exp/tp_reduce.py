"""Qwen4-owned reduction dispatch; draft and eager paths keep their contract."""

import os

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.flashinfer_all_reduce import get_flashinfer_allreduce

# Model close can also run on a single rank after partial initialization.
# Retain IPC mappings until process exit rather than freeing a peer's input.
_PROCESS_LIFETIME_TARGET_REDUCE = []


def install_target_graph_all_reduce(attention, mlp, dispatch=None):
    """Include the actual routed finalizer, even when shared FFN is unsharded."""
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
        PureTpRouterNvfp4Cutlass,
    )

    dispatch = all_reduce_target_graph if dispatch is None else dispatch
    attention.tp_all_reduce = dispatch
    mlp.tp_all_reduce = dispatch
    router = getattr(getattr(mlp, "fused_moe", None), "router", None)
    if type(router) is PureTpRouterNvfp4Cutlass:
        router.tp_all_reduce = dispatch


class Qwen4TargetReduce:
    """Model-owned workspace for TP2 target-verify rows 33..256.

    All ranks construct this eagerly and register graph pointers after capture.
    The caller enables it only during target verification. Generic collectives
    and the draft keep their existing dispatch and workspace.
    """

    def __init__(self, communicator, *, max_bytes=256 * 2560 * 2):
        from rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce import (
            FlashInferIpcAllReduce,
        )

        self.workspace = FlashInferIpcAllReduce(
            communicator.group, communicator.device, max_bytes=max_bytes
        )
        self.active = False
        self.closed = False

    def __call__(self, tensor, group):
        if (
            self.active
            and not self.closed
            and group == collective_torch.Group.TP
            and tensor.is_cuda
            and tensor.dtype == torch.bfloat16
            and tensor.ndim == 2
            and 33 <= tensor.shape[0] <= 256
            and tensor.shape[1] == 2560
            and tensor.is_contiguous()
            and tensor.data_ptr() % 16 == 0
        ):
            return self.workspace.all_reduce(tensor)
        return all_reduce_target_graph(tensor, group)

    def finish_graph_capture(self):
        if not self.closed:
            self.workspace.finish_graph_capture()

    def close(self):
        if not self.closed:
            _PROCESS_LIFETIME_TARGET_REDUCE.append(self.workspace)
            self.workspace = None
            self.active = False
            self.closed = True


def prepare_target_reduce(config, parallelism):
    role = str(getattr(parallelism, "role_type", "")).rsplit(".", 1)[-1]
    communicator = get_flashinfer_allreduce()
    if not (
        not getattr(config, "is_mtp", False)
        and role in ("DECODE", "PDFUSION")
        and parallelism.tp_size == 2
        and parallelism.dp_size == 1
        and parallelism.local_world_size == 2
        and communicator is not None
        and not communicator.disabled
        and communicator.cpp_small_enabled
    ):
        return None
    enabled = os.getenv("RTP_QWEN4_TARGET_REDUCE", "legacy") == "ipc256"
    peers = [None] * 2
    dist.all_gather_object(peers, enabled, group=communicator.group)
    if not all(peers):
        return None
    return Qwen4TargetReduce(communicator)


class Qwen4PrefillReduce(Qwen4TargetReduce):
    """Dedicated eager staging for the measured TP2 long-Prefill rows."""

    def __init__(self, communicator):
        super().__init__(communicator, max_bytes=16384 * 2560 * 2)
        self.active = True

    def __call__(self, tensor, group):
        if (
            not self.closed
            and group == collective_torch.Group.TP
            and tensor.is_cuda
            and tensor.dtype == torch.bfloat16
            and tensor.ndim == 2
            and tensor.shape[0] in (4093, 4094, 4095, 4096, 8192, 16384)
            and tensor.shape[1] == 2560
            and tensor.is_contiguous()
            and tensor.data_ptr() % 16 == 0
            and not torch.cuda.is_current_stream_capturing()
        ):
            return self.workspace.all_reduce(tensor)
        return all_reduce_target_graph(tensor, group)


def prepare_prefill_reduce(config, parallelism):
    role = str(getattr(parallelism, "role_type", "")).rsplit(".", 1)[-1]
    communicator = get_flashinfer_allreduce()
    if not (
        not getattr(config, "is_mtp", False)
        and role == "PREFILL"
        and parallelism.tp_size == 2
        and parallelism.dp_size == 1
        and parallelism.local_world_size == 2
        and communicator is not None
        and not communicator.disabled
        and communicator.cpp_small_enabled
    ):
        return None
    enabled = os.getenv("RTP_QWEN4_PREFILL_REDUCE", "legacy") == "ipc"
    peers = [None] * 2
    dist.all_gather_object(peers, enabled, group=communicator.group)
    if not all(peers):
        return None
    return Qwen4PrefillReduce(communicator)


def all_reduce_target_graph(tensor, group):
    if tensor.is_cuda and group == collective_torch.Group.TP:
        communicator = get_flashinfer_allreduce()
        if communicator is not None and communicator.should_use_registered_graph_input(
            tensor
        ):
            return collective_torch.all_reduce(tensor, group, inplace=False)
    return collective_torch.all_reduce(tensor, group)
