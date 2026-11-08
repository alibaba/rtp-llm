"""Qwen4 TP resource lifetime and independently selected reduction policies."""

import os

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.flashinfer_all_reduce import get_flashinfer_allreduce

# Model close can also run on a single rank after partial initialization.
# Retain IPC mappings until process exit rather than freeing a peer's input.
_PROCESS_LIFETIME_REDUCE_WORKSPACES = []


def install_tp_all_reduce(attention, mlp, dispatch):
    """Include the actual routed finalizer, even when shared FFN is unsharded."""
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
        PureTpRouterNvfp4Cutlass,
    )

    attention.tp_all_reduce = dispatch
    mlp.tp_all_reduce = dispatch
    router = getattr(getattr(mlp, "fused_moe", None), "router", None)
    if type(router) is PureTpRouterNvfp4Cutlass:
        router.tp_all_reduce = dispatch


class Qwen4ReduceResource:
    """Model-owned IPC workspace lifetime, independent of dispatch policy.

    All ranks create the resource before use and register pointers after
    capture. Closing detaches the owner but retains mappings until process exit
    because shutdown or partial initialization need not be rank-symmetric.
    """

    def __init__(self, communicator, *, max_bytes):
        from rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce import (
            FlashInferIpcAllReduce,
        )

        self.workspace = FlashInferIpcAllReduce(
            communicator.group, communicator.device, max_bytes=max_bytes
        )
        self.closed = False

    def finish_graph_capture(self):
        if not self.closed:
            self.workspace.finish_graph_capture()

    def close(self):
        if not self.closed:
            _PROCESS_LIFETIME_REDUCE_WORKSPACES.append(self.workspace)
            self.workspace = None
            self.closed = True


class Qwen4TargetReduce(Qwen4ReduceResource):
    """Target-verify policy for TP2 rows33..256, enabled by the caller."""

    def __init__(self, communicator):
        super().__init__(communicator, max_bytes=256 * 2560 * 2)
        self.active = False

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
        return all_reduce_with_graph_input(tensor, group)

    def close(self):
        super().close()
        self.active = False


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
    mode = os.getenv("RTP_QWEN4_TARGET_REDUCE", "legacy")
    peers = [None] * 2
    dist.all_gather_object(peers, mode, group=communicator.group)
    if any(peer not in ("legacy", "ipc256") for peer in peers):
        raise ValueError(
            "RTP_QWEN4_TARGET_REDUCE must be legacy or ipc256 on every rank"
        )
    if any(peer != "ipc256" for peer in peers):
        return None
    return Qwen4TargetReduce(communicator)


class Qwen4PrefillReduce(Qwen4ReduceResource):
    """Dedicated eager staging for the measured TP2 long-Prefill rows."""

    def __init__(self, communicator):
        super().__init__(communicator, max_bytes=16384 * 2560 * 2)

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
        return all_reduce_with_graph_input(tensor, group)


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
    return Qwen4PrefillReduce(communicator)


def all_reduce_with_graph_input(tensor, group):
    """Use out-of-place reduction when a registered Graph input is admitted."""
    if tensor.is_cuda and group == collective_torch.Group.TP:
        communicator = get_flashinfer_allreduce()
        if communicator is not None and communicator.should_use_registered_graph_input(
            tensor
        ):
            return collective_torch.all_reduce(tensor, group, inplace=False)
    return collective_torch.all_reduce(tensor, group)
