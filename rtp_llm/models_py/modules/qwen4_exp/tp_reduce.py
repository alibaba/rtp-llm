"""Qwen4-owned reduction dispatch; draft and eager paths keep their contract."""

import logging
from functools import partial

import torch

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.all_reduce import get_all_reduce
from rtp_llm.models_py.distributed.collective_utils import exchange_group_state
from rtp_llm.models_py.modules.qwen4_exp._profiler import record_function_range
from rtp_llm.models_py.modules.qwen4_exp.tp2_ar_hc_fused import (
    try_defer_sublayer_tp2,
)

# Model close can also run on a single rank after partial initialization.
# Retain IPC mappings until process exit rather than freeing a peer's input.
_PROCESS_LIFETIME_TARGET_REDUCE = []


def install_target_graph_all_reduce(
    attention, mlp, dispatch=None, *, fp8_gdn_decode=False, hopper_communicator=None
):
    """Include the actual routed finalizer, even when shared FFN is unsharded."""
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
        PureTpRouterNvfp4Cutlass,
    )

    standalone_reduce = (
        all_reduce_target_graph
        if hopper_communicator is None
        else partial(all_reduce_target_graph, hopper_communicator=hopper_communicator)
    )
    dispatch = standalone_reduce if dispatch is None else dispatch
    attention.tp_all_reduce = dispatch
    if fp8_gdn_decode:
        attention.tp_all_reduce_decode = partial(standalone_reduce, inplace=None)
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
    """Extend an already admitted IPC backend; agree before allocating."""
    role = str(getattr(parallelism, "role_type", "")).rsplit(".", 1)[-1]
    communicator = get_all_reduce()
    if not (
        not getattr(config, "is_mtp", False)
        and config.hidden_size == 2560
        and role in ("DECODE", "PDFUSION")
        and parallelism.tp_size == 2
        and parallelism.dp_size == 1
        and parallelism.local_world_size == 2
        and communicator is not None
        and not communicator.disabled
        and communicator.cpp_small_enabled
    ):
        return None
    from rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce import (
        IpcInitializationError,
    )

    # rank_data plus the exported metadata/staging allocations. Account for
    # raw CUDA IPC allocations, which are outside Torch's caching allocator.
    max_bytes = 256 * 2560 * 2
    required_bytes = 8 * 1024 * 1024 + 2 * max_bytes
    error = None
    try:
        required_bytes += communicator._workspace.ops.meta_size()
        available_bytes, _ = torch.cuda.mem_get_info(communicator.device)
        if available_bytes < required_bytes:
            error = "insufficient free memory for target IPC workspace"
    except Exception as exc:
        error = str(exc)
    peers = exchange_group_state(communicator.group, error, "qwen4_target_memory")
    if any(peer is not None for peer in peers):
        logging.warning(
            "Qwen4 target IPC unavailable; using existing reducer: %s", peers
        )
        return None
    try:
        return Qwen4TargetReduce(communicator, max_bytes=max_bytes)
    except IpcInitializationError:
        # The workspace constructor has already agreed failure and released
        # any partial setup on all ranks. Keep the original communicator.
        logging.warning(
            "Qwen4 target IPC setup failed; using existing reducer", exc_info=True
        )
        return None


def all_reduce_target_graph(tensor, group, *, inplace=True, hopper_communicator=None):
    deferred = try_defer_sublayer_tp2(tensor, group)
    if deferred is not None:
        return deferred
    with record_function_range("qwen4.tp_all_reduce"):
        if tensor.is_cuda and group == collective_torch.Group.TP:
            if hopper_communicator is not None and hopper_communicator.should_use(tensor):
                if inplace is True:
                    return hopper_communicator.all_reduce_into(tensor, tensor)
                return hopper_communicator.all_reduce(tensor)
            communicator = get_all_reduce()
            if communicator is not None and communicator.should_use_registered_graph_input(
                tensor
            ):
                return collective_torch.all_reduce(tensor, group, inplace=False)
        if inplace is None:
            return collective_torch.all_reduce(tensor, group, inplace=None)
        return collective_torch.all_reduce(tensor, group)
