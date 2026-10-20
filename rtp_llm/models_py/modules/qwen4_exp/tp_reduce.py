"""Qwen4-owned reduction dispatch; draft and eager paths keep their contract."""

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.flashinfer_all_reduce import get_flashinfer_allreduce


def install_target_graph_all_reduce(attention, mlp):
    """Include the actual routed finalizer, even when shared FFN is unsharded."""
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
        PureTpRouterNvfp4Cutlass,
    )

    attention.tp_all_reduce = all_reduce_target_graph
    mlp.tp_all_reduce = all_reduce_target_graph
    router = getattr(getattr(mlp, "fused_moe", None), "router", None)
    if type(router) is PureTpRouterNvfp4Cutlass:
        router.tp_all_reduce = all_reduce_target_graph


def all_reduce_target_graph(tensor, group):
    if tensor.is_cuda and group == collective_torch.Group.TP:
        communicator = get_flashinfer_allreduce()
        if communicator is not None and communicator.should_use_registered_graph_input(
            tensor
        ):
            return collective_torch.all_reduce(tensor, group, inplace=False)
    return collective_torch.all_reduce(tensor, group)
