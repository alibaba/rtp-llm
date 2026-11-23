"""Qwen4-owned reduction dispatch; draft and eager paths keep their contract."""

from functools import partial

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.flashinfer_all_reduce import get_flashinfer_allreduce
from rtp_llm.models_py.modules.qwen4_exp._profiler import record_function_range
from rtp_llm.models_py.modules.qwen4_exp.tp2_ar_hc_fused import (
    try_defer_sublayer_tp2,
)


def install_target_graph_all_reduce(attention, mlp, *, fp8_gdn_decode=False):
    """Include the actual routed finalizer, even when shared FFN is unsharded."""
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
        PureTpRouterNvfp4Cutlass,
    )

    attention.tp_all_reduce = all_reduce_target_graph
    if fp8_gdn_decode:
        # The fresh projection output has no other consumer before TP reduction.
        attention.tp_all_reduce_decode = partial(all_reduce_target_graph, inplace=None)
    mlp.tp_all_reduce = all_reduce_target_graph
    router = getattr(getattr(mlp, "fused_moe", None), "router", None)
    if type(router) is PureTpRouterNvfp4Cutlass:
        router.tp_all_reduce = all_reduce_target_graph


def all_reduce_target_graph(tensor, group, *, inplace=True):
    deferred = try_defer_sublayer_tp2(tensor, group)
    if deferred is not None:
        return deferred
    with record_function_range("qwen4.tp_all_reduce"):
        if tensor.is_cuda and group == collective_torch.Group.TP:
            communicator = get_flashinfer_allreduce()
            if (
                communicator is not None
                and communicator.should_use_registered_graph_input(tensor)
            ):
                return collective_torch.all_reduce(tensor, group, inplace=False)
        if inplace is None:
            return collective_torch.all_reduce(tensor, group, inplace=None)
        return collective_torch.all_reduce(tensor, group)
