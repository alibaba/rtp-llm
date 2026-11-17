"""Model-owned TP2 LSA reduction with the shared expert epilogue."""

import logging
import socket

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.collective_utils import exchange_group_state

# Captured graph nodes contain native context addresses. Retain them if model
# close occurs on only one rank or a graph runner still references the model.
_PROCESS_LIFETIME_CONTEXTS = []
_MAX_ROWS = 32
logger = logging.getLogger(__name__)


def _aligned_buffer_like(value):
    output = torch.empty_like(value)
    if output.data_ptr() % 16 == 0:
        return output
    # A contiguous view can start at an unaligned BF16 offset. Construct an
    # aligned slice explicitly if the allocator also returns an unaligned base.
    storage = value.new_empty(value.numel() + 7)
    offset = (-storage.data_ptr() % 16) // value.element_size()
    return storage.narrow(0, offset, value.numel()).view_as(value)


class LsaMoeReduce:
    """One symmetric window shared by serialized forwards of a single model."""

    def __init__(self, ops, group, device, hidden_size, uid):
        self.hidden_size = hidden_size
        self.ops = ops
        self.device = device
        self.closed = False
        with torch.cuda.device(device):
            self.handle = ops.create(
                dist.get_rank(group), device.index, hidden_size, uid
            )

    def supports(self, value):
        # TP consumes the same logical batch shape on both ranks. Local tensor
        # addresses and strides must never select a different collective.
        return (
            value.ndim == 2
            and 1 <= value.shape[0] <= _MAX_ROWS
            and value.shape[1] == self.hidden_size
        )

    def __call__(self, partial, shared, gate):
        if not (
            not self.closed
            and self.supports(partial)
            and partial.shape == shared.shape
            and gate.shape == (partial.shape[0], 1)
            and all(
                value.device == self.device and value.dtype == torch.bfloat16
                for value in (partial, shared, gate)
            )
        ):
            raise ValueError("LSA epilogue requires local BF16 [M,H] and gate [M,1]")
        partial = partial.contiguous()  # Native code stages it with a byte copy.
        shared = shared.contiguous()
        gate = gate.contiguous()
        if shared.data_ptr() % 16:
            aligned = _aligned_buffer_like(shared)
            aligned.copy_(shared)
            shared = aligned
        output = _aligned_buffer_like(partial)
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            self.ops.apply(
                self.handle,
                partial.data_ptr(),
                shared.data_ptr(),
                gate.data_ptr(),
                output.data_ptr(),
                partial.shape[0],
                stream.cuda_stream,
            )
        return output

    def close(self):
        if self.closed:
            return
        self.closed = True
        _PROCESS_LIFETIME_CONTEXTS.append(self)


def prepare_model_reduce(model, config, parallelism):
    """Select LSA only after both TP ranks agree on capability and layout."""
    role = str(parallelism.role_type).rsplit(".", 1)[-1]
    if role != "DECODE" or parallelism.tp_size != 2 or not dist.is_initialized():
        return None
    from rtp_llm.models_py.distributed import collective_torch

    group = collective_torch._get_group(collective_torch.Group.TP)
    if dist.get_world_size(group) != 2:
        return None

    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
        PureTpRouterNvfp4Cutlass,
    )

    state = {
        "error": None,
        "host": socket.gethostname(),
        "layers": len(model.layers),
        "hidden": config.hidden_size,
    }
    ops = None
    device = None
    try:
        if not (
            not getattr(config, "is_mtp", False)
            and parallelism.ep_size == 2
            and parallelism.dp_size == parallelism.pp_size == 1
            and parallelism.local_world_size == 2
            and parallelism.get_ffn_tp_size() == 1
            and type(config.hidden_size) is int
            and 0 < config.hidden_size <= (2**31 - 1) // _MAX_ROWS
            and config.hidden_size % 8 == 0
            and getattr(config, "_qwen4_dense_storage", None) == "source_bf16"
            and not config.eplb_config.enable_eplb()
        ):
            raise ValueError(
                "LSA shared merge requires local EP2, BF16 Dense and a positive hidden size divisible by 8"
            )
        for layer in model.layers:
            mlp = layer.mlp
            if not (
                type(mlp.fused_moe.router) is PureTpRouterNvfp4Cutlass
                and mlp.shared_expert is not None
                and mlp.shared_expert_gate is not None
                and not mlp.use_ep_shared_allreduce
                and not mlp.use_unified_tp_allreduce
                and getattr(mlp, "_shared_stream", None) is None
                and tuple(mlp.shared_expert_gate.weight.shape)
                == (1, config.hidden_size)
                and mlp.shared_expert_gate.weight.is_cuda
                and mlp.shared_expert_gate.weight.dtype == torch.bfloat16
            ):
                raise ValueError(
                    "Unsupported shared-expert or routed collective layout"
                )
        from rtp_llm.ops import rtp_llm_ops

        ops = rtp_llm_ops._qwen38_lsa
        device = torch.device("cuda", torch.cuda.current_device())
        state["bus_id"] = ops.device(device.index)
    except Exception as error:
        state["error"] = str(error)
    peers = exchange_group_state(group, state, "qwen38-lsa-layout")
    if (
        any(peer["error"] is not None for peer in peers)
        or len({peer["host"] for peer in peers}) != 1
        or len({peer["layers"] for peer in peers}) != 1
        or len({peer["hidden"] for peer in peers}) != 1
        or len({peer["bus_id"] for peer in peers}) != 2
    ):
        logger.info("MoE LSA unavailable for this TP group: %s", peers)
        return None

    rank = dist.get_rank(group)
    state = {"error": None, "uid": None}
    try:
        ops.check_peer(device.index, peers[1 - rank]["bus_id"])
        if rank == 0:
            state["uid"] = ops.unique_id().hex()
    except Exception as error:
        state["error"] = str(error)
    peers = exchange_group_state(group, state, "qwen38-lsa-peer")
    if any(peer["error"] is not None for peer in peers):
        logger.info("MoE LSA peer setup unavailable: %s", peers)
        return None

    # From here both ranks enter the same native collective. Propagate errors
    # instead of allowing one rank to revert to a different protocol.
    owner = LsaMoeReduce(
        ops, group, device, config.hidden_size, bytes.fromhex(peers[0]["uid"])
    )
    for layer in model.layers:
        layer.mlp._reduce_shared_gate = owner
    return owner
