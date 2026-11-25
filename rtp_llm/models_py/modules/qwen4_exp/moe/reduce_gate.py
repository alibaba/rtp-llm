"""Model-owned TP2 LSA reduction with the shared expert epilogue."""

import logging
import os
import socket
from functools import partial

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.collective_utils import exchange_group_state

# Captured graph nodes contain native context addresses. Retain them if model
# close occurs on only one rank or a graph runner still references the model.
_PROCESS_LIFETIME_CONTEXTS = []
_MAX_ROWS = 32
_NCCL_HEADROOM_BYTES = 8 * 1024 * 1024
logger = logging.getLogger(__name__)


def _aligned_buffer_like(value):
    output = torch.empty_like(value, memory_format=torch.contiguous_format)
    if output.data_ptr() % 16 == 0:
        return output
    # A contiguous view can start at an unaligned BF16 offset. Construct an
    # aligned slice explicitly if the allocator also returns an unaligned base.
    storage = value.new_empty(value.numel() + 7)
    offset = (-storage.data_ptr() % 16) // value.element_size()
    return storage.narrow(0, offset, value.numel()).view_as(value)


class LsaMoeReduce:
    """One symmetric window shared by serialized forwards of a single model."""

    def __init__(self, ops, device, hidden_size, handle, buffers):
        self.hidden_size = hidden_size
        self.ops = ops
        self.device = device
        self.closed = False
        self.handle = handle
        self.partial_staging, self.shared_staging, self.gate_staging = buffers

    def supports(self, value):
        # TP consumes the same logical batch shape on both ranks. Local tensor
        # addresses and strides must never select a different collective.
        return (
            value.ndim == 2
            and 1 <= value.shape[0] <= _MAX_ROWS
            and value.shape[1] == self.hidden_size
        )

    def prepare_output(self, value):
        # Allocate while the routed branch still owns its ordinary collective.
        # Each call retains its own output, including distinct Graph nodes.
        if self.closed:
            raise RuntimeError("LSA owner is closed; restart the TP group")
        return _aligned_buffer_like(value)

    def __call__(self, partial, shared, gate, output):
        # Once the routed collective is suppressed, a local failure cannot
        # choose a different collective. Follow the TP supervisor's fatal
        # failure policy; never resume this worker on another collective.
        if not (
            not self.closed
            and self.supports(partial)
            and partial.shape == shared.shape
            and gate.shape == (partial.shape[0], 1)
            and output.shape == partial.shape
            and output.is_contiguous()
            and output.data_ptr() % 16 == 0
            and all(
                value.device == self.device and value.dtype == torch.bfloat16
                for value in (partial, shared, gate, output)
            )
        ):
            raise ValueError("LSA epilogue requires local BF16 [M,H] and gate [M,1]")
        rows = partial.shape[0]
        # Staging is model-owned and allocated before communicator bootstrap.
        # The serialized Decode stream and the final LSA barrier protect reuse.
        if not partial.is_contiguous():
            self.partial_staging[:rows].copy_(partial)
            partial = self.partial_staging[:rows]
        if not shared.is_contiguous() or shared.data_ptr() % 16:
            self.shared_staging[:rows].copy_(shared)
            shared = self.shared_staging[:rows]
        if not gate.is_contiguous():
            self.gate_staging[:rows].copy_(gate)
            gate = self.gate_staging[:rows]
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
    if (
        role != "DECODE"
        or parallelism.tp_size != 2
        or getattr(config, "is_mtp", False)
        or not dist.is_initialized()
    ):
        return None
    from rtp_llm.models_py.distributed import collective_torch

    group = collective_torch._get_group(collective_torch.Group.TP)
    if dist.get_world_size(group) != 2:
        return None

    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
        PureTpRouterNvfp4Cutlass,
    )

    quant_config = getattr(config, "quant_config", None)
    timeout_ms = getattr(quant_config, "decode_lsa_timeout_ms", 0)
    startup_timeout_s = getattr(quant_config, "decode_lsa_startup_timeout_s", None)
    exchange = partial(exchange_group_state, group, timeout_s=startup_timeout_s)
    state = {
        "timeout_ms": timeout_ms,
        "startup_timeout_s": startup_timeout_s,
        "error": None,
        "host": socket.gethostname(),
        "layers": len(model.layers),
        "hidden": config.hidden_size,
    }
    ops = None
    device = None
    try:
        if not getattr(quant_config, "decode_lsa_enabled", True):
            raise ValueError("LSA disabled by the Decode configuration")
        if type(timeout_ms) is not int or not 0 <= timeout_ms <= 2**31 - 1:
            raise ValueError(
                "decode_lsa_timeout_ms must be an integer from 0 through 2147483647"
            )
        if (
            os.environ.get("NCCL_P2P_DISABLE") == "1"
            or os.environ.get("NCCL_CUMEM_ENABLE") == "0"
        ):
            raise ValueError("NCCL P2P access or CUDA memory management is disabled")
        if not (
            parallelism.ep_size == 2
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
        device = torch.device("cuda", torch.cuda.current_device())
        window_bytes = _MAX_ROWS * config.hidden_size * torch.bfloat16.itemsize
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if free_bytes < window_bytes + _NCCL_HEADROOM_BYTES:
            raise ValueError(
                "insufficient free CUDA memory for the LSA window and headroom"
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
                and mlp.shared_expert_gate.weight.device == device
                and mlp.shared_expert_gate.weight.dtype == torch.bfloat16
            ):
                raise ValueError(
                    "Unsupported shared-expert or routed collective layout"
                )
        from rtp_llm.ops import rtp_llm_ops

        ops = rtp_llm_ops._qwen38_lsa
        state["bus_id"] = ops.device(device.index)
    except Exception as error:
        state["error"] = str(error)
    peers = exchange(state, "qwen38-lsa-layout")
    if (
        any(peer["error"] is not None for peer in peers)
        or len({peer["host"] for peer in peers}) != 1
        or len({peer["layers"] for peer in peers}) != 1
        or len({peer["hidden"] for peer in peers}) != 1
        or len({peer["timeout_ms"] for peer in peers}) != 1
        or len({peer["startup_timeout_s"] for peer in peers}) != 1
        or len({peer["bus_id"] for peer in peers}) != 2
    ):
        logger.info("MoE LSA unavailable for this TP group: %s", peers)
        return None

    rank = dist.get_rank(group)
    handle = None
    buffers = None
    try:
        state = {"error": None, "uid": None}
        try:
            ops.check_peer(device.index, peers[1 - rank]["bus_id"])
            if rank == 0:
                state["uid"] = ops.unique_id().hex()
            handle = ops.allocate(device.index, config.hidden_size)
            ops.set_timeout(handle, timeout_ms)
            buffers = (
                torch.empty(
                    (_MAX_ROWS, config.hidden_size), device=device, dtype=torch.bfloat16
                ),
                torch.empty(
                    (_MAX_ROWS, config.hidden_size), device=device, dtype=torch.bfloat16
                ),
                torch.empty((_MAX_ROWS, 1), device=device, dtype=torch.bfloat16),
            )
        except Exception as error:
            state["error"] = str(error)
        peers = exchange(state, "qwen38-lsa-allocation")
        if any(peer["error"] is not None for peer in peers):
            logger.info("MoE LSA local setup unavailable: %s", peers)
            return None

        # Communicator bootstrap itself is collective. An error in that call
        # is fatal. Once it returns, each rank can query its capabilities
        # and decline together before the next NCCL collective.
        ops.connect(handle, rank, bytes.fromhex(peers[0]["uid"]))
        state = {"error": None}
        try:
            ops.check_capability(handle)
        except Exception as error:
            state["error"] = str(error)
        peers = exchange(state, "qwen38-lsa-capability")
        if any(peer["error"] is not None for peer in peers):
            logger.info("MoE LSA communicator capability unavailable: %s", peers)
            return None

        # Window registration and device communicator creation are
        # collective. Let errors propagate instead of switching protocols.
        ops.activate(handle)
        state = {"error": None}
        try:
            ops.check_activation(handle)
        except Exception as error:
            state["error"] = str(error)
        peers = exchange(state, "qwen38-lsa-ready")
        if any(peer["error"] is not None for peer in peers):
            logger.info("MoE LSA activation unavailable: %s", peers)
            return None
        owner = LsaMoeReduce(ops, device, config.hidden_size, handle, buffers)
        handle = None
    finally:
        if handle is not None:
            ops.destroy(handle)
    logger.info(
        "MoE LSA ready on %s; timeout_ms=%d (0 waits for peer or local cancellation)",
        device,
        timeout_ms,
    )
    for layer in model.layers:
        layer.mlp._reduce_shared_gate = owner
    return owner
