"""BF16 shared-expert GEMM layout selection for SM120 Decode."""

import logging

import torch
import torch.nn.functional as F

from rtp_llm.models_py.distributed.collective_utils import exchange_group_state
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)


class _PackedSharedLinear(CudaF16Linear):
    def __init__(self, projection, min_rows, max_rows):
        super().__init__(projection.weight.t())
        self.packed_weight = projection.weight.contiguous()
        self.min_rows = min_rows
        self.max_rows = max_rows

    def forward(self, x):
        if x.ndim == 2 and self.min_rows <= x.shape[0] <= self.max_rows:
            return F.linear(x, self.packed_weight)
        return super().forward(x)


def _packing_agreement(group, state, phase):
    peers = exchange_group_state(group, state, phase) if group is not None else [state]
    if (
        any(peer["error"] is not None for peer in peers)
        or len({peer["packed_bytes"] for peer in peers}) != 1
    ):
        logger.warning("Keeping original shared GEMMs on the TP group: %s", peers)
        return False
    return True


@torch.no_grad()
def prepare_shared_gemm(layers, source_layers, *, group=None):
    """Commit packed projections and their update policy on the entire TP group."""
    projections = []
    state = {"error": None, "packed_bytes": 0}
    try:
        if len(layers) != len(source_layers) or not layers:
            raise ValueError("Shared GEMM layers and source owners must match")
        device = torch.device("cuda", torch.cuda.current_device())
        for layer, source in zip(layers, source_layers):
            shared = layer.mlp.shared_expert
            for name, key, shape, min_rows, max_rows in (
                ("up_proj", W.ffn_w13, (1280, 2560), 1, 16),
                ("down_proj", W.ffn_w2, (2560, 640), 2, 32),
            ):
                projection = getattr(shared, name)
                owner = source.get(key)
                if type(projection) is not CudaF16Linear or owner is None:
                    raise ValueError(
                        "Shared GEMM requires BF16 projections and source owners"
                    )
                weight = projection.weight
                if not (
                    projection.bias is None
                    and weight.dtype == torch.bfloat16
                    and weight.device == owner.device == device
                    and tuple(weight.shape) == shape
                    and tuple(weight.stride()) == (1, shape[0])
                    and owner.data_ptr() == weight.data_ptr()
                    and owner.numel() == weight.numel()
                ):
                    raise ValueError("Unqualified shared GEMM weight layout")
                state["packed_bytes"] += weight.numel() * weight.element_size()
                projections.append(
                    (shared, name, projection, owner, min_rows, max_rows)
                )
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if free_bytes < state["packed_bytes"]:
            raise torch.OutOfMemoryError("Insufficient memory for shared GEMM packing")
    except Exception as error:
        state["error"] = str(error)
    if not _packing_agreement(group, state, "qwen4-shared-gemm-preflight"):
        return False

    # Keep candidates private until both ranks finish allocating and copying.
    # Build incrementally so even a mid-layer OOM can release earlier copies.
    packed = []
    try:
        for _, _, projection, _, min_rows, max_rows in projections:
            packed.append(_PackedSharedLinear(projection, min_rows, max_rows))
        torch.cuda.synchronize(device)
    except Exception as error:
        state["error"] = str(error)
    if not _packing_agreement(group, state, "qwen4-shared-gemm-packed"):
        packed.clear()
        return False

    for (shared, name, _, owner, _, _), projection in zip(projections, packed):
        # Mark the WeightManager's owner only after the group commits. Failed
        # packing must preserve the original update policy on every rank.
        owner._rtp_requires_restart_for_update = True
        setattr(shared, name, projection)
    logger.info(
        "Packed %d shared-expert GEMM bytes on %s", state["packed_bytes"], device
    )
    return True
