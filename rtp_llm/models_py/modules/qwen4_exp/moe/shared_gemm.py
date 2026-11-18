"""BF16 shared-expert GEMM layout selection for SM120 Decode."""

import logging

import torch
import torch.nn.functional as F

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


@torch.no_grad()
def prepare_shared_gemm(layers, source_layers):
    """Pack the measured layout once before graph capture and fence live updates."""
    if len(layers) != len(source_layers) or not layers:
        return
    device = torch.device("cuda", torch.cuda.current_device())
    projections = []
    packed_bytes = 0
    for layer, source in zip(layers, source_layers):
        shared = layer.mlp.shared_expert
        for name, key, shape, min_rows, max_rows in (
            ("up_proj", W.ffn_w13, (1280, 2560), 1, 16),
            ("down_proj", W.ffn_w2, (2560, 640), 2, 32),
        ):
            projection = getattr(shared, name)
            owner = source.get(key)
            if type(projection) is not CudaF16Linear or owner is None:
                logger.info("Keeping original shared GEMMs for non-BF16 or split projections")
                return
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
                logger.info("Keeping original shared GEMMs for unqualified weight layout")
                return
            packed_bytes += weight.numel() * weight.element_size()
            projections.append((shared, name, projection, owner, min_rows, max_rows))

    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < packed_bytes:
        logger.warning("Insufficient CUDA memory to pack %d shared GEMM bytes", packed_bytes)
        return
    try:
        packed = [
            _PackedSharedLinear(projection, min_rows, max_rows)
            for _, _, projection, _, min_rows, max_rows in projections
        ]
    except torch.OutOfMemoryError:
        logger.warning("Shared GEMM packing ran out of memory. Keeping original GEMMs.")
        return

    for (shared, name, _, owner, _, _), projection in zip(projections, packed):
        # WeightManager's update API owns this tensor, so marking only the
        # transpose view would allow updates behind the packed Graph weights.
        owner._rtp_requires_restart_for_update = True
        setattr(shared, name, projection)
    logger.info("Packed %d bytes of BF16 shared-expert GEMM weights on %s", packed_bytes, device)
