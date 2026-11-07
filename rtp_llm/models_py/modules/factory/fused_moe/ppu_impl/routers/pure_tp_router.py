"""PPU pure-TP router with FP8 per-block [128,128] activation quantization."""

from typing import Optional

import torch

from rtp_llm.models_py.kernels.ppu.fp8_quant import fp8_quantize_128
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
    PureTpRouterFp8PerBlock,
)


class PpuPureTpRouterFp8PerBlock(PureTpRouterFp8PerBlock):
    """Pure TP router for PPU.

    The CUDA parent's ``_do_quant`` consults ``is_deep_gemm_e8m0_used()``,
    a CUDA SM100/120 probe that is not safe on PPU; the PPU wheel has no
    UE8M0 scales, so quantize with the PPU per-group FP32-scale quantizer
    directly.
    """

    def __init__(
        self,
        config,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config)

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        payload = super().prepare(a1, a1_scale, a2_scale, topk_weights, topk_ids)
        # ``recompute_topk_ids_sum_expert_count`` already emitted partition-local
        # ids with ``-1`` outside the partition. Restoring global ids here only
        # for the executor to subtract ``start_expert_id`` again would cost ~9
        # int64 elementwise kernels per MoE layer, so hand the local ids through.
        payload.expert_ids_are_local = True
        return payload

    def _do_quant(self, a1: torch.Tensor):
        if a1.dtype != torch.bfloat16:
            raise ValueError(
                f"PPU FP8 per-block quant expects BF16 input, got {a1.dtype}"
            )
        return fp8_quantize_128(a1)
