"""PPU pure-TP router with FP8 per-block [128,128] activation quantization."""

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
    PureTpRouterFp8PerBlock,
)
from rtp_llm.ops.compute_ops import trt_fp8_quantize_128


class PpuPureTpRouterFp8PerBlock(PureTpRouterFp8PerBlock):
    """Pure TP router for PPU.

    The CUDA parent's ``_do_quant`` consults ``is_deep_gemm_e8m0_used()``,
    a CUDA SM100/120 probe that is not safe on PPU; the PPU wheel has no
    UE8M0 scales, so quantize with the PPU-verified ``trt_fp8_quantize_128``
    directly.
    """

    def __init__(
        self,
        config,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config)

    def _do_quant(self, a1: torch.Tensor):
        if a1.dtype != torch.bfloat16:
            raise ValueError(
                f"PPU FP8 per-block quant expects BF16 input, got {a1.dtype}"
            )
        return trt_fp8_quantize_128(a1, False)
