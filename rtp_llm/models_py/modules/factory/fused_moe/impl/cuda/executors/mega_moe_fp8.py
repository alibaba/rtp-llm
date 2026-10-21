"""FP8-weight recipe for the shared MegaMoE executor."""

from __future__ import annotations

import logging

import torch

from rtp_llm.models_py.kernels.cuda.quant_layouts import (
    FP4_BLOCK,
    prepare_fp8_weight_scale_for_deepgemm,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.fp8_buffer import (
    get_or_create_mega_buf_fp8,
)

from .mega_moe import MegaMoeExecutor

logger = logging.getLogger(__name__)


class MegaMoeFp8Backend(MegaMoeExecutor):
    _fp8_routed = True
    _kernel_name = "fp8_fp8_mega_moe"
    _buffer_factory = "get_symm_buffer_for_mega_moe_fp8"

    def _allocate_mega_buffer(self):
        return get_or_create_mega_buf_fp8(**self._buffer_kwargs())

    def setup_weights_from_fp8(
        self,
        w1_fp8: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_fp8: torch.Tensor,
        w2_scale: torch.Tensor,
        w3_fp8: torch.Tensor,
        w3_scale: torch.Tensor,
    ) -> None:
        """Setup pre-quantized FP8 per-block expert weights for fp8_fp8_mega_moe.

        Inputs follow DeepGEMM convention: w1 is gate, w3 is up, w2 is down.
        Weight scales are either raw 128x128 per-block scales or DeepGEMM's
        packed int32 layout from the FP8 loader.
        """
        import deep_gemm

        cfg = self.cfg
        E = cfg.n_local_experts
        D = cfg.dim
        inter = cfg.moe_inter_dim

        for name, weight in (
            ("w1_fp8", w1_fp8),
            ("w2_fp8", w2_fp8),
            ("w3_fp8", w3_fp8),
        ):
            if weight.dtype != torch.float8_e4m3fn:
                raise TypeError(
                    f"mega_moe_fp8 requires FP8 e4m3 weights; {name} has {weight.dtype}"
                )

        logger.info(
            "[GLM5 MegaMoE FP8] preparing FP8 weights: layer=%d E=%d D=%d inter=%d",
            cfg.layer_id,
            E,
            D,
            inter,
        )

        w13_fp8 = torch.cat([w1_fp8, w3_fp8], dim=1).contiguous()
        if w1_scale.dtype == torch.float8_e8m0fnu:
            # The loader may hand us strided gate/up views. E8M0 has no
            # generic cat kernel; concatenate its bytes without conversion.
            s13 = torch.cat(
                [w1_scale.view(torch.uint8), w3_scale.view(torch.uint8)], dim=1
            ).view(w1_scale.dtype)
        else:
            s13 = torch.cat([w1_scale, w3_scale], dim=1).contiguous()
        del w1_fp8, w1_scale, w3_fp8, w3_scale

        if s13.dtype == torch.float32 or w2_scale.dtype == torch.float32:
            if s13.dtype != torch.float32 or w2_scale.dtype != torch.float32:
                raise TypeError(
                    "mega_moe_fp8 requires both FP8 MoE scales to be raw "
                    f"float32 or both prepacked int32/e8m0, got {s13.dtype} "
                    f"and {w2_scale.dtype}"
                )
            from rtp_llm.models_py.kernels.cuda.fp8_kernel import requant_weight_ue8m0

            logger.info(
                "[GLM5 MegaMoE FP8] requantizing FP8 weights to UE8M0 scales: "
                "layer=%d",
                cfg.layer_id,
            )
            w13_fp8, s13_int = requant_weight_ue8m0(w13_fp8, s13)
            del s13
            torch.cuda.empty_cache()
            w2_fp8, s2_int = requant_weight_ue8m0(w2_fp8.contiguous(), w2_scale)
            del w2_scale
            torch.cuda.empty_cache()
        else:
            s13_int = prepare_fp8_weight_scale_for_deepgemm(s13, 2 * inter, D, E)
            s2_int = prepare_fp8_weight_scale_for_deepgemm(w2_scale, D, inter, E)
            del s13, w2_scale
        torch.cuda.empty_cache()

        (l1_w, l1_sf), (l2_w, l2_sf) = deep_gemm.transform_weights_for_mega_moe_fp8(
            (w13_fp8, s13_int),
            (w2_fp8.contiguous(), s2_int),
        )
        del w13_fp8, s13_int, w2_fp8, s2_int
        torch.cuda.empty_cache()

        self._mega_l1_w = l1_w
        self._mega_l1_sf = l1_sf
        self._mega_l2_w = l2_w
        self._mega_l2_sf = l2_sf
        self._setup_buffer_and_warmup()

    def setup_weights_from_fp4(self, *args, **kwargs) -> None:
        raise ValueError("moe_strategy=mega_moe_fp8 only accepts FP8 per-block weights")

    def _run_kernel(self, y, tokens):
        import deep_gemm

        deep_gemm.fp8_fp8_mega_moe(
            y,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            self._mega_buf,
            recipe=(1, 1, FP4_BLOCK),
            activation="swiglu",
            activation_clamp=self._activation_clamp,
            fast_math=self._fast_math,
            assume_all_topk_valid=True,
        )
