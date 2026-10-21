"""FP8 routed and shared experts using the common MegaMoE executor."""

from __future__ import annotations

from rtp_llm.models_py.kernels.cuda.quant_layouts import FP4_BLOCK
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_fp8_scale import (
    stage_shared_fp8_input_scales,
)

from .mega_moe_fp8 import MegaMoeFp8Backend


class MegaMoeFp8SharedBackend(MegaMoeFp8Backend):
    includes_shared_expert = True
    _defer_buffer_until_shared_weights = True
    _defer_jit_until_shared_weights = True

    def _buffer_kwargs(self):
        kwargs = super()._buffer_kwargs()
        kwargs["num_shared_experts"] = self._num_shared_experts
        return kwargs

    def _allocate_mega_buffer(self):
        if self.cfg.n_activated_experts > 31:
            raise ValueError(
                "MegaMoE FP8 SE requires topk <= 31 because shared output uses one combine slot"
            )
        buf = super()._allocate_mega_buffer()
        if int(buf.num_shared_experts) != self._num_shared_experts:
            raise RuntimeError("DeepGEMM FP8 buffer/shared weight mismatch")
        return buf

    def _stage_kernel_inputs(self, tokens, *, inputs_prepacked):
        import deep_gemm

        block_m = deep_gemm.get_block_m_for_mega_moe_fp8(
            self.cfg.ep_size,
            self.cfg.n_routed_experts,
            self._mega_buf.num_max_tokens_per_rank,
            tokens,
            self.cfg.n_activated_experts,
        )
        stage_shared_fp8_input_scales(
            self._mega_buf.x_sf, self._mega_buf.shared_l1_acts_sf, tokens, block_m
        )

    def _run_kernel(self, y, tokens):
        import deep_gemm

        deep_gemm.fp8_fp8_mega_moe(
            y,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            self._mega_buf,
            shared_l1_weights=(self._shared_l1_w, self._shared_l1_sf),
            shared_l2_weights=(self._shared_l2_w, self._shared_l2_sf),
            recipe=(1, 1, FP4_BLOCK),
            activation="swiglu",
            activation_clamp=self._activation_clamp,
            fast_math=self._fast_math,
            assume_all_topk_valid=True,
        )
