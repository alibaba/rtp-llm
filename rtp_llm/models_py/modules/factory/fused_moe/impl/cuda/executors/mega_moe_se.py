"""Shared FP8 experts fused with routed FP4 experts in MegaMoE."""

from __future__ import annotations

import torch

from rtp_llm.models_py.kernels.cuda.quant_layouts import FP4_BLOCK
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.se_buffer import (
    _get_or_create_mega_se_buf,
    _get_or_create_mega_se_output,
    _mega_moe_se_available,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_fp8_scale import (
    stage_shared_fp8_input_scales,
)

from .mega_moe import MegaMoeExecutor, _mega_output_capacity

_ROUTED_RECIPE = (1, 1, FP4_BLOCK)
_SHARED_RECIPE = (1, 128, 128)
_MMA_TYPE = "fp8xfp4"


class MegaMoeSEExecutor(MegaMoeExecutor):
    includes_shared_expert = True
    _defer_buffer_until_shared_weights = True
    _defer_jit_until_shared_weights = True

    @classmethod
    def check_conditions(cls, checker, config) -> None:
        super().check_conditions(checker, config)
        checker.check(config.n_shared_experts > 0)
        checker.check(not config.has_shared_expert_gate)
        checker.check(
            _mega_moe_se_available(
                allow_single_rank=getattr(config, "mega_moe_policy", None) is not None
            )
        )

    def setup_weights(self, layer_weights):
        if self._prepared_policy is not None:
            return super().setup_weights(layer_weights)
        from rtp_llm.utils.model_weight import W

        super().setup_weights(layer_weights)
        self.setup_shared_expert_from_fp8(
            layer_weights.pop(W.ffn_w13),
            layer_weights.pop(W.ffn_s13),
            layer_weights.pop(W.ffn_w2),
            layer_weights.pop(W.ffn_s2),
        )
        self.maybe_warmup_fused_shared_jit_once()

    def _buffer_kwargs(self):
        kwargs = super()._buffer_kwargs()
        kwargs.pop("use_fp8_dispatch")
        kwargs.update(num_shared_experts=self._num_shared_experts, mma_type=_MMA_TYPE)
        return kwargs

    def _allocate_mega_buffer(self):
        if self.cfg.n_activated_experts > 31:
            raise ValueError(
                "MegaMoE SE requires topk <= 31 because shared output uses one combine slot"
            )
        kwargs = self._buffer_kwargs()
        kwargs.pop("mma_type")
        buf = _get_or_create_mega_se_buf(**kwargs)
        actual = getattr(buf, "num_shared_experts", None)
        if actual is not None and int(actual) != self._num_shared_experts:
            raise RuntimeError("DeepGEMM MegaMoE SE buffer/shared weight mismatch")
        return buf

    def _allocate_mega_output(self):
        return _get_or_create_mega_se_output(
            _mega_output_capacity(self._mega_buf, self.cfg.max_tokens_per_rank),
            self.cfg.dim,
            torch.bfloat16,
            self._mega_l1_w.device,
        )

    def _block_m(self, tokens: int) -> int:
        import deep_gemm

        return int(
            deep_gemm.get_block_m_for_mega_moe(
                self.cfg.ep_size,
                self.cfg.n_routed_experts,
                self._mega_buf.num_max_tokens_per_rank,
                int(tokens),
                self.cfg.n_activated_experts,
                _MMA_TYPE,
            )
        )

    def _stage_kernel_inputs(self, tokens, *, inputs_prepacked):
        stage_shared_fp8_input_scales(
            self._mega_buf.x_sf,
            self._mega_buf.shared_l1_acts_sf,
            tokens,
            self._block_m(tokens),
        )

    def _run_kernel(self, y, tokens):
        import deep_gemm

        deep_gemm.fp8_fp4_mega_moe(
            y,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            self._mega_buf,
            shared_l1_weights=(self._shared_l1_w, self._shared_l1_sf),
            shared_l2_weights=(self._shared_l2_w, self._shared_l2_sf),
            recipe=_ROUTED_RECIPE,
            shared_recipe=_SHARED_RECIPE,
            activation="swiglu",
            activation_clamp=self._activation_clamp,
            fast_math=self._fast_math,
        )


__all__ = ["MegaMoeSEExecutor"]
