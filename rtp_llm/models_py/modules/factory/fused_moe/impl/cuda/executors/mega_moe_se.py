"""Shared FP8 experts fused with routed FP4 or FP8 experts in MegaMoE."""

from __future__ import annotations

import torch

from rtp_llm.config.mega_moe import FP8_MEGA_MOE_STRATEGIES
from rtp_llm.models_py.kernels.cuda.quant_layouts import FP4_BLOCK
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.fused_buffer import (
    get_or_create_mega_buf_fused,
    make_shared_mid_workspace,
    transform_shared_expert_fp8_for_fused,
)
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
        if getattr(config, "moe_strategy", "mega_moe_se") in FP8_MEGA_MOE_STRATEGIES:
            return
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
        kwargs["num_shared_experts"] = self._num_shared_experts
        if not self._fp8_routed:
            kwargs.pop("use_fp8_dispatch")
            kwargs["mma_type"] = _MMA_TYPE
        return kwargs

    def _allocate_mega_buffer(self):
        if self.cfg.n_activated_experts > 31:
            raise ValueError(
                "MegaMoE SE requires topk <= 31 because shared output uses one combine slot"
            )
        if self._fp8_routed:
            buf = super()._allocate_mega_buffer()
        else:
            kwargs = self._buffer_kwargs()
            kwargs.pop("mma_type")
            buf = _get_or_create_mega_se_buf(**kwargs)
        actual = getattr(buf, "num_shared_experts", None)
        if actual is not None and int(actual) != self._num_shared_experts:
            raise RuntimeError("DeepGEMM MegaMoE SE buffer/shared weight mismatch")
        return buf

    def _allocate_mega_output(self):
        if self._fp8_routed:
            return super()._allocate_mega_output()
        return _get_or_create_mega_se_output(
            _mega_output_capacity(self._mega_buf, self.cfg.max_tokens_per_rank),
            self.cfg.dim,
            torch.bfloat16,
            self._mega_l1_w.device,
        )

    def _block_m(self, tokens: int) -> int:
        import deep_gemm

        args = (
            self.cfg.ep_size,
            self.cfg.n_routed_experts,
            self._mega_buf.num_max_tokens_per_rank,
            int(tokens),
            self.cfg.n_activated_experts,
        )
        if self._fp8_routed:
            return int(deep_gemm.get_block_m_for_mega_moe_fp8(*args))
        return int(deep_gemm.get_block_m_for_mega_moe(*args, _MMA_TYPE))

    def _stage_kernel_inputs(self, tokens, *, inputs_prepacked):
        stage_shared_fp8_input_scales(
            self._mega_buf.x_sf,
            self._mega_buf.shared_l1_acts_sf,
            tokens,
            self._block_m(tokens),
        )

    def _run_kernel(self, y, tokens):
        import deep_gemm

        kwargs = (
            {"assume_all_topk_valid": True}
            if self._fp8_routed
            else {"shared_recipe": _SHARED_RECIPE}
        )
        getattr(deep_gemm, self._kernel_name)(
            y,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            self._mega_buf,
            shared_l1_weights=(self._shared_l1_w, self._shared_l1_sf),
            shared_l2_weights=(self._shared_l2_w, self._shared_l2_sf),
            recipe=_ROUTED_RECIPE,
            activation="swiglu",
            activation_clamp=self._activation_clamp,
            fast_math=self._fast_math,
            **kwargs,
        )


class MegaMoeFusedBackend(MegaMoeExecutor):
    """Adapter for the legacy fused kernel's separate shared-workspace ABI."""

    includes_shared_expert = True
    _fp4_transform_name = "transform_weights_for_mega_moe_fused"
    _defer_jit_until_shared_weights = True

    @classmethod
    def check_conditions(cls, checker, config) -> None:
        super().check_conditions(checker, config)
        try:
            import deep_gemm
        except ImportError:
            checker.check(False)
            return

        checker.check(hasattr(deep_gemm, cls._fp4_transform_name))
        checker.check(
            hasattr(deep_gemm, "transform_shared_expert_weights_for_mega_moe_fused")
        )

    def _allocate_mega_buffer(self):
        return get_or_create_mega_buf_fused(**self._buffer_kwargs())

    def setup_shared_expert_from_fp8(
        self,
        w1_w: torch.Tensor,  # [2*inter, dim] float8_e4m3fn (gate||up stacked on N)
        w1_s: torch.Tensor,  # float32 / e8m0 block SF, or packed int32
        w2_w: torch.Tensor,  # [dim, inter] float8_e4m3fn
        w2_s: torch.Tensor,  # float32 / e8m0 block SF, or packed int32
    ) -> None:
        """Setup pre-quantized FP8 per-block shared-expert weights for fused MegaMoE."""
        cfg = self.cfg
        (l1_w, l1_sf), (l2_w, l2_sf) = transform_shared_expert_fp8_for_fused(
            w1_w,
            w1_s,
            w2_w,
            w2_s,
            dim=cfg.dim,
            inter=cfg.moe_inter_dim,
        )
        self._shared_l1_w = l1_w
        self._shared_l1_sf = l1_sf
        self._shared_l2_w = l2_w
        self._shared_l2_sf = l2_sf
        self._setup_shared_expert_workspace()

    def _setup_shared_expert_workspace(self) -> None:
        if self._mega_buf is None:
            raise RuntimeError(
                "setup routed MegaMoE weights before shared expert weights"
            )
        cfg = self.cfg
        device = self._mega_l1_w.device
        capacity = _mega_output_capacity(self._mega_buf, cfg.max_tokens_per_rank)
        self._shared_mid_fp8, self._shared_mid_sf = make_shared_mid_workspace(
            capacity,
            cfg.moe_inter_dim,
            device,
        )

    def _check_shared_expert_ready(self) -> None:
        if (
            self._shared_l1_w is None
            or self._shared_l1_sf is None
            or self._shared_l2_w is None
            or self._shared_l2_sf is None
            or self._shared_mid_fp8 is None
            or self._shared_mid_sf is None
        ):
            raise RuntimeError("shared expert FP8 weights/workspace are not set up")

    def _validate_capacity(self, tokens):
        super()._validate_capacity(tokens)
        if tokens > self._shared_mid_fp8.size(0):
            raise RuntimeError(
                f"MegaMoE shared workspace rows={self._shared_mid_fp8.size(0)} "
                f"is smaller than input tokens={tokens}"
            )

    def _run_kernel(self, y, tokens):
        import deep_gemm

        deep_gemm.fp8_fp4_mega_moe_fused(
            y,
            self._shared_l1_w,
            self._shared_l1_sf,
            self._shared_l2_w,
            self._shared_l2_sf,
            self._shared_mid_fp8,
            self._shared_mid_sf,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            self._mega_buf,
            recipe=(1, 1, FP4_BLOCK),
            activation="swiglu",
            activation_clamp=self._activation_clamp,
            fast_math=self._fast_math,
        )


__all__ = ["MegaMoeSEExecutor", "MegaMoeFusedBackend"]
