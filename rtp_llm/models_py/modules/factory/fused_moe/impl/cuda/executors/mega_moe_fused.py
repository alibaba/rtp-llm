"""Legacy fused routed-FP4/shared-FP8 MegaMoE kernel recipe."""

from __future__ import annotations

import torch

from rtp_llm.models_py.kernels.cuda.quant_layouts import FP4_BLOCK
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.fused_buffer import (
    get_or_create_mega_buf_fused,
    make_shared_mid_workspace,
    transform_shared_expert_fp8_for_fused,
)

from .mega_moe import MegaMoeExecutor, _mega_output_capacity


class MegaMoeFusedBackend(MegaMoeExecutor):
    includes_shared_expert = True
    _defer_jit_until_shared_weights = True
    _kernel_name = "fp8_fp4_mega_moe_fused"
    _buffer_factory = "get_symm_buffer_for_mega_moe_fused"

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
