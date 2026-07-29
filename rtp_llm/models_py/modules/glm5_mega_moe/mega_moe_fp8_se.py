"""FP8xFP8 MegaMoE with the shared expert fused into the mega kernel."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch

from .input_packer_triton import stage_shared_expert_input_scales
from .mega_moe import GLM5MegaMoeCfg
from .mega_moe_fp8 import GLM5MegaMoEFP8, _ceil_div, _infer_fp8_scale_recipe
from .quant_layouts import FP8_BLOCK, prepare_fp8_weight_scale_for_deepgemm

logger = logging.getLogger(__name__)


class GLM5MegaMoEFP8SE(GLM5MegaMoEFP8):
    """DeepGEMM ``fp8_fp8_mega_moe`` with fused FP8 shared experts."""

    def __init__(self, cfg: GLM5MegaMoeCfg):
        super().__init__(cfg)
        self._shared_l1_w: Optional[torch.Tensor] = None
        self._shared_l1_sf: Optional[torch.Tensor] = None
        self._shared_l2_w: Optional[torch.Tensor] = None
        self._shared_l2_sf: Optional[torch.Tensor] = None

    def clone_for_cuda_graph(self) -> "GLM5MegaMoEFP8SE":
        clone = super().clone_for_cuda_graph()
        clone._shared_l1_w = self._shared_l1_w
        clone._shared_l1_sf = self._shared_l1_sf
        clone._shared_l2_w = self._shared_l2_w
        clone._shared_l2_sf = self._shared_l2_sf
        return clone

    def _setup_buffer_and_warmup(self) -> None:
        # Routed weights are installed by MegaMoeWrapper before the wrapper has
        # access to shared weights. Defer allocation so the symmetric buffer is
        # created once with the required shared-expert workspace.
        if self._num_shared_experts == 0:
            return
        super()._setup_buffer_and_warmup()

    @staticmethod
    def _reshape_shared_scale(
        scale: torch.Tensor,
        mn: int,
        k: int,
        recipe: Tuple[int, int],
    ) -> torch.Tensor:
        if scale.dtype == torch.int32:
            return scale
        expected = (_ceil_div(mn, recipe[0]), _ceil_div(k, recipe[1]))
        if tuple(scale.shape) == expected:
            return scale
        if scale.numel() == expected[0] * expected[1]:
            return scale.reshape(expected)
        raise ValueError(
            "FP8 shared-expert scale shape mismatch: "
            f"got {tuple(scale.shape)}, expected {expected} for "
            f"mn={mn}, k={k}, recipe={recipe}"
        )

    def setup_shared_expert_from_fp8(
        self,
        w1_fp8: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_fp8: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> None:
        """Prepare checkpoint FP8 shared weights for fused MegaMoE.

        ``w1_fp8`` is stacked ``[gate | up]`` with shape
        ``[2 * shared_intermediate, hidden]``; ``w2_fp8`` is
        ``[hidden, shared_intermediate]``.
        """
        import deep_gemm

        if not hasattr(deep_gemm, "get_block_m_for_mega_moe_fp8"):
            raise RuntimeError(
                "moe_strategy=mega_moe_fp8_se requires DeepGEMM fused "
                "shared-expert support (get_block_m_for_mega_moe_fp8 is missing)"
            )
        for name, weight in (("w1_fp8", w1_fp8), ("w2_fp8", w2_fp8)):
            if weight.dtype != torch.float8_e4m3fn:
                raise TypeError(
                    "mega_moe_fp8_se requires FP8 e4m3 shared weights; "
                    f"{name} has {weight.dtype}"
                )

        cfg = self.cfg
        hidden = cfg.dim
        if w1_fp8.dim() != 2 or w2_fp8.dim() != 2:
            raise ValueError(
                "FP8 shared-expert weights must be 2D, got "
                f"w1={tuple(w1_fp8.shape)}, w2={tuple(w2_fp8.shape)}"
            )
        shared_intermediate = int(w2_fp8.shape[1])
        expected_w1 = (2 * shared_intermediate, hidden)
        expected_w2 = (hidden, shared_intermediate)
        if tuple(w1_fp8.shape) != expected_w1 or tuple(w2_fp8.shape) != expected_w2:
            raise ValueError(
                "FP8 shared-expert weight shape mismatch: expected "
                f"w1={expected_w1}, w2={expected_w2}; got "
                f"w1={tuple(w1_fp8.shape)}, w2={tuple(w2_fp8.shape)}"
            )
        if shared_intermediate % cfg.moe_inter_dim != 0:
            raise ValueError(
                "shared intermediate width must be an integer multiple of the "
                f"routed expert width: {shared_intermediate} % {cfg.moe_inter_dim} != 0"
            )
        num_shared_experts = shared_intermediate // cfg.moe_inter_dim
        if num_shared_experts <= 0:
            raise ValueError("mega_moe_fp8_se requires at least one shared expert")

        l1_recipe = _infer_fp8_scale_recipe(w1_scale, 2 * shared_intermediate, hidden)
        l2_recipe = _infer_fp8_scale_recipe(w2_scale, hidden, shared_intermediate)
        if l1_recipe != l2_recipe or l1_recipe != self._fp8_weight_recipe:
            raise ValueError(
                "routed and shared FP8 weights must use the same recipe, got "
                f"routed={self._fp8_weight_recipe}, shared_l1={l1_recipe}, "
                f"shared_l2={l2_recipe}"
            )

        w1_scale = self._reshape_shared_scale(
            w1_scale, 2 * shared_intermediate, hidden, l1_recipe
        )
        w2_scale = self._reshape_shared_scale(
            w2_scale, hidden, shared_intermediate, l2_recipe
        )
        if l1_recipe == (FP8_BLOCK, FP8_BLOCK) and (
            w1_scale.dtype == torch.float32 or w2_scale.dtype == torch.float32
        ):
            if w1_scale.dtype != torch.float32 or w2_scale.dtype != torch.float32:
                raise TypeError(
                    "both FP8 shared-expert scales must be raw float32 or "
                    "both prepacked int32"
                )
            from rtp_llm.models_py.kernels.cuda.fp8_kernel import requant_weight_ue8m0

            w1_fp8, w1_scale_int = requant_weight_ue8m0(w1_fp8.contiguous(), w1_scale)
            w2_fp8, w2_scale_int = requant_weight_ue8m0(w2_fp8.contiguous(), w2_scale)
        else:
            w1_scale_int = prepare_fp8_weight_scale_for_deepgemm(
                w1_scale,
                2 * shared_intermediate,
                hidden,
                recipe=l1_recipe,
            )
            w2_scale_int = prepare_fp8_weight_scale_for_deepgemm(
                w2_scale,
                hidden,
                shared_intermediate,
                recipe=l2_recipe,
            )

        with torch.cuda.device(w1_fp8.device):
            (l1_w, l1_sf), (l2_w, l2_sf) = deep_gemm.transform_weights_for_mega_moe_fp8(
                (w1_fp8.contiguous(), w1_scale_int),
                (w2_fp8.contiguous(), w2_scale_int),
            )
        self._shared_l1_w = l1_w
        self._shared_l1_sf = l1_sf
        self._shared_l2_w = l2_w
        self._shared_l2_sf = l2_sf
        self._num_shared_experts = num_shared_experts

        logger.info(
            "[MegaMoE FP8 SE] prepared shared weights: layer=%d shared=%d "
            "shared_intermediate=%d recipe=%s",
            cfg.layer_id,
            num_shared_experts,
            shared_intermediate,
            self._fp8_weight_recipe,
        )
        self._setup_buffer_and_warmup()

    def _fused_shared_expert_kwargs(self, tokens: int) -> Dict[str, Any]:
        import deep_gemm

        if (
            self._shared_l1_w is None
            or self._shared_l1_sf is None
            or self._shared_l2_w is None
            or self._shared_l2_sf is None
        ):
            raise RuntimeError("FP8 fused shared-expert weights are not initialized")
        buf = self._mega_buf
        block_m = deep_gemm.get_block_m_for_mega_moe_fp8(
            buf.group.size(),
            self.cfg.n_routed_experts,
            buf.num_max_tokens_per_rank,
            tokens,
            self.cfg.n_activated_experts,
        )
        stage_shared_expert_input_scales(
            buf.x_sf[:tokens],
            buf.shared_l1_acts_sf,
            tokens,
            block_m,
        )
        return {
            "shared_l1_weights": (self._shared_l1_w, self._shared_l1_sf),
            "shared_l2_weights": (self._shared_l2_w, self._shared_l2_sf),
        }
