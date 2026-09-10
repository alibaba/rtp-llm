"""FP8xFP4 MegaMoE with the FP8 shared expert fused in DeepGEMM."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from .input_packer_triton import stage_shared_expert_input_scales
from .mega_moe import GLM5MegaMoE, GLM5MegaMoeCfg
from .quant_layouts import FP4_BLOCK, MXFP8_BLOCK, prepare_fp8_weight_scale_for_deepgemm

logger = logging.getLogger(__name__)

_SHARED_WEIGHT_RECIPE = (1, MXFP8_BLOCK)


class GLM5MegaMoESE(GLM5MegaMoE):
    """Run routed MXFP4 and shared MXFP8 experts in one mega kernel."""

    def __init__(self, cfg: GLM5MegaMoeCfg):
        super().__init__(cfg)
        self._num_shared_experts = 0
        self._shared_l1_w: Optional[torch.Tensor] = None
        self._shared_l1_sf: Optional[torch.Tensor] = None
        self._shared_l2_w: Optional[torch.Tensor] = None
        self._shared_l2_sf: Optional[torch.Tensor] = None

    def _setup_buffer_and_warmup(self) -> None:
        # MegaMoeWrapper installs routed weights before it can access the
        # shared FFN tensors. Allocate once, after the shared width is known.
        if self._num_shared_experts == 0:
            return
        if self.cfg.n_activated_experts > 31:
            raise ValueError(
                "mega_moe_se requires topk <= 31 because the shared result "
                f"uses one combine slot, got topk={self.cfg.n_activated_experts}"
            )
        super()._setup_buffer_and_warmup()

    @staticmethod
    def _reshape_shared_scale(scale: torch.Tensor, mn: int, k: int) -> torch.Tensor:
        if scale.dtype == torch.int32:
            return scale
        expected = (mn, (k + MXFP8_BLOCK - 1) // MXFP8_BLOCK)
        if tuple(scale.shape) == expected:
            return scale
        if scale.numel() == expected[0] * expected[1]:
            return scale.reshape(expected)
        raise ValueError(
            "MXFP8 shared-expert scale shape mismatch: "
            f"got {tuple(scale.shape)}, expected {expected} for weight "
            f"shape=({mn}, {k})"
        )

    def setup_shared_expert_from_fp8(
        self,
        w1_fp8: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_fp8: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> None:
        """Prepare full-rank shared weights with per-row/K32 MXFP8 scales.

        ``w1_fp8`` follows DeepGEMM's logical ``[gate | up]`` order and has
        shape ``[2 * shared_intermediate, hidden]``. ``w2_fp8`` has shape
        ``[hidden, shared_intermediate]``.
        """
        import deep_gemm

        for name, weight in (("w1_fp8", w1_fp8), ("w2_fp8", w2_fp8)):
            if weight.dtype != torch.float8_e4m3fn:
                raise TypeError(
                    "mega_moe_se requires FP8 e4m3 shared weights; "
                    f"{name} has dtype={weight.dtype}"
                )

        cfg = self.cfg
        if w1_fp8.dim() != 2 or w2_fp8.dim() != 2:
            raise ValueError(
                "shared-expert weights must be 2D, got "
                f"w13={tuple(w1_fp8.shape)}, w2={tuple(w2_fp8.shape)}"
            )
        shared_intermediate = int(w2_fp8.shape[1])
        expected_w1 = (2 * shared_intermediate, cfg.dim)
        expected_w2 = (cfg.dim, shared_intermediate)
        if tuple(w1_fp8.shape) != expected_w1 or tuple(w2_fp8.shape) != expected_w2:
            raise ValueError(
                "shared-expert weight shape mismatch: expected "
                f"w13={expected_w1}, w2={expected_w2}; got "
                f"w13={tuple(w1_fp8.shape)}, w2={tuple(w2_fp8.shape)}"
            )
        if shared_intermediate % cfg.moe_inter_dim != 0:
            raise ValueError(
                "shared intermediate width must be an integer multiple of "
                f"routed width: {shared_intermediate} % {cfg.moe_inter_dim} != 0"
            )
        num_shared_experts = shared_intermediate // cfg.moe_inter_dim
        if num_shared_experts <= 0:
            raise ValueError("mega_moe_se requires at least one shared expert")

        # Routed weights have already selected the current rank's CUDA device.
        # Shared MXFP8 tensors may still reside on CPU when checkpoint loading
        # uses FORCE_CPU_LOAD_WEIGHTS=1, so move only this layer before invoking
        # DeepGEMM's CUDA-only scale/weight layout transforms.
        if self._mega_l1_w is None:
            raise RuntimeError("routed MegaMoE weights must be prepared first")
        device = self._mega_l1_w.device
        with torch.cuda.device(device):
            w1_fp8 = w1_fp8.to(device=device, non_blocking=True)
            w1_scale = w1_scale.to(device=device, non_blocking=True)
            w2_fp8 = w2_fp8.to(device=device, non_blocking=True)
            w2_scale = w2_scale.to(device=device, non_blocking=True)

        w1_scale = self._reshape_shared_scale(
            w1_scale, 2 * shared_intermediate, cfg.dim
        )
        w2_scale = self._reshape_shared_scale(w2_scale, cfg.dim, shared_intermediate)
        w1_sf = prepare_fp8_weight_scale_for_deepgemm(
            w1_scale,
            2 * shared_intermediate,
            cfg.dim,
            recipe=_SHARED_WEIGHT_RECIPE,
        )
        w2_sf = prepare_fp8_weight_scale_for_deepgemm(
            w2_scale,
            cfg.dim,
            shared_intermediate,
            recipe=_SHARED_WEIGHT_RECIPE,
        )

        with torch.cuda.device(device):
            (shared_l1_w, shared_l1_sf), (
                shared_l2_w,
                shared_l2_sf,
            ) = deep_gemm.transform_weights_for_mega_moe(
                (w1_fp8.contiguous(), w1_sf),
                (w2_fp8.contiguous(), w2_sf),
                activation=self._activation_name,
            )
        self._shared_l1_w = shared_l1_w
        self._shared_l1_sf = shared_l1_sf
        self._shared_l2_w = shared_l2_w
        self._shared_l2_sf = shared_l2_sf
        self._num_shared_experts = num_shared_experts
        logger.info(
            "[GLM5 MegaMoE SE] prepared MXFP8 shared weights: layer=%d "
            "shared_experts=%d shared_intermediate=%d recipe=(1,1,%d)",
            cfg.layer_id,
            num_shared_experts,
            shared_intermediate,
            MXFP8_BLOCK,
        )
        self._setup_buffer_and_warmup()

    def clone_for_cuda_graph(self) -> "GLM5MegaMoESE":
        clone = super().clone_for_cuda_graph()
        clone._shared_l1_w = self._shared_l1_w
        clone._shared_l1_sf = self._shared_l1_sf
        clone._shared_l2_w = self._shared_l2_w
        clone._shared_l2_sf = self._shared_l2_sf
        clone._num_shared_experts = self._num_shared_experts
        return clone

    def _fused_shared_expert_kwargs(self, tokens: int) -> Dict[str, Any]:
        import deep_gemm

        if any(
            tensor is None
            for tensor in (
                self._shared_l1_w,
                self._shared_l1_sf,
                self._shared_l2_w,
                self._shared_l2_sf,
            )
        ):
            raise RuntimeError("mega_moe_se shared-expert weights are not initialized")
        buf = self._mega_buf
        block_m = deep_gemm.get_block_m_for_mega_moe(
            int(buf.group.size()),
            self.cfg.n_routed_experts,
            buf.num_max_tokens_per_rank,
            tokens,
            self.cfg.n_activated_experts,
            "fp8xfp4",
        )
        stage_shared_expert_input_scales(
            buf.x_sf[:tokens],
            buf.shared_l1_acts_sf,
            tokens,
            int(block_m),
        )
        return {
            "shared_l1_weights": (self._shared_l1_w, self._shared_l1_sf),
            "shared_l2_weights": (self._shared_l2_w, self._shared_l2_sf),
            "shared_recipe": (1, 1, FP4_BLOCK),
        }


__all__ = ["GLM5MegaMoESE"]
