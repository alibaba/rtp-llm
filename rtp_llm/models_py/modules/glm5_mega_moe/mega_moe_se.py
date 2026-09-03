"""GLM-5 FP8xFP4 MegaMoE with the shared expert fused in DeepGEMM.

This strategy intentionally uses the unified ``fp8_fp4_mega_moe`` optional-
shared API.  It does not use the legacy ``fp8_fp4_mega_moe_fused`` operator,
its dedicated buffer type, or an external shared intermediate workspace.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from .input_packer_se import get_mega_moe_se_input_packer
from .jit_warmup_se import (
    clamp_token_counts,
    format_token_counts,
    generate_mega_moe_se_jit_token_counts,
    mega_moe_se_jit_warmup_enabled,
    parse_mega_moe_se_jit_warmup_tokens_override,
)
from .mega_moe import GLM5MegaMoE, _mega_output_capacity, _sync_cuda_graph_warmup_ranks
from .mega_se_buf import (
    get_or_create_mega_moe_se_buf,
    get_or_create_mega_moe_se_clone_buf,
    get_or_create_mega_moe_se_output,
)
from .quant_layouts_se import (
    FP4_BLOCK,
    MXFP8_SHARED_WEIGHT_RECIPE,
    SHARED_WEIGHT_RECIPE,
    prepare_shared_fp8_scale_for_mega_moe_se,
)
from .shared_fp8_scale import stage_shared_fp8_input_scales

logger = logging.getLogger(__name__)

_MEGA_MOE_SE_JIT_WARMED_KEYS: set[tuple] = set()


class GLM5MegaMoESE(GLM5MegaMoE):
    """Run routed FP4 experts and dense shared FP8 experts in one kernel."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._num_shared_experts = 0
        self._shared_l1_w: Optional[torch.Tensor] = None
        self._shared_l1_sf: Optional[torch.Tensor] = None
        self._shared_l2_w: Optional[torch.Tensor] = None
        self._shared_l2_sf: Optional[torch.Tensor] = None
        self._shared_weight_recipe = SHARED_WEIGHT_RECIPE

    @staticmethod
    def _infer_shared_scale_recipe(
        scale: torch.Tensor, mn: int, k: int
    ) -> Tuple[int, int]:
        """Infer legacy block-FP8 versus ModelOpt MXFP8 shared scales."""
        if scale.dtype == torch.int32:
            return SHARED_WEIGHT_RECIPE[1:]

        trailing = tuple(scale.shape[-2:])
        expected_mxfp8 = (mn, (k + 31) // 32)
        if trailing == expected_mxfp8 or scale.numel() == (
            expected_mxfp8[0] * expected_mxfp8[1]
        ):
            return MXFP8_SHARED_WEIGHT_RECIPE[1:]

        expected_legacy = ((mn + 127) // 128, (k + 127) // 128)
        if trailing == expected_legacy or scale.numel() == (
            expected_legacy[0] * expected_legacy[1]
        ):
            return SHARED_WEIGHT_RECIPE[1:]

        raise ValueError(
            "Cannot infer MegaMoE-SE shared FP8 scale recipe from "
            f"shape={tuple(scale.shape)} for weight ({mn}, {k}); expected "
            f"trailing dims {expected_mxfp8} for MXFP8 or "
            f"{expected_legacy} for legacy block FP8"
        )

    def _setup_buffer_and_warmup(self) -> None:
        """Defer allocation until both routed and shared weights are ready."""
        if self._num_shared_experts == 0:
            return

        import torch.distributed as dist

        cfg = self.cfg
        if not dist.is_initialized():
            raise RuntimeError(
                "GLM5 MegaMoE SE requires torch.distributed to be initialized"
            )
        if cfg.n_activated_experts > 31:
            raise ValueError(
                "mega_moe_se requires topk <= 31 because shared output uses "
                f"one combine slot, got topk={cfg.n_activated_experts}"
            )

        self._mega_group = dist.group.WORLD
        self._mega_buf = get_or_create_mega_moe_se_buf(
            group=self._mega_group,
            num_experts=cfg.n_routed_experts,
            num_max_tokens_per_rank=max(cfg.max_tokens_per_rank, 1),
            num_topk=cfg.n_activated_experts,
            hidden=cfg.dim,
            intermediate_hidden=cfg.moe_inter_dim,
            num_shared_experts=self._num_shared_experts,
            activation="swiglu",
        )
        # ``num_shared_experts`` is a construction argument, but it is not a
        # public attribute on deep-gemm 2.6.1's Python ``SymmBuffer``.  Some
        # builds expose it for diagnostics, so validate only when available.
        buffer_num_shared_experts = getattr(self._mega_buf, "num_shared_experts", None)
        if (
            buffer_num_shared_experts is not None
            and int(buffer_num_shared_experts) != self._num_shared_experts
        ):
            raise RuntimeError(
                "DeepGEMM MegaMoE SE buffer/shared weight mismatch: "
                f"buffer={buffer_num_shared_experts}, "
                f"weights={self._num_shared_experts}"
            )
        self._mega_y = get_or_create_mega_moe_se_output(
            _mega_output_capacity(self._mega_buf, cfg.max_tokens_per_rank),
            cfg.dim,
            torch.bfloat16,
            self._mega_l1_w.device,
        )
        self._input_packer = get_mega_moe_se_input_packer()

    def setup_shared_expert_from_fp8(
        self,
        w1_w: torch.Tensor,
        w1_s: torch.Tensor,
        w2_w: torch.Tensor,
        w2_s: torch.Tensor,
    ) -> None:
        """Transform full-rank FP8 shared weights for unified MegaMoE."""
        import deep_gemm

        cfg = self.cfg
        if w1_w.dtype != torch.float8_e4m3fn or w2_w.dtype != torch.float8_e4m3fn:
            raise TypeError(
                "mega_moe_se requires FP8 e4m3 shared weights, got "
                f"w13={w1_w.dtype}, w2={w2_w.dtype}"
            )

        shared_intermediate = int(w2_w.shape[1])
        if shared_intermediate % cfg.moe_inter_dim != 0:
            raise ValueError(
                "shared intermediate size must be a multiple of the routed "
                f"intermediate size, got {shared_intermediate} and {cfg.moe_inter_dim}"
            )
        num_shared_experts = shared_intermediate // cfg.moe_inter_dim
        if num_shared_experts <= 0:
            raise ValueError("mega_moe_se requires at least one shared expert")

        expected_w1 = (2 * shared_intermediate, cfg.dim)
        expected_w2 = (cfg.dim, shared_intermediate)
        if tuple(w1_w.shape) != expected_w1 or tuple(w2_w.shape) != expected_w2:
            raise ValueError(
                "shared expert weight shape mismatch: "
                f"expected w13={expected_w1}, w2={expected_w2}; "
                f"got w13={tuple(w1_w.shape)}, w2={tuple(w2_w.shape)}"
            )

        w1_recipe = self._infer_shared_scale_recipe(
            w1_s, 2 * shared_intermediate, cfg.dim
        )
        w2_recipe = self._infer_shared_scale_recipe(
            w2_s, cfg.dim, shared_intermediate
        )
        if w1_recipe != w2_recipe:
            raise ValueError(
                "mega_moe_se requires the same shared FP8 scale recipe for "
                f"w13 and w2, got {w1_recipe} and {w2_recipe}"
            )
        self._shared_weight_recipe = (1, *w1_recipe)

        # Legacy 128x128 float32 inverse scales must be requantized to native
        # UE8M0. ModelOpt MXFP8 fp32 scales are already exact powers of two and
        # only need the same 1x32 DeepGEMM layout transform used by vLLM.
        if w1_recipe == SHARED_WEIGHT_RECIPE[1:] and (
            w1_s.dtype == torch.float32 or w2_s.dtype == torch.float32
        ):
            if w1_s.dtype != torch.float32 or w2_s.dtype != torch.float32:
                raise TypeError(
                    "mega_moe_se requires both shared scales to be raw float32 "
                    f"or both native/packed UE8M0, got {w1_s.dtype} and {w2_s.dtype}"
                )
            from rtp_llm.models_py.kernels.cuda.fp8_kernel import requant_weight_ue8m0

            w1_w, w1_sf = requant_weight_ue8m0(w1_w.contiguous(), w1_s)
            w2_w, w2_sf = requant_weight_ue8m0(w2_w.contiguous(), w2_s)
        else:
            w1_sf = prepare_shared_fp8_scale_for_mega_moe_se(
                w1_s,
                2 * shared_intermediate,
                cfg.dim,
                recipe=w1_recipe,
            )
            w2_sf = prepare_shared_fp8_scale_for_mega_moe_se(
                w2_s,
                cfg.dim,
                shared_intermediate,
                recipe=w2_recipe,
            )

        (self._shared_l1_w, self._shared_l1_sf), (
            self._shared_l2_w,
            self._shared_l2_sf,
        ) = deep_gemm.transform_weights_for_mega_moe(
            (w1_w.contiguous(), w1_sf),
            (w2_w.contiguous(), w2_sf),
        )
        self._num_shared_experts = num_shared_experts
        logger.info(
            "[GLM5 MegaMoE SE] prepared shared weights: layer=%d "
            "shared_experts=%d shared_intermediate=%d",
            cfg.layer_id,
            num_shared_experts,
            shared_intermediate,
        )
        self._setup_buffer_and_warmup()

    def maybe_warmup_fused_shared_jit_once(self) -> None:
        self._maybe_warmup_jit_once()

    def _resolve_jit_warmup_token_counts(self, num_sms: int) -> list[int]:
        cfg = self.cfg
        override = parse_mega_moe_se_jit_warmup_tokens_override()
        if override is not None:
            return clamp_token_counts(override, cfg.max_tokens_per_rank)
        return generate_mega_moe_se_jit_token_counts(
            num_ranks=cfg.ep_size,
            num_experts=cfg.n_routed_experts,
            num_experts_per_rank=cfg.n_local_experts,
            num_topk=cfg.n_activated_experts,
            intermediate_hidden=cfg.moe_inter_dim,
            num_sms=num_sms,
            max_tokens_per_rank=cfg.max_tokens_per_rank,
        )

    def _maybe_warmup_jit_once(self) -> None:
        if self._num_shared_experts == 0 or not mega_moe_se_jit_warmup_enabled():
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MegaMoE SE JIT warmup must not run inside CUDA graph capture"
            )

        import deep_gemm
        import torch.distributed as dist

        cfg = self.cfg
        num_sms = int(deep_gemm.get_num_sms())
        token_counts = self._resolve_jit_warmup_token_counts(num_sms)
        if not token_counts:
            return
        warmup_key = (
            "fp8_fp4_mega_moe_se",
            cfg.ep_size,
            cfg.n_routed_experts,
            cfg.n_local_experts,
            cfg.n_activated_experts,
            cfg.dim,
            cfg.moe_inter_dim,
            int(cfg.max_tokens_per_rank),
            self._num_shared_experts,
            self._shared_weight_recipe,
            cfg.swiglu_limit,
            num_sms,
            tuple(token_counts),
        )
        if warmup_key in _MEGA_MOE_SE_JIT_WARMED_KEYS:
            return
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            logger.info(
                "[GLM5 MegaMoE SE] JIT warmup start: layer=%d tokens=[%s] "
                "shared_experts=%d",
                cfg.layer_id,
                format_token_counts(token_counts),
                self._num_shared_experts,
            )
        self._warmup_jit(token_counts)
        _MEGA_MOE_SE_JIT_WARMED_KEYS.add(warmup_key)
        if rank == 0:
            logger.info(
                "[GLM5 MegaMoE SE] JIT warmup done: layer=%d tokens=[%s]",
                cfg.layer_id,
                format_token_counts(token_counts),
            )

    def _check_shared_expert_ready(self) -> None:
        if any(
            tensor is None
            for tensor in (
                self._shared_l1_w,
                self._shared_l1_sf,
                self._shared_l2_w,
                self._shared_l2_sf,
            )
        ):
            raise RuntimeError("mega_moe_se shared expert weights are not set up")

    def clone_for_cuda_graph(self) -> "GLM5MegaMoESE":
        clone = object.__new__(type(self))
        torch.nn.Module.__init__(clone)
        clone.cfg = self.cfg
        clone._mega_l1_w = self._mega_l1_w
        clone._mega_l1_sf = self._mega_l1_sf
        clone._mega_l2_w = self._mega_l2_w
        clone._mega_l2_sf = self._mega_l2_sf
        clone._shared_l1_w = self._shared_l1_w
        clone._shared_l1_sf = self._shared_l1_sf
        clone._shared_l2_w = self._shared_l2_w
        clone._shared_l2_sf = self._shared_l2_sf
        clone._num_shared_experts = self._num_shared_experts
        clone._shared_weight_recipe = self._shared_weight_recipe
        clone._mega_buf = get_or_create_mega_moe_se_clone_buf(
            self._mega_buf,
            self._mega_group,
            self.cfg,
            self._num_shared_experts,
        )
        clone._mega_y = (
            torch.empty_like(self._mega_y) if self._mega_y is not None else None
        )
        clone._input_packer = get_mega_moe_se_input_packer()
        clone._mega_group = self._mega_group
        return clone

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_impl(x, weights, indices, inputs_prepacked=False)

    def _forward_impl(
        self,
        x: torch.Tensor,
        weights: torch.Tensor | None,
        indices: torch.Tensor | None,
        *,
        inputs_prepacked: bool,
    ) -> torch.Tensor:
        import deep_gemm

        self._check_shared_expert_ready()
        tokens = int(x.size(0))
        buf = self._mega_buf
        if tokens > buf.num_max_tokens_per_rank:
            raise RuntimeError(
                f"GLM5 MegaMoE SE input tokens={tokens} exceeds "
                f"num_max_tokens_per_rank={buf.num_max_tokens_per_rank}"
            )
        if tokens > self._mega_y.size(0):
            raise RuntimeError(
                f"GLM5 MegaMoE SE output rows={self._mega_y.size(0)} is "
                f"smaller than input tokens={tokens}"
            )

        block_m = deep_gemm.get_block_m_for_mega_moe(
            self.cfg.ep_size,
            self.cfg.n_routed_experts,
            buf.num_max_tokens_per_rank,
            tokens,
            self.cfg.n_activated_experts,
            "fp8xfp4",
        )
        block_m = int(block_m)
        if inputs_prepacked:
            stage_shared_fp8_input_scales(
                buf.x_sf,
                buf.shared_l1_acts_sf,
                tokens,
                block_m,
            )
        else:
            if weights is None or indices is None:
                raise ValueError("weights and indices are required before packing")
            self._input_packer.pack(
                x,
                weights,
                indices,
                buf,
                tokens,
                block_m,
            )
        self._maybe_pre_kernel_barrier(tokens)
        _sync_cuda_graph_warmup_ranks(
            f"glm5.mega_moe_se.layer{self.cfg.layer_id}.before_deepgemm",
            x.device,
        )

        y = self._mega_y[:tokens]
        deep_gemm.fp8_fp4_mega_moe(
            y,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            buf,
            shared_l1_weights=(self._shared_l1_w, self._shared_l1_sf),
            shared_l2_weights=(self._shared_l2_w, self._shared_l2_sf),
            recipe=(1, 1, FP4_BLOCK),
            shared_recipe=self._shared_weight_recipe,
            activation="swiglu",
            activation_clamp=(
                self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None
            ),
            fast_math=False,
        )
        return y


__all__ = ["GLM5MegaMoESE"]
