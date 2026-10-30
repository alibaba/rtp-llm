"""GLM-5 GC MoE with FP8 shared experts fused into the GC operator."""

from __future__ import annotations

import logging

import torch

from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe import (
    _sync_cuda_graph_warmup_ranks,
)
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_se import GLM5MegaMoESE
from rtp_llm.models_py.modules.glm5_mega_moe.quant_layouts_se import (
    FP4_BLOCK,
    SHARED_WEIGHT_RECIPE,
)
from rtp_llm.models_py.modules.glm5_mega_moe.shared_fp8_scale import (
    stage_shared_fp8_input_scales,
)

from .gc_buf import get_or_create_gc_clone_buffer
from .gc_moe import (
    clone_gc_module,
    gc_warmup_token_counts,
    setup_gc_buffer,
    transform_gc_routed_fp4,
)
from .input_packer_se import get_gc_moe_se_input_packer

logger = logging.getLogger(__name__)


class GLM5GCMoESE(GLM5MegaMoESE):
    """Keep Mega's checkpoint preparation but use only GC runtime objects."""

    def _transform_routed_fp4_weights(self, l1, l2, w1_layout: str):
        return transform_gc_routed_fp4(l1, l2, w1_layout)

    def _transform_shared_fp8_weights(self, l1, l2):
        import gc_moe

        return gc_moe.transform_weights_for_gc_moe(l1, l2, activation="swiglu")

    def _setup_buffer_and_warmup(self) -> None:
        if self._num_shared_experts == 0:
            return
        if self.cfg.n_activated_experts + 1 > 32:
            raise ValueError("gc_moe_se needs topk plus one shared slot <= 32")
        setup_gc_buffer(self, self._num_shared_experts)
        self._input_packer = get_gc_moe_se_input_packer()
        self._maybe_warmup_jit_once()

    def _maybe_warmup_jit_once(self) -> None:
        if self._num_shared_experts == 0 or getattr(self, "_gc_jit_warmed", False):
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "GC MoE SE JIT eager warmup cannot run during graph capture"
            )
        counts = gc_warmup_token_counts(self.cfg.max_tokens_per_rank)
        logger.info(
            "[GLM5 GC MoE SE] eager warmup: layer=%d shared=%d tokens=%s",
            self.cfg.layer_id,
            self._num_shared_experts,
            counts,
        )
        self._warmup_jit(counts)
        self._gc_jit_warmed = True

    def clone_for_cuda_graph(self, *, share_mega_buf: bool = False) -> "GLM5GCMoESE":
        clone = clone_gc_module(
            self, self._num_shared_experts, get_gc_moe_se_input_packer()
        )
        if share_mega_buf:
            clone._mega_buf = self._mega_buf
        else:
            clone._mega_buf = get_or_create_gc_clone_buffer(
                self._mega_buf, self._mega_group, self.cfg, self._num_shared_experts
            )
        clone._gc_jit_warmed = share_mega_buf and self._gc_jit_warmed
        if not clone._gc_jit_warmed:
            clone._maybe_warmup_jit_once()
        return clone

    def _forward_impl(self, x, weights, indices, *, inputs_prepacked: bool):
        import gc_moe

        self._check_shared_expert_ready()
        tokens = int(x.size(0))
        buf = self._mega_buf
        if tokens > buf.num_max_tokens_per_rank or tokens > self._mega_y.size(0):
            raise RuntimeError("GLM5 GC MoE SE input exceeds actual buffer capacity")
        block_m = int(
            gc_moe.get_block_m_for_gc_moe(
                self.cfg.ep_size,
                self.cfg.n_routed_experts,
                buf.num_max_tokens_per_rank,
                tokens,
                self.cfg.n_activated_experts,
                "fp8xfp4",
            )
        )
        if inputs_prepacked:
            stage_shared_fp8_input_scales(
                buf.x_sf, buf.shared_l1_acts_sf, tokens, block_m
            )
        else:
            if weights is None or indices is None:
                raise ValueError("GC MoE SE routing weights and ids are required")
            self._input_packer.pack(x, weights, indices, buf, tokens, block_m)
        _sync_cuda_graph_warmup_ranks(
            f"glm5.gc_moe_se.layer{self.cfg.layer_id}.before_gc", x.device
        )
        y = self._mega_y[:tokens]
        gc_moe.fp8_fp4_gc_moe(
            y,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            buf,
            shared_l1_weights=(self._shared_l1_w, self._shared_l1_sf),
            shared_l2_weights=(self._shared_l2_w, self._shared_l2_sf),
            recipe=(1, 1, FP4_BLOCK),
            shared_recipe=SHARED_WEIGHT_RECIPE,
            activation="swiglu",
            activation_clamp=None,
            fast_math=False,
        )
        return y
