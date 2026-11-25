"""GLM-5 GC MoE with FP8 shared experts fused into the GC operator."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import is_dataclass, replace
from functools import lru_cache
from importlib import metadata

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

_adaptive_tile32 = ContextVar("adaptive_gc_moe_se_tile32", default=False)


@contextmanager
def adaptive_gc_moe_se_tile32():
    token = _adaptive_tile32.set(True)
    try:
        yield
    finally:
        _adaptive_tile32.reset(token)


@lru_cache(maxsize=1)
def _tile32_api(gc_moe):
    # This wheel exposes the checked call object but has no public block_m override.
    # Until the wheel exposes a public override, only use its known call contract.
    try:
        if metadata.version("gc_moe") != "0.6.1+ce37121.cu132":
            return None
        from gc_moe import gc as public
        from gc_moe.apis import gc_moe as api
        from gc_moe.jit_kernels.impls.sm100_fp8_fp4_gc_moe import launch_gc_moe
    except ImportError:
        return None
    config_type = getattr(api, "ValidatedCall", None)
    if (
        getattr(public, "fp8_fp4_gc_moe", None) is not gc_moe.fp8_fp4_gc_moe
        or getattr(api, "get_block_m_for_gc_moe", None)
        is not gc_moe.get_block_m_for_gc_moe
        or not is_dataclass(config_type)
        or not {"block_m", "shared_experts"}.issubset(config_type.__dataclass_fields__)
        or not callable(getattr(api, "validate_call", None))
        or not callable(launch_gc_moe)
    ):
        return None
    return api.validate_call, launch_gc_moe


def _eligible_tile32(x, cfg, buf, tokens, stock_block_m):
    return (
        _adaptive_tile32.get()
        and tokens in (56, 64)
        and stock_block_m == 16
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.shape == (tokens, 6144)
        and (cfg.ep_size, cfg.n_routed_experts, cfg.n_activated_experts) == (4, 256, 8)
        and (buf.hidden, buf.intermediate_hidden) == (6144, 2048)
        and (buf.group.size(), buf.num_experts, buf.num_topk) == (4, 256, 8)
        and buf.mma_type == "fp8xfp4"
        and buf.num_shared_experts == 1
        and callable(getattr(buf, "_for_call", None))
        and torch.cuda.get_device_capability(x.device) == (10, 3)
    )


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

    def _forward_impl(
        self, x, weights, indices, *, inputs_prepacked: bool, activation_clamp=None
    ):
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
        y = self._mega_y[:tokens]
        l1 = (self._mega_l1_w, self._mega_l1_sf)
        l2 = (self._mega_l2_w, self._mega_l2_sf)
        call_args = dict(
            shared_l1_weights=(self._shared_l1_w, self._shared_l1_sf),
            shared_l2_weights=(self._shared_l2_w, self._shared_l2_sf),
            recipe=(1, 1, FP4_BLOCK),
            shared_recipe=SHARED_WEIGHT_RECIPE,
            activation="swiglu",
            activation_clamp=activation_clamp,
            fast_math=False,
        )
        tile32 = (
            _tile32_api(gc_moe)
            if _eligible_tile32(x, self.cfg, buf, tokens, block_m)
            else None
        )
        config = None
        if tile32 is not None:
            validate_call, launch_gc_moe = tile32
            validated = validate_call(
                y,
                l1,
                l2,
                buf,
                **call_args,
                cumulative_local_expert_recv_stats=None,
                activation_alpha=None,
            )
            if validated.block_m == block_m and validated.shared_experts == 1:
                block_m = 32
                config = replace(validated, block_m=block_m)
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
        if config is None:
            gc_moe.fp8_fp4_gc_moe(y, l1, l2, buf, **call_args)
        else:
            launch_gc_moe(
                config,
                y,
                l1,
                l2,
                buf._for_call(config.shared_experts),
                call_args["shared_l1_weights"],
                call_args["shared_l2_weights"],
                None,
            )
        return y
