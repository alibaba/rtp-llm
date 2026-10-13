"""GLM-5 FP8xFP4 MegaMoE with the shared expert fused in DeepGEMM.

This strategy intentionally uses the unified ``fp8_fp4_mega_moe`` optional-
shared API.  It does not use the legacy ``fp8_fp4_mega_moe_fused`` operator,
its dedicated buffer type, or an external shared intermediate workspace.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from rtp_llm.models_py.kernels.cuda.mega_moe_shared_quant_layouts import (
    FP4_BLOCK,
    SHARED_WEIGHT_RECIPE,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_backend import (
    MegaMoeBackend,
    _mega_output_capacity,
    _sync_cuda_graph_warmup_ranks,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.backend_jit_warmup import (
    format_token_counts,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.backend_jit_warmup import (
    generate_mega_moe_jit_token_counts as generate_mega_moe_se_jit_token_counts,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_buffer import (
    get_or_create_mega_moe_se_buf,
    get_or_create_mega_moe_se_clone_buf,
    get_or_create_mega_moe_se_output,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_fp8_scale import (
    stage_shared_fp8_input_scales,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_input_packer import (
    get_mega_moe_se_input_packer,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_weights import (
    prepare_shared_expert_weights,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.topology import (
    validated_world_ep_group,
)

logger = logging.getLogger(__name__)

_MEGA_MOE_SE_JIT_WARMED_KEYS: set[tuple] = set()


class MegaMoeSharedBackend(MegaMoeBackend):
    """Run routed FP4 experts and dense shared FP8 experts in one kernel."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._shared_l1_w: Optional[torch.Tensor] = None
        self._shared_l1_sf: Optional[torch.Tensor] = None
        self._shared_l2_w: Optional[torch.Tensor] = None
        self._shared_l2_sf: Optional[torch.Tensor] = None

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

        self._mega_group = validated_world_ep_group(self.cfg, dist)
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
        weights = prepare_shared_expert_weights(
            self.cfg,
            w1_w,
            w1_s,
            w2_w,
            w2_s,
            fp8_routed=False,
        )
        self._shared_l1_w, self._shared_l1_sf = weights.l1
        self._shared_l2_w, self._shared_l2_sf = weights.l2
        self._num_shared_experts = weights.num_experts
        logger.info(
            "GLM5 MegaMoE shared weights prepared: layer=%d shared_experts=%d",
            self.cfg.layer_id,
            weights.num_experts,
        )
        self._setup_buffer_and_warmup()

    def maybe_warmup_fused_shared_jit_once(self) -> None:
        self._maybe_warmup_jit_once()

    def _resolve_jit_warmup_token_counts(self, num_sms: int) -> list[int]:
        cfg = self.cfg
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
        if self._num_shared_experts == 0:
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
            type(self).__name__,
            id(self._mega_group),
            self._num_shared_experts,
            str(self._mega_l1_w.device),
            "fp8_fp4_mega_moe_se",
            cfg.ep_size,
            cfg.n_routed_experts,
            cfg.n_local_experts,
            cfg.n_activated_experts,
            cfg.dim,
            cfg.moe_inter_dim,
            int(cfg.max_tokens_per_rank),
            self._num_shared_experts,
            SHARED_WEIGHT_RECIPE,
            cfg.activation_clamp,
            cfg.fast_math,
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

    def _clone_mega_buffer(self):
        return get_or_create_mega_moe_se_clone_buf(
            self._mega_buf, self._mega_group, self.cfg, self._num_shared_experts
        )

    def _clone_input_packer(self):
        return get_mega_moe_se_input_packer()

    def clone_for_cuda_graph(
        self, *, share_mega_buf: bool = False
    ) -> "MegaMoeSharedBackend":
        clone = super().clone_for_cuda_graph(share_mega_buf=share_mega_buf)
        clone._shared_l1_w = self._shared_l1_w
        clone._shared_l1_sf = self._shared_l1_sf
        clone._shared_l2_w = self._shared_l2_w
        clone._shared_l2_sf = self._shared_l2_sf
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
            shared_recipe=SHARED_WEIGHT_RECIPE,
            activation="swiglu",
            activation_clamp=self.cfg.activation_clamp,
            fast_math=self.cfg.fast_math,
        )
        return y


__all__ = ["MegaMoeSharedBackend"]
