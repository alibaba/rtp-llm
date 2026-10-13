"""GLM-5 FP8xFP8 MegaMoE with the shared expert fused in DeepGEMM."""

from __future__ import annotations

import logging
from typing import Optional

import torch

from rtp_llm.models_py.kernels.cuda.mega_moe_quant_layouts import FP4_BLOCK
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_backend import (
    _mega_output_capacity,
    _sync_cuda_graph_warmup_ranks,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8 import (
    MegaMoeFp8Backend,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.backend_buffer import (
    get_or_create_mega_output,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.backend_input_packer import (
    get_mega_moe_input_packer,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.fp8_buffer import (
    get_or_create_mega_buf_fp8,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_fp8_scale import (
    stage_shared_fp8_input_scales,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_weights import (
    prepare_shared_expert_weights,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.topology import (
    validated_world_ep_group,
)

logger = logging.getLogger(__name__)


class MegaMoeFp8SharedBackend(MegaMoeFp8Backend):
    """Run routed and shared FP8 experts in ``fp8_fp8_mega_moe``."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._shared_l1_w: Optional[torch.Tensor] = None
        self._shared_l1_sf: Optional[torch.Tensor] = None
        self._shared_l2_w: Optional[torch.Tensor] = None
        self._shared_l2_sf: Optional[torch.Tensor] = None

    def _setup_buffer_and_warmup(self) -> None:
        # Routed weights are installed by MegaMoeWrapper before it exposes the
        # shared weights. Defer allocation until both weight sets are ready.
        if self._num_shared_experts == 0:
            return

        import torch.distributed as dist

        cfg = self.cfg
        device = self._mega_l1_w.device
        assert (
            dist.is_initialized()
        ), "GLM5 MegaMoE FP8 SE requires torch.distributed initialised"
        if cfg.n_activated_experts > 31:
            raise ValueError(
                "mega_moe_fp8_se requires topk <= 31 because the shared expert "
                f"uses one of 32 combine slots, got topk={cfg.n_activated_experts}"
            )
        group = validated_world_ep_group(self.cfg, dist)
        self._mega_group = group
        self._mega_buf = get_or_create_mega_buf_fp8(
            group=group,
            num_experts=cfg.n_routed_experts,
            num_max_tokens_per_rank=max(cfg.max_tokens_per_rank, 1),
            num_topk=cfg.n_activated_experts,
            hidden=cfg.dim,
            intermediate_hidden=cfg.moe_inter_dim,
            num_shared_experts=self._num_shared_experts,
            use_fp8_dispatch=True,
            activation="swiglu",
        )
        if int(self._mega_buf.num_shared_experts) != self._num_shared_experts:
            raise RuntimeError(
                "DeepGEMM FP8 buffer/shared weight mismatch: "
                f"buffer={self._mega_buf.num_shared_experts}, "
                f"weights={self._num_shared_experts}"
            )
        self._mega_y = get_or_create_mega_output(
            _mega_output_capacity(self._mega_buf, cfg.max_tokens_per_rank),
            cfg.dim,
            torch.bfloat16,
            device,
        )
        self._input_packer = get_mega_moe_input_packer()

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
            fp8_routed=True,
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
            raise RuntimeError("mega_moe_fp8_se shared expert weights are not set up")

    def clone_for_cuda_graph(
        self, *, share_mega_buf: bool = False
    ) -> "MegaMoeFp8SharedBackend":
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
        tokens = x.size(0)
        buf = self._mega_buf
        if tokens > buf.num_max_tokens_per_rank:
            raise RuntimeError(
                f"GLM5 MegaMoE FP8 SE input tokens={tokens} exceeds "
                f"num_max_tokens_per_rank={buf.num_max_tokens_per_rank}"
            )
        if tokens > self._mega_y.size(0):
            raise RuntimeError(
                f"GLM5 MegaMoE FP8 SE output rows={self._mega_y.size(0)} "
                f"is smaller than input tokens={tokens}"
            )

        if not inputs_prepacked:
            if weights is None or indices is None:
                raise ValueError("weights and indices are required before packing")
            self._input_packer.pack(x, weights, indices, buf, tokens)
        block_m = deep_gemm.get_block_m_for_mega_moe_fp8(
            self.cfg.ep_size,
            self.cfg.n_routed_experts,
            buf.num_max_tokens_per_rank,
            tokens,
            self.cfg.n_activated_experts,
        )
        stage_shared_fp8_input_scales(
            buf.x_sf,
            buf.shared_l1_acts_sf,
            tokens,
            block_m,
        )
        _sync_cuda_graph_warmup_ranks(
            f"glm5.mega_moe_fp8_se.layer{self.cfg.layer_id}.before_deepgemm",
            x.device,
        )

        y = self._mega_y[:tokens]
        deep_gemm.fp8_fp8_mega_moe(
            y,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            buf,
            shared_l1_weights=(self._shared_l1_w, self._shared_l1_sf),
            shared_l2_weights=(self._shared_l2_w, self._shared_l2_sf),
            recipe=(1, 1, FP4_BLOCK),
            activation="swiglu",
            activation_clamp=self.cfg.activation_clamp,
            fast_math=self.cfg.fast_math,
            assume_all_topk_valid=True,
        )
        return y
