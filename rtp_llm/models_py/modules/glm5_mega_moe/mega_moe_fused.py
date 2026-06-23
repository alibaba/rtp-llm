"""GLM-5 MegaMoE: fused routed experts plus FP8 shared expert.

``deep_gemm.fp8_fp4_mega_moe_fused`` keeps the routed experts in FP4
(per-group, gran_k=32) but consumes the shared expert as **FP8 e4m3 weights
with 128×128 per-block UE8M0 scale factors**. The FP8 shared-expert weight
transform + scratch workspace live in :mod:`.mega_fused_buf`.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from .input_packer import get_mega_moe_input_packer
from .jit_warmup import format_token_counts, mega_moe_jit_warmup_enabled
from .mega_buf import get_or_create_mega_output
from .mega_fused_buf import (
    get_or_create_mega_buf_fused,
    make_shared_mid_workspace,
    transform_shared_expert_fp8_for_fused,
)
from .mega_moe import (
    GLM5MegaMoE,
    GLM5MegaMoeCfg,
    _mega_output_capacity,
    _sync_cuda_graph_warmup_ranks,
)
from .quant_layouts import FP4_BLOCK

logger = logging.getLogger(__name__)

_MEGA_MOE_FUSED_JIT_WARMED_KEYS: set[tuple] = set()
_CUDA_GRAPH_CLONE_FUSED_BUF_CACHE: dict[tuple, object] = {}


def _get_or_create_cuda_graph_clone_buf_fused(src_buf, group, cfg: GLM5MegaMoeCfg):
    if src_buf is None or group is None:
        return src_buf
    key = (
        id(src_buf),
        id(group),
        cfg.n_routed_experts,
        cfg.max_tokens_per_rank,
        cfg.n_activated_experts,
        cfg.dim,
        cfg.moe_inter_dim,
    )
    cached = _CUDA_GRAPH_CLONE_FUSED_BUF_CACHE.get(key)
    if cached is not None:
        return cached

    import deep_gemm

    cached = deep_gemm.get_symm_buffer_for_mega_moe_fused(
        group=group,
        num_experts=cfg.n_routed_experts,
        num_max_tokens_per_rank=max(cfg.max_tokens_per_rank, 1),
        num_topk=cfg.n_activated_experts,
        hidden=cfg.dim,
        intermediate_hidden=cfg.moe_inter_dim,
        use_fp8_dispatch=True,
        activation="swiglu",
    )
    _CUDA_GRAPH_CLONE_FUSED_BUF_CACHE[key] = cached
    logger.info(
        "[GLM5 MegaMoE Fused] allocated CUDA graph clone symm buffer: layer=%d "
        "max_tokens_per_rank=%d hidden=%d",
        cfg.layer_id,
        cfg.max_tokens_per_rank,
        cfg.dim,
    )
    return cached


class GLM5MegaMoEFused(GLM5MegaMoE):
    """GLM-5 MegaMoE wrapper for ``fp8_fp4_mega_moe_fused``."""

    def __init__(self, cfg: GLM5MegaMoeCfg):
        super().__init__(cfg)
        self._shared_l1_w: Optional[torch.Tensor] = None
        self._shared_l1_sf: Optional[torch.Tensor] = None
        self._shared_l2_w: Optional[torch.Tensor] = None
        self._shared_l2_sf: Optional[torch.Tensor] = None
        self._shared_mid_fp8: Optional[torch.Tensor] = None
        self._shared_mid_sf: Optional[torch.Tensor] = None

    def _setup_buffer_and_warmup(self) -> None:
        """Allocate the fused symmetric-memory buffer (``SymmBufferFused``).

        ``fp8_fp4_mega_moe_fused`` needs its own buffer type, distinct from the
        routed ``fp8_fp4_mega_moe`` buffer the base class allocates. The JIT
        warmup is deferred to :meth:`maybe_warmup_fused_shared_jit_once` (it must
        run after the shared-expert weights are set up, since the fused kernel
        consumes both routed and shared weights together).
        """
        import torch.distributed as dist

        cfg = self.cfg
        device = self._mega_l1_w.device

        assert (
            dist.is_initialized()
        ), "GLM5 MegaMoE Fused requires torch.distributed initialised"
        group = dist.group.WORLD
        self._mega_group = group

        self._mega_buf = get_or_create_mega_buf_fused(
            group=group,
            num_experts=cfg.n_routed_experts,
            num_max_tokens_per_rank=max(cfg.max_tokens_per_rank, 1),
            num_topk=cfg.n_activated_experts,
            hidden=cfg.dim,
            intermediate_hidden=cfg.moe_inter_dim,
            use_fp8_dispatch=True,
            activation="swiglu",
        )
        self._mega_y = get_or_create_mega_output(
            _mega_output_capacity(self._mega_buf, cfg.max_tokens_per_rank),
            cfg.dim,
            torch.bfloat16,
            device,
        )
        self._input_packer = get_mega_moe_input_packer()

    def clone_for_cuda_graph(self) -> "GLM5MegaMoEFused":
        clone = super().clone_for_cuda_graph()
        clone._mega_buf = _get_or_create_cuda_graph_clone_buf_fused(
            self._mega_buf, self._mega_group, self.cfg
        )
        clone._shared_l1_w = self._shared_l1_w
        clone._shared_l1_sf = self._shared_l1_sf
        clone._shared_l2_w = self._shared_l2_w
        clone._shared_l2_sf = self._shared_l2_sf
        clone._shared_mid_fp8 = (
            torch.empty_like(self._shared_mid_fp8)
            if self._shared_mid_fp8 is not None
            else None
        )
        clone._shared_mid_sf = (
            torch.empty_like(self._shared_mid_sf)
            if self._shared_mid_sf is not None
            else None
        )
        return clone

    def setup_shared_expert_from_fp8(
        self,
        w1_w: torch.Tensor,  # [2*inter, dim] float8_e4m3fn (gate||up stacked on N)
        w1_s: torch.Tensor,  # [2*inter//128, dim//128] float32 per-block SF
        w2_w: torch.Tensor,  # [dim, inter] float8_e4m3fn
        w2_s: torch.Tensor,  # [dim//128, inter//128] float32 per-block SF
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

    def maybe_warmup_fused_shared_jit_once(self) -> None:
        """Compile fused routed+shared MegaMoE JIT buckets once per shape."""
        if not mega_moe_jit_warmup_enabled():
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MegaMoE fused JIT warmup must not run inside CUDA graph capture"
            )
        self._check_shared_expert_ready()

        import deep_gemm
        import torch.distributed as dist

        cfg = self.cfg
        num_sms = int(deep_gemm.get_num_sms())
        token_counts = self._resolve_jit_warmup_token_counts(num_sms)
        if not token_counts:
            return

        warmup_key = (
            "fused_shared",
            cfg.ep_size,
            cfg.n_routed_experts,
            cfg.n_local_experts,
            cfg.n_activated_experts,
            cfg.dim,
            cfg.moe_inter_dim,
            int(cfg.max_tokens_per_rank),
            cfg.swiglu_limit,
            num_sms,
            tuple(token_counts),
        )
        if warmup_key in _MEGA_MOE_FUSED_JIT_WARMED_KEYS:
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            logger.info(
                "[GLM5 MegaMoE] fused JIT warmup start: layer=%d tokens=[%s] "
                "max_tokens_per_rank=%d ep=%d experts=%d topk=%d hidden=%d "
                "intermediate=%d num_sms=%d",
                cfg.layer_id,
                format_token_counts(token_counts),
                cfg.max_tokens_per_rank,
                cfg.ep_size,
                cfg.n_routed_experts,
                cfg.n_activated_experts,
                cfg.dim,
                cfg.moe_inter_dim,
                num_sms,
            )
        self._warmup_fused_shared_jit(token_counts)
        _MEGA_MOE_FUSED_JIT_WARMED_KEYS.add(warmup_key)
        if rank == 0:
            logger.info(
                "[GLM5 MegaMoE] fused JIT warmup done: layer=%d tokens=[%s]",
                cfg.layer_id,
                format_token_counts(token_counts),
            )

    @torch.inference_mode()
    def _warmup_fused_shared_jit(self, token_counts: list[int]) -> None:
        """Compile fused routed+shared MegaMoE JIT buckets with synthetic tokens."""
        import torch.distributed as dist

        cfg = self.cfg
        device = self._mega_l1_w.device
        max_tokens = max(token_counts)
        x = torch.zeros((max_tokens, cfg.dim), dtype=torch.bfloat16, device=device)
        weights = torch.zeros(
            (max_tokens, cfg.n_activated_experts),
            dtype=torch.float32,
            device=device,
        )
        local_expert_ids = cfg.local_expert_start + torch.arange(
            cfg.n_activated_experts, dtype=torch.long, device=device
        ) % max(cfg.n_local_experts, 1)
        indices = local_expert_ids.view(1, -1).expand(max_tokens, -1).contiguous()

        for token_count in token_counts:
            dist.barrier()
            self.forward_with_shared_expert(
                x[:token_count],
                weights[:token_count],
                indices[:token_count],
            )
            torch.cuda.synchronize(device)
        dist.barrier()

    def forward_with_shared_expert(
        self,
        x: torch.Tensor,  # [T, D] BF16 local-rank tokens
        weights: torch.Tensor,  # [T, topk] FP32 router weights
        indices: torch.Tensor,  # [T, topk] int64 GLOBAL expert IDs
    ) -> torch.Tensor:
        """Run fused routed MegaMoE plus the FP8 shared expert in one kernel."""
        import deep_gemm

        self._check_shared_expert_ready()

        T = x.size(0)
        buf = self._mega_buf
        if T > buf.num_max_tokens_per_rank:
            raise RuntimeError(
                f"GLM5 MegaMoE input tokens={T} exceeds num_max_tokens_per_rank="
                f"{buf.num_max_tokens_per_rank}. Raise the budget at startup."
            )
        if T > self._mega_y.size(0):
            raise RuntimeError(
                f"GLM5 MegaMoE output buffer rows={self._mega_y.size(0)} is smaller "
                f"than input tokens={T}. This indicates inconsistent aligned "
                "MegaMoE buffer sizing."
            )
        if T > self._shared_mid_fp8.size(0):
            raise RuntimeError(
                f"GLM5 MegaMoE shared workspace rows={self._shared_mid_fp8.size(0)} "
                f"is smaller than input tokens={T}."
            )

        self._input_packer.pack(x, weights, indices, buf, T)
        self._maybe_pre_kernel_barrier(T)
        _sync_cuda_graph_warmup_ranks(
            f"glm5.mega_moe_fused.layer{self.cfg.layer_id}.before_deepgemm",
            x.device,
        )

        y = self._mega_y[:T]
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
            buf,
            recipe=(1, 1, FP4_BLOCK),
            activation="swiglu",
            activation_clamp=None,  # (self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None),
            fast_math=False,
        )
        return y
