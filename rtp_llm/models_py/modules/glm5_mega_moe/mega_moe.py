"""GLM-5 MegaMoE: DeepGEMM fp8_fp4_mega_moe strategy.

Ported from feat/dsv4_on_dev branch's MegaMoEStrategy.
Adapted for GLM-5 shapes:
  - hidden_size: 6144
  - moe_intermediate_size: 2048
  - n_routed_experts: 256
  - num_experts_per_tok: 8
  - FP8 per-block [128,128] weights (converted to FP4 at load time)

The mega kernel fuses dispatch + L1 GEMM + SwiGLU + L2 GEMM + combine
into one kernel backed by a PyTorch symmetric-memory buffer for NVLink
communication. Requires SM100, PyTorch >= 2.9, DeepGEMM >= 2.5.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

try:
    from rtp_llm.ops.compute_ops import cuda_graph_warmup_forward_enabled
except ImportError:

    def cuda_graph_warmup_forward_enabled() -> bool:
        return False


from .input_packer import get_mega_moe_input_packer
from .jit_warmup import (
    clamp_token_counts,
    format_token_counts,
    generate_mega_moe_jit_token_counts,
    mega_moe_jit_warmup_enabled,
    parse_jit_warmup_tokens_override,
)
from .mega_buf import (
    get_or_create_mega_buf,
    get_or_create_mega_output,
    mega_moe_enabled,
)
from .quant_layouts import FP4_BLOCK, prepare_fp4_weight_scale_for_deepgemm

logger = logging.getLogger(__name__)

_MEGA_MOE_JIT_WARMED_KEYS: set[tuple] = set()
_CUDA_GRAPH_CLONE_BUF_CACHE: Dict[tuple, object] = {}
_PRE_KERNEL_BARRIER_ENV = "GLM5_MEGA_MOE_PRE_KERNEL_BARRIER"
_PRE_KERNEL_BARRIER_VERBOSE_ENV = "GLM5_MEGA_MOE_PRE_KERNEL_BARRIER_VERBOSE"
_PRE_KERNEL_BARRIER_LOGGED_KEYS: set[tuple[int, int]] = set()


def _mega_output_capacity(buf, requested_capacity: int) -> int:
    """Output rows must cover DeepGEMM's internally aligned token capacity."""
    capacity = max(int(requested_capacity), 1)
    aligned_capacity = getattr(buf, "num_max_tokens_per_rank", None)
    if aligned_capacity is not None:
        capacity = max(capacity, int(aligned_capacity))
    return capacity


def _get_or_create_cuda_graph_clone_buf(src_buf, group, cfg: GLM5MegaMoeCfg):
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
    cached = _CUDA_GRAPH_CLONE_BUF_CACHE.get(key)
    if cached is not None:
        return cached

    import deep_gemm

    cached = deep_gemm.get_symm_buffer_for_mega_moe(
        group=group,
        num_experts=cfg.n_routed_experts,
        num_max_tokens_per_rank=max(cfg.max_tokens_per_rank, 1),
        num_topk=cfg.n_activated_experts,
        hidden=cfg.dim,
        intermediate_hidden=cfg.moe_inter_dim,
        use_fp8_dispatch=True,
        activation="swiglu",
    )
    _CUDA_GRAPH_CLONE_BUF_CACHE[key] = cached
    logging.info(
        "[GLM5 MegaMoE] allocated CUDA graph clone symm buffer: layer=%d "
        "max_tokens_per_rank=%d hidden=%d",
        cfg.layer_id,
        cfg.max_tokens_per_rank,
        cfg.dim,
    )
    return cached


def _pre_kernel_barrier_enabled() -> bool:
    return os.environ.get(_PRE_KERNEL_BARRIER_ENV, "0") == "1"


def _pre_kernel_barrier_verbose_enabled() -> bool:
    return os.environ.get(_PRE_KERNEL_BARRIER_VERBOSE_ENV, "0") == "1"


def _log_pre_kernel_barrier(
    phase: str,
    layer_id: int,
    rank: int,
    world_size: int,
    tokens: int,
    device: torch.device,
) -> None:
    if _pre_kernel_barrier_verbose_enabled():
        logger.info(
            "[GLM5 MegaMoE] pre-kernel barrier %s: layer=%d rank=%d/%d "
            "tokens=%d device=%s",
            phase,
            layer_id,
            rank,
            world_size,
            tokens,
            device,
        )
        return

    if phase != "enter":
        return
    key = (layer_id, rank)
    if key in _PRE_KERNEL_BARRIER_LOGGED_KEYS:
        return
    _PRE_KERNEL_BARRIER_LOGGED_KEYS.add(key)
    logger.info(
        "[GLM5 MegaMoE] pre-kernel barrier enabled: layer=%d rank=%d/%d "
        "tokens=%d device=%s; set %s=1 to log every barrier",
        layer_id,
        rank,
        world_size,
        tokens,
        device,
        _PRE_KERNEL_BARRIER_VERBOSE_ENV,
    )


def _sync_cuda_graph_warmup_ranks(
    _phase: str, device: torch.device | None = None
) -> None:
    if not cuda_graph_warmup_forward_enabled():
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return

    world_size = torch.distributed.get_world_size()
    if world_size <= 1:
        return

    if torch.cuda.is_available():
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        else:
            torch.cuda.synchronize()

    from rtp_llm.models_py.distributed import collective_torch

    collective_torch.barrier(collective_torch.Group.DP_AND_TP)


@dataclass(frozen=True)
class GLM5MegaMoeCfg:
    """Configuration for GLM-5 MegaMoE."""

    layer_id: int
    dim: int  # hidden_size = 6144
    moe_inter_dim: int  # moe_intermediate_size = 2048
    n_routed_experts: int  # 256
    n_activated_experts: int  # top_k = 8
    swiglu_limit: float  # activation clamp (0 = no clamp)
    swiglu_alpha: float  # OAI SwiGLU alpha (0 = regular SwiGLU)
    ep_size: int
    ep_rank: int
    n_local_experts: int
    local_expert_start: int
    local_expert_end: int
    max_tokens_per_rank: int


class GLM5MegaMoE(nn.Module):
    """GLM-5 MegaMoE: fused DeepGEMM fp8_fp4_mega_moe kernel.

    This module handles:
    1. Loading and converting FP8 weights to FP4 format
    2. Allocating symmetric memory buffer
    3. Packing inputs (BF16 → FP8 + UE8M0)
    4. Calling the fused mega kernel
    5. JIT warmup at initialization

    Usage:
        cfg = GLM5MegaMoeCfg(...)
        moe = GLM5MegaMoE(cfg)
        moe.setup_weights(layer_weights)
        y = moe(x, weights, indices)
    """

    def __init__(self, cfg: GLM5MegaMoeCfg):
        super().__init__()
        self.cfg = cfg
        self._mega_l1_w: Optional[torch.Tensor] = None
        self._mega_l1_sf: Optional[torch.Tensor] = None
        self._mega_l2_w: Optional[torch.Tensor] = None
        self._mega_l2_sf: Optional[torch.Tensor] = None
        self._mega_buf = None
        self._mega_y: Optional[torch.Tensor] = None
        self._input_packer = None
        self._mega_group = None

    def clone_for_cuda_graph(self) -> "GLM5MegaMoE":
        clone = object.__new__(type(self))
        nn.Module.__init__(clone)
        clone.cfg = self.cfg
        clone._mega_l1_w = self._mega_l1_w
        clone._mega_l1_sf = self._mega_l1_sf
        clone._mega_l2_w = self._mega_l2_w
        clone._mega_l2_sf = self._mega_l2_sf
        clone._mega_buf = _get_or_create_cuda_graph_clone_buf(
            self._mega_buf, self._mega_group, self.cfg
        )
        clone._mega_y = (
            torch.empty_like(self._mega_y) if self._mega_y is not None else None
        )
        clone._input_packer = get_mega_moe_input_packer()
        clone._mega_group = self._mega_group
        return clone

    @classmethod
    def from_params(
        cls,
        layer_id: int,
        dim: int = 6144,
        moe_inter_dim: int = 2048,
        n_routed_experts: int = 256,
        n_activated_experts: int = 8,
        swiglu_limit: float = 0.0,
        swiglu_alpha: float = 0.0,
        ep_size: int = 4,
        ep_rank: int = 0,
        max_tokens_per_rank: int = 8192,
    ) -> "GLM5MegaMoE":
        """Create GLM5MegaMoE with GLM-5 default parameters."""
        n_local_experts = n_routed_experts // max(ep_size, 1)
        local_expert_start = ep_rank * n_local_experts
        cfg = GLM5MegaMoeCfg(
            layer_id=layer_id,
            dim=dim,
            moe_inter_dim=moe_inter_dim,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            ep_size=ep_size,
            ep_rank=ep_rank,
            n_local_experts=n_local_experts,
            local_expert_start=local_expert_start,
            local_expert_end=local_expert_start + n_local_experts,
            max_tokens_per_rank=max_tokens_per_rank,
        )
        return cls(cfg)

    def setup_weights_from_fp4(
        self,
        w1_w: torch.Tensor,  # [E_local, 2*inter, dim//2] int8 (FP4 packed, gate+up)
        w1_s: torch.Tensor,  # [E_local, 2*inter, dim//FP4_BLOCK] float8_e8m0fnu
        w2_w: torch.Tensor,  # [E_local, dim, inter//2] int8 (FP4 packed)
        w2_s: torch.Tensor,  # [E_local, dim, inter//FP4_BLOCK] float8_e8m0fnu
    ) -> None:
        """Setup weights from pre-quantized FP4 format (same as DSv4 checkpoint).

        If your checkpoint already has FP4 weights (int8 packed + UE8M0 scales),
        use this method directly.
        """
        import deep_gemm
        import torch.distributed as dist

        cfg = self.cfg
        E = cfg.n_local_experts
        D = cfg.dim
        inter = cfg.moe_inter_dim
        device = w1_w.device

        # Transform scales to DeepGEMM int32 layout
        s13_int = prepare_fp4_weight_scale_for_deepgemm(w1_s, 2 * inter, D, E)
        s2_int = prepare_fp4_weight_scale_for_deepgemm(w2_s, D, inter, E)

        # Apply mega MoE transform: L1 gate/up interleave + UTCCP transpose SF
        (l1_w, l1_sf), (l2_w, l2_sf) = deep_gemm.transform_weights_for_mega_moe(
            (w1_w, s13_int),
            (w2_w, s2_int),
        )
        del s13_int, s2_int
        torch.cuda.empty_cache()

        self._mega_l1_w = l1_w
        self._mega_l1_sf = l1_sf
        self._mega_l2_w = l2_w
        self._mega_l2_sf = l2_sf

        self._setup_buffer_and_warmup()

    def setup_weights_from_fp8(
        self,
        w1_fp8: torch.Tensor,  # [E_local, N_out, K_in] float8_e4m3fn (gate projection)
        w1_scale: torch.Tensor,  # [E_local, N_out, K_in//128] float32
        w2_fp8: torch.Tensor,  # [E_local, K_out, K_in] float8_e4m3fn (down projection)
        w2_scale: torch.Tensor,  # [E_local, K_out, K_in//128] float32
        w3_fp8: torch.Tensor,  # [E_local, N_out, K_in] float8_e4m3fn (up projection)
        w3_scale: torch.Tensor,  # [E_local, N_out, K_in//128] float32
    ) -> None:
        """Setup weights from FP8 per-block format (GLM-5 checkpoint).

        Converts FP8 → BF16 → FP4 for the mega kernel.
        w1 = gate, w2 = down, w3 = up (following DeepSeek/GLM convention).
        """
        import deep_gemm
        from deep_gemm.utils import per_token_cast_to_fp4

        cfg = self.cfg
        E = cfg.n_local_experts
        D = cfg.dim
        inter = cfg.moe_inter_dim
        device = w1_fp8.device
        fp8_block = 128

        logger.info(
            "[GLM5 MegaMoE] Converting FP8 weights to FP4: layer=%d E=%d D=%d inter=%d",
            cfg.layer_id,
            E,
            D,
            inter,
        )

        # --- L1 (gate + up): dequant FP8 → BF16 → FP4 ---
        # Gate: [E, inter, D] and Up: [E, inter, D] → combined [E, 2*inter, D]
        def _dequant_fp8(w_fp8, w_scale):
            """Dequantize FP8 per-block to BF16."""
            shape = w_fp8.shape  # [E, N, K]
            E_, N_, K_ = shape
            if K_ % fp8_block != 0:
                raise ValueError(
                    f"FP8 MoE K dimension must be divisible by {fp8_block}, "
                    f"got weight shape={tuple(shape)}"
                )
            if not torch.is_floating_point(w_scale):
                raise ValueError(
                    "MegaMoE FP8 fallback requires raw floating-point FP8 "
                    "checkpoint scales. Got packed/transformed scale "
                    f"dtype={w_scale.dtype}, shape={tuple(w_scale.shape)}. "
                    "For moe_strategy=mega_moe, MoE weights should be wrapped "
                    "to FP4 at load time before MegaMoeWrapper initialization."
                )
            n_blocks_k = K_ // fp8_block
            n_blocks_n = (N_ + fp8_block - 1) // fp8_block
            if w_scale.shape == (E_, N_, n_blocks_k):
                scale_per_row = w_scale
            elif w_scale.shape == (E_, n_blocks_n, n_blocks_k):
                scale_per_row = w_scale.repeat_interleave(fp8_block, dim=1)[:, :N_, :]
            elif w_scale.numel() == E_ * N_ * n_blocks_k:
                scale_per_row = w_scale.reshape(E_, N_, n_blocks_k)
            elif w_scale.numel() == E_ * n_blocks_n * n_blocks_k:
                scale_per_block = w_scale.reshape(E_, n_blocks_n, n_blocks_k)
                scale_per_row = scale_per_block.repeat_interleave(fp8_block, dim=1)[
                    :, :N_, :
                ]
            else:
                raise ValueError(
                    "Cannot interpret FP8 MoE scale shape "
                    f"{tuple(w_scale.shape)} for weight shape={tuple(shape)}"
                )
            w_f = w_fp8.float().view(E_, N_, n_blocks_k, fp8_block)
            s_exp = scale_per_row.to(dtype=w_f.dtype).unsqueeze(-1).expand_as(w_f)
            return (w_f * s_exp).reshape(E_, N_, K_).to(torch.bfloat16)

        w1_bf16 = _dequant_fp8(w1_fp8, w1_scale)  # [E, inter, D]
        w3_bf16 = _dequant_fp8(w3_fp8, w3_scale)  # [E, inter, D]
        del w1_fp8, w1_scale, w3_fp8, w3_scale

        # Combine gate + up into [E, 2*inter, D]
        w13_bf16 = torch.cat([w1_bf16, w3_bf16], dim=1)  # [E, 2*inter, D]
        del w1_bf16, w3_bf16
        torch.cuda.empty_cache()

        # Quantize L1 to FP4: [E, 2*inter, D] → packed [E, 2*inter, D//2] + sf
        w13_packed = torch.empty(
            (E, 2 * inter, D // 2), dtype=torch.int8, device=device
        )
        s13_raw = torch.empty(
            (E, 2 * inter, D // FP4_BLOCK), dtype=torch.float, device=device
        )
        for i in range(E):
            w13_packed[i], s13_raw[i] = per_token_cast_to_fp4(
                w13_bf16[i], use_ue8m0=True, gran_k=FP4_BLOCK
            )
        del w13_bf16

        s13_int = prepare_fp4_weight_scale_for_deepgemm(s13_raw, 2 * inter, D, E)
        del s13_raw
        torch.cuda.empty_cache()

        # --- L2 (down): dequant FP8 → BF16 → FP4 ---
        w2_bf16 = _dequant_fp8(w2_fp8, w2_scale)  # [E, D, inter]
        del w2_fp8, w2_scale

        w2_packed = torch.empty((E, D, inter // 2), dtype=torch.int8, device=device)
        s2_raw = torch.empty(
            (E, D, inter // FP4_BLOCK), dtype=torch.float, device=device
        )
        for i in range(E):
            w2_packed[i], s2_raw[i] = per_token_cast_to_fp4(
                w2_bf16[i], use_ue8m0=True, gran_k=FP4_BLOCK
            )
        del w2_bf16

        s2_int = prepare_fp4_weight_scale_for_deepgemm(s2_raw, D, inter, E)
        del s2_raw
        torch.cuda.empty_cache()

        # Apply mega MoE transform
        (l1_w, l1_sf), (l2_w, l2_sf) = deep_gemm.transform_weights_for_mega_moe(
            (w13_packed, s13_int),
            (w2_packed, s2_int),
        )
        del w13_packed, s13_int, w2_packed, s2_int
        torch.cuda.empty_cache()

        self._mega_l1_w = l1_w
        self._mega_l1_sf = l1_sf
        self._mega_l2_w = l2_w
        self._mega_l2_sf = l2_sf

        logger.info("[GLM5 MegaMoE] FP8→FP4 conversion done: layer=%d", cfg.layer_id)
        self._setup_buffer_and_warmup()

    def setup_weights_from_bf16(
        self,
        w1_bf16: torch.Tensor,  # [E_local, inter, D] gate
        w2_bf16: torch.Tensor,  # [E_local, D, inter] down
        w3_bf16: torch.Tensor,  # [E_local, inter, D] up
    ) -> None:
        """Setup weights from BF16 format (for testing or non-quantized models).

        Converts BF16 → FP4 for the mega kernel.
        """
        import deep_gemm
        from deep_gemm.utils import per_token_cast_to_fp4

        cfg = self.cfg
        E = cfg.n_local_experts
        D = cfg.dim
        inter = cfg.moe_inter_dim
        device = w1_bf16.device

        # Combine gate + up
        w13_bf16 = torch.cat([w1_bf16, w3_bf16], dim=1)  # [E, 2*inter, D]
        del w1_bf16, w3_bf16

        w13_packed = torch.empty(
            (E, 2 * inter, D // 2), dtype=torch.int8, device=device
        )
        s13_raw = torch.empty(
            (E, 2 * inter, D // FP4_BLOCK), dtype=torch.float, device=device
        )
        for i in range(E):
            w13_packed[i], s13_raw[i] = per_token_cast_to_fp4(
                w13_bf16[i], use_ue8m0=True, gran_k=FP4_BLOCK
            )
        del w13_bf16

        s13_int = prepare_fp4_weight_scale_for_deepgemm(s13_raw, 2 * inter, D, E)
        del s13_raw
        torch.cuda.empty_cache()

        w2_packed = torch.empty((E, D, inter // 2), dtype=torch.int8, device=device)
        s2_raw = torch.empty(
            (E, D, inter // FP4_BLOCK), dtype=torch.float, device=device
        )
        for i in range(E):
            w2_packed[i], s2_raw[i] = per_token_cast_to_fp4(
                w2_bf16[i], use_ue8m0=True, gran_k=FP4_BLOCK
            )
        del w2_bf16

        s2_int = prepare_fp4_weight_scale_for_deepgemm(s2_raw, D, inter, E)
        del s2_raw
        torch.cuda.empty_cache()

        (l1_w, l1_sf), (l2_w, l2_sf) = deep_gemm.transform_weights_for_mega_moe(
            (w13_packed, s13_int),
            (w2_packed, s2_int),
        )
        del w13_packed, s13_int, w2_packed, s2_int
        torch.cuda.empty_cache()

        self._mega_l1_w = l1_w
        self._mega_l1_sf = l1_sf
        self._mega_l2_w = l2_w
        self._mega_l2_sf = l2_sf
        self._setup_buffer_and_warmup()

    def _setup_buffer_and_warmup(self) -> None:
        """Allocate symmetric memory buffer and run JIT warmup."""
        import torch.distributed as dist

        cfg = self.cfg
        device = self._mega_l1_w.device

        assert (
            dist.is_initialized()
        ), "GLM5 MegaMoE requires torch.distributed initialised"
        group = dist.group.WORLD
        self._mega_group = group

        self._mega_buf = get_or_create_mega_buf(
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
        self._maybe_warmup_jit_once()

    def _resolve_jit_warmup_token_counts(self, num_sms: int) -> list[int]:
        cfg = self.cfg
        max_tokens_per_rank = int(cfg.max_tokens_per_rank)
        override = parse_jit_warmup_tokens_override()
        if override is not None:
            return clamp_token_counts(override, max_tokens_per_rank)
        return generate_mega_moe_jit_token_counts(
            num_ranks=cfg.ep_size,
            num_experts=cfg.n_routed_experts,
            num_experts_per_rank=cfg.n_local_experts,
            num_topk=cfg.n_activated_experts,
            intermediate_hidden=cfg.moe_inter_dim,
            num_sms=num_sms,
            max_tokens_per_rank=max_tokens_per_rank,
        )

    def _maybe_warmup_jit_once(self) -> None:
        if not mega_moe_jit_warmup_enabled():
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MegaMoE JIT warmup must not run inside CUDA graph capture"
            )

        import deep_gemm
        import torch.distributed as dist

        cfg = self.cfg
        num_sms = int(deep_gemm.get_num_sms())
        token_counts = self._resolve_jit_warmup_token_counts(num_sms)
        if not token_counts:
            return

        warmup_key = (
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
        if warmup_key in _MEGA_MOE_JIT_WARMED_KEYS:
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            logger.info(
                "[GLM5 MegaMoE] JIT warmup start: layer=%d tokens=[%s] "
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
        self._warmup_jit(token_counts)
        _MEGA_MOE_JIT_WARMED_KEYS.add(warmup_key)
        if rank == 0:
            logger.info(
                "[GLM5 MegaMoE] JIT warmup done: layer=%d tokens=[%s]",
                cfg.layer_id,
                format_token_counts(token_counts),
            )

    @torch.inference_mode()
    def _warmup_jit(self, token_counts: list[int]) -> None:
        """Compile MegaMoE JIT buckets with synthetic tokens."""
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
            self.forward(
                x[:token_count],
                weights[:token_count],
                indices[:token_count],
            )
            torch.cuda.synchronize(device)
        dist.barrier()

    def forward(
        self,
        x: torch.Tensor,  # [T, D] BF16 local-rank tokens
        weights: torch.Tensor,  # [T, topk] FP32 router weights
        indices: torch.Tensor,  # [T, topk] int64 GLOBAL expert IDs
    ) -> torch.Tensor:
        """Run the fused DeepGEMM Mega MoE kernel.

        Fuses dispatch + L1 GEMM + SwiGLU + L2 GEMM + combine via symmetric
        memory NVLink communication.

        Returns [T, D] BF16 combined routed-expert output.
        """
        import deep_gemm

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

        # Pack inputs into symmetric memory buffer
        self._input_packer.pack(x, weights, indices, buf, T)
        self._maybe_pre_kernel_barrier(T)
        _sync_cuda_graph_warmup_ranks(
            f"glm5.mega_moe.layer{self.cfg.layer_id}.before_deepgemm",
            x.device,
        )

        y = self._mega_y[:T]
        deep_gemm.fp8_fp4_mega_moe(
            y,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            buf,
            recipe=(1, 1, FP4_BLOCK),
            activation="swiglu",
            activation_clamp=(
                self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None
            ),
            fast_math=True,
        )
        return y

    def _maybe_pre_kernel_barrier(self, tokens: int) -> None:
        """Optional host-side rendezvous before the DeepGEMM MegaMoE kernel."""
        if not _pre_kernel_barrier_enabled():
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{_PRE_KERNEL_BARRIER_ENV}=1 is incompatible with CUDA graph "
                "capture"
            )

        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError(
                f"{_PRE_KERNEL_BARRIER_ENV}=1 requires torch.distributed "
                "to be initialized"
            )

        cfg = self.cfg
        group = getattr(self, "_mega_group", dist.group.WORLD)
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        device = self._mega_l1_w.device
        _log_pre_kernel_barrier("enter", cfg.layer_id, rank, world_size, tokens, device)

        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.current_stream().synchronize()
                try:
                    dist.barrier(
                        group=group,
                        device_ids=[torch.cuda.current_device()],
                    )
                except TypeError:
                    dist.barrier(group=group)
        else:
            dist.barrier(group=group)

        _log_pre_kernel_barrier("leave", cfg.layer_id, rank, world_size, tokens, device)
