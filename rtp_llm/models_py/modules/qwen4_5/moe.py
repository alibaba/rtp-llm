"""Layer 0 Dense MLP and Layers 1..39 Latent MoE + Shared Expert for Qwen 4.5-Flash.

Exact Qwen 4.5-Flash (Qwen4_5ForCausalLM, 1109-bf16-re-mxfp4-ple-fp8) specification:
  - Layer 0: ``Qwen4_5DenseMLP`` (H=5120, I=12800, BF16, SiTU-GLU(4.0, 25.0)).
  - Layers 1..39: ``Qwen4_5LatentMoE``:
      1. ``routed = F.linear(x, latent_down_proj)``                 (5120 -> 2560)
      2. ``routed = gemma_rmsnorm(routed, pre_moe_latent_ln, eps)`` ((1.0 + weight) in FP32)
      3. ``routed = routed * sigmoid(F.linear(x, latent_gate_proj))`` (5120 -> 2560)
      4. Router on ``routed`` (``router_precision="fp32_logits"``):
           ``logits = F.linear(routed, gate_weight).float()``       ([M, 640] FP32)
           ``scores = sigmoid(logits)``
           ``selection = scores + expert_bias.float()``             (FP32 [640])
           ``topk_ids = selection.topk(top_k, dim=-1).indices``
           ``topk_weights = scores.gather(-1, topk_ids)``
           ``topk_weights = topk_weights / (topk_weights.sum(-1, keepdim=True) + 1e-8)``
           ``topk_weights = topk_weights.to(x.dtype).float()``
      5. Routed MXFP4 experts on ``routed`` (2560 -> 2560 -> 2560, E=640, top_k=10)
      6. ``routed = F.linear(routed, latent_up_proj)``              (2560 -> 5120)
      7. ``routed = routed * sigmoid(F.linear(x, latent_scale_gate))`` ([M, 1])
      8. Shared expert on original ``x``:
           ``shared = shared_expert(x) * sigmoid(shared_expert_gate(x))``
      9. Return ``routed + shared``.

NOTE: Post-FFN short-conv (``ffn_output_sconv``) and output clamping
(``clamp(-clip_mlp_output, clip_mlp_output)``, i.e. ``clamp(-1000.0, 1000.0)``)
are applied immediately after ``Qwen4_5DenseMLP`` / ``Qwen4_5LatentMoE`` inside
``DecoderLayer`` and are intentionally not baked into these modules.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from rtp_llm.models_py.modules.qwen4_5.activation import SituAndMul
from rtp_llm.models_py.modules.qwen4_5.contracts import Qwen4_5ArchConfig
from rtp_llm.models_py.modules.qwen4_5.mxfp4_ops import (
    MXFP4_BLOCK_SIZE,
    MXFP4MoEWeights,
    mxfp4_moe_forward,
    mxfp4_moe_forward_ref,
    prepare_mxfp4_moe_weights,
    supports_triton_mxfp4,
)


# ---------------------------------------------------------------------------
# Gemma-style RMSNorm: x_fp32 * rsqrt(mean(x_fp32^2) + eps) * (1.0 + weight)
# ---------------------------------------------------------------------------


def gemma_rmsnorm_ref(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Pure PyTorch FP32 reference for Gemma-style RMSNorm ``(1.0 + weight)``."""
    x_f32 = x.float()
    inv_rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = x_f32 * inv_rms * (1.0 + weight.float())
    return out.to(x.dtype)


@triton.jit
def _gemma_rmsnorm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    D,
    stride_x_row: tl.int64,
    stride_out_row: tl.int64,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x = tl.load(x_ptr + pid_row * stride_x_row + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.div_rn(tl.sum(x * x, axis=0), D.to(tl.float32))
    inv_rms = libdevice.rsqrt(mean_sq + eps)
    y = x * inv_rms * (1.0 + w)

    tl.store(
        out_ptr + pid_row * stride_out_row + offs,
        y.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    use_triton: bool = True,
) -> torch.Tensor:
    """Gemma-style RMSNorm computing ``(1.0 + weight)`` in FP32."""
    D = x.shape[-1]
    if not use_triton or not x.is_cuda or D > 16384:
        return gemma_rmsnorm_ref(x, weight, eps=eps)
    if x.numel() == 0:
        return torch.empty_like(x)

    orig_shape = x.shape
    x_2d = x.reshape(-1, D)
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()
    w = weight.contiguous()
    num_rows = x_2d.shape[0]
    out = torch.empty((num_rows, D), dtype=x.dtype, device=x.device)

    BLOCK_D = triton.next_power_of_2(D)
    num_warps = 8 if BLOCK_D >= 2048 else 4
    _gemma_rmsnorm_kernel[(num_rows,)](
        x_2d,
        w,
        out,
        D,
        x_2d.stride(0),
        out.stride(0),
        eps=float(eps),
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )
    return out.reshape(orig_shape)


class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm with zero-initialized ``weight`` (scale = ``1.0 + weight``)."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        use_triton: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.use_triton = bool(use_triton)
        self.weight = nn.Parameter(
            torch.zeros(hidden_size, dtype=dtype, device=device)
        )

    def forward(
        self, x: torch.Tensor, *, use_triton: Optional[bool] = None
    ) -> torch.Tensor:
        flag = self.use_triton if use_triton is None else use_triton
        return gemma_rmsnorm(x, self.weight, eps=self.eps, use_triton=flag)



@triton.jit
def _qwen45_sigmoid_topk_router_kernel(
    logits_ptr,
    bias_ptr,
    ids_ptr,
    weights_ptr,
    E,
    stride_lm: tl.int64,
    E_PAD: tl.constexpr,
    TOP_K: tl.constexpr,
    TOP_K_PAD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NORMALIZE: tl.constexpr,
    ROUND_BF16: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs_e = tl.arange(0, E_PAD)
    mask_e = offs_e < E

    raw = tl.load(
        logits_ptr + row * stride_lm + offs_e, mask=mask_e, other=0.0
    ).to(tl.float32)
    scores = tl.where(mask_e, tl.div_rn(1.0, 1.0 + libdevice.exp(-raw)), 0.0)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_e, mask=mask_e, other=0.0).to(tl.float32)
        sel = tl.where(mask_e, scores + bias, -3.4e38)
    else:
        sel = tl.where(mask_e, scores, -3.4e38)

    slots = tl.arange(0, TOP_K_PAD)
    chosen_w = tl.zeros((TOP_K_PAD,), dtype=tl.float32)
    ids_base = ids_ptr + row * TOP_K
    for i in tl.static_range(TOP_K):
        best_val = tl.max(sel, axis=0)
        best_idx = tl.min(tl.where(sel == best_val, offs_e, 2147483647), axis=0)
        best_score = tl.sum(tl.where(offs_e == best_idx, scores, 0.0), axis=0)
        chosen_w = tl.where(slots == i, best_score, chosen_w)
        tl.store(ids_base + i, best_idx)
        sel = tl.where(offs_e == best_idx, -3.4e38, sel)

    if NORMALIZE:
        denom = tl.sum(chosen_w, axis=0) + 1e-8
        chosen_w = tl.div_rn(chosen_w, denom)
    if ROUND_BF16:
        chosen_w = chosen_w.to(tl.bfloat16).to(tl.float32)

    tl.store(weights_ptr + row * TOP_K + slots, chosen_w, mask=slots < TOP_K)


# ---------------------------------------------------------------------------
# Layer 0 Dense MLP & Shared Expert (BF16 + SiTU-GLU)
# ---------------------------------------------------------------------------


class Qwen4_5DenseMLP(nn.Module):
    """Dense gated MLP with SiTU-GLU activation.

    Used for:
      - Layer 0 Dense MLP (``hidden_size=5120, intermediate_size=12800``)
      - Layers 1..39 Shared Expert (``hidden_size=5120, intermediate_size=1280``)
    """

    def __init__(
        self,
        hidden_size: int = 5120,
        intermediate_size: int = 12800,
        *,
        gate_beta: float = 4.0,
        up_beta: float = 25.0,
        use_triton: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.gate_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False, dtype=dtype, device=device
        )
        self.up_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False, dtype=dtype, device=device
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size, bias=False, dtype=dtype, device=device
        )
        self.act_fn = SituAndMul(
            gate_beta=gate_beta, up_beta=up_beta, use_triton=use_triton
        )
        self.register_buffer("_gate_up_weight", None, persistent=False)

    @classmethod
    def from_config(
        cls,
        config: Qwen4_5ArchConfig,
        *,
        is_shared_expert: bool = False,
        use_triton: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> "Qwen4_5DenseMLP":
        inter = (
            config.shared_expert_intermediate_size
            if is_shared_expert
            else config.intermediate_size
        )
        return cls(
            hidden_size=config.hidden_size,
            intermediate_size=inter,
            gate_beta=config.hidden_act_gate_beta,
            up_beta=config.hidden_act_up_beta,
            use_triton=use_triton,
            dtype=dtype,
            device=device,
        )

    @torch.no_grad()
    def prepare_weights(self, *, fuse_gate_up: bool = True) -> None:
        """Optionally cache concatenated ``[gate_proj; up_proj]`` for single-GEMM forward."""
        if fuse_gate_up:
            self._gate_up_weight = torch.cat(
                [self.gate_proj.weight.data, self.up_proj.weight.data], dim=0
            ).contiguous()
        else:
            self._gate_up_weight = None

    def forward(
        self, hidden_states: torch.Tensor, *, use_triton: Optional[bool] = None
    ) -> torch.Tensor:
        if self._gate_up_weight is not None and (use_triton is None or use_triton):
            gate_up = F.linear(hidden_states, self._gate_up_weight)
            act = self.act_fn(gate_up)
            return self.down_proj(act)

        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        if use_triton is None:
            act = self.act_fn(gate, up)
        else:
            orig_flag = self.act_fn.use_triton
            try:
                self.act_fn.use_triton = use_triton
                act = self.act_fn(gate, up)
            finally:
                self.act_fn.use_triton = orig_flag
        return self.down_proj(act)


# ---------------------------------------------------------------------------
# Layers 1..39 Latent MoE + Shared Expert (MXFP4 Routed Experts)
# ---------------------------------------------------------------------------


class Qwen4_5LatentMoE(nn.Module):
    """Qwen 4.5-Flash Latent MoE + Shared Expert with B300 SM103 MXFP4 Routed Experts.

    NOTE: ``ffn_output_sconv`` and ``clamp(-1000.0, 1000.0)`` are applied right after
    this module in ``DecoderLayer``.
    """

    def __init__(
        self,
        config: Optional[Qwen4_5ArchConfig] = None,
        *,
        hidden_size: int = 5120,
        routed_expert_hidden_size: int = 2560,
        moe_intermediate_size: int = 2560,
        shared_expert_intermediate_size: int = 1280,
        num_experts: int = 640,
        num_experts_per_tok: int = 10,
        rms_norm_eps: float = 1e-6,
        gate_beta: float = 4.0,
        up_beta: float = 25.0,
        shared_expert_gate: bool = True,
        router_precision: str = "fp32_logits",
        router_normalize_topk_scores: bool = True,
        moe_router_enable_expert_bias: bool = True,
        use_triton_mxfp4: bool = True,
        compute_mode: str = "bf16",
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        init_mxfp4_weights: bool = True,
        ep_size: int = 1,
        ep_rank: int = 0,
        ep_group: Any = None,
    ) -> None:
        super().__init__()
        if config is not None:
            hidden_size = config.hidden_size
            routed_expert_hidden_size = config.routed_expert_hidden_size
            moe_intermediate_size = config.moe_intermediate_size
            shared_expert_intermediate_size = config.shared_expert_intermediate_size
            num_experts = config.num_experts
            num_experts_per_tok = config.num_experts_per_tok
            rms_norm_eps = config.rms_norm_eps
            gate_beta = config.hidden_act_gate_beta
            up_beta = config.hidden_act_up_beta
            shared_expert_gate = config.shared_expert_gate
            router_precision = config.router_precision
            router_normalize_topk_scores = config.router_normalize_topk_scores
            moe_router_enable_expert_bias = config.moe_router_enable_expert_bias

        if routed_expert_hidden_size % MXFP4_BLOCK_SIZE != 0:
            raise ValueError(
                f"routed_expert_hidden_size={routed_expert_hidden_size} must be divisible by {MXFP4_BLOCK_SIZE}"
            )
        if moe_intermediate_size % MXFP4_BLOCK_SIZE != 0:
            raise ValueError(
                f"moe_intermediate_size={moe_intermediate_size} must be divisible by {MXFP4_BLOCK_SIZE}"
            )

        self.hidden_size = int(hidden_size)
        self.latent_dim = int(routed_expert_hidden_size)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.shared_expert_intermediate_size = int(shared_expert_intermediate_size)
        self.num_experts = int(num_experts)
        self.ep_size = int(ep_size)
        self.ep_rank = int(ep_rank)
        self.ep_group = ep_group
        if self.ep_size < 1 or self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts={self.num_experts} must be divisible by ep_size={self.ep_size}"
            )
        if not (0 <= self.ep_rank < self.ep_size):
            raise ValueError(f"ep_rank={self.ep_rank} out of range for ep_size={self.ep_size}")
        self.local_num_experts = self.num_experts // self.ep_size
        self.ep_start = self.ep_rank * self.local_num_experts
        self.ep_end = self.ep_start + self.local_num_experts
        self.top_k = int(num_experts_per_tok)
        self.rms_norm_eps = float(rms_norm_eps)
        self.gate_beta = float(gate_beta)
        self.up_beta = float(up_beta)
        self.router_precision = str(router_precision)
        self.normalize_topk_scores = bool(router_normalize_topk_scores)
        self.use_triton_mxfp4 = bool(use_triton_mxfp4)
        self.compute_mode = str(compute_mode)
        self.weights_transposed = False

        # Latent projections & pre-MoE Gemma RMSNorm (BF16)
        self.latent_down_proj = nn.Linear(
            self.hidden_size, self.latent_dim, bias=False, dtype=dtype, device=device
        )
        self.pre_moe_latent_layernorm = GemmaRMSNorm(
            self.latent_dim,
            eps=self.rms_norm_eps,
            use_triton=self.use_triton_mxfp4,
            dtype=dtype,
            device=device,
        )
        self.latent_gate_proj = nn.Linear(
            self.hidden_size, self.latent_dim, bias=False, dtype=dtype, device=device
        )
        self.latent_up_proj = nn.Linear(
            self.latent_dim, self.hidden_size, bias=False, dtype=dtype, device=device
        )
        self.latent_scale_gate = nn.Linear(
            self.hidden_size, 1, bias=False, dtype=dtype, device=device
        )

        # Router (BF16 gate weight + FP32 expert bias)
        self.gate = nn.Linear(
            self.latent_dim, self.num_experts, bias=False, dtype=dtype, device=device
        )
        if moe_router_enable_expert_bias:
            self.expert_bias: Optional[nn.Parameter] = nn.Parameter(
                torch.zeros(self.num_experts, dtype=torch.float32, device=device)
            )
        else:
            self.register_parameter("expert_bias", None)

        # Shared expert (BF16 Dense MLP + sigmoid scalar gate)
        self.shared_expert = Qwen4_5DenseMLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.shared_expert_intermediate_size,
            gate_beta=self.gate_beta,
            up_beta=self.up_beta,
            use_triton=self.use_triton_mxfp4,
            dtype=dtype,
            device=device,
        )
        if shared_expert_gate:
            self.shared_expert_gate: Optional[nn.Linear] = nn.Linear(
                self.hidden_size, 1, bias=False, dtype=dtype, device=device
            )
        else:
            self.shared_expert_gate = None

        # Routed experts MXFP4 packed buffers (uint8; local shard [E_local, N, K//2])
        if init_mxfp4_weights:
            self.register_buffer(
                "w13_u8",
                torch.zeros(
                    (
                        self.local_num_experts,
                        2 * self.moe_intermediate_size,
                        self.latent_dim // 2,
                    ),
                    dtype=torch.uint8,
                    device=device,
                ),
            )
            self.register_buffer(
                "w13_scale",
                torch.full(
                    (
                        self.local_num_experts,
                        2 * self.moe_intermediate_size,
                        self.latent_dim // MXFP4_BLOCK_SIZE,
                    ),
                    127,
                    dtype=torch.uint8,
                    device=device,
                ),
            )
            self.register_buffer(
                "w2_u8",
                torch.zeros(
                    (
                        self.local_num_experts,
                        self.latent_dim,
                        self.moe_intermediate_size // 2,
                    ),
                    dtype=torch.uint8,
                    device=device,
                ),
            )
            self.register_buffer(
                "w2_scale",
                torch.full(
                    (
                        self.local_num_experts,
                        self.latent_dim,
                        self.moe_intermediate_size // MXFP4_BLOCK_SIZE,
                    ),
                    127,
                    dtype=torch.uint8,
                    device=device,
                ),
            )
        else:
            self.register_buffer("w13_u8", None)
            self.register_buffer("w13_scale", None)
            self.register_buffer("w2_u8", None)
            self.register_buffer("w2_scale", None)

    @classmethod
    def from_config(
        cls,
        config: Qwen4_5ArchConfig,
        *,
        use_triton_mxfp4: bool = True,
        compute_mode: str = "bf16",
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        init_mxfp4_weights: bool = True,
        ep_size: int = 1,
        ep_rank: int = 0,
        ep_group: Any = None,
    ) -> "Qwen4_5LatentMoE":
        return cls(
            config=config,
            use_triton_mxfp4=use_triton_mxfp4,
            compute_mode=compute_mode,
            dtype=dtype,
            device=device,
            init_mxfp4_weights=init_mxfp4_weights,
            ep_size=ep_size,
            ep_rank=ep_rank,
            ep_group=ep_group,
        )

    @torch.no_grad()
    def load_mxfp4_weights(
        self,
        *,
        w13_u8: Optional[torch.Tensor] = None,
        w13_scale: Optional[torch.Tensor] = None,
        w_gate_u8: Optional[torch.Tensor] = None,
        w_gate_scale: Optional[torch.Tensor] = None,
        w_up_u8: Optional[torch.Tensor] = None,
        w_up_scale: Optional[torch.Tensor] = None,
        w2_u8: torch.Tensor,
        w2_scale: torch.Tensor,
        transpose_weights: bool = False,
    ) -> None:
        """Load MXFP4 routed expert weights from fused or separate checkpoint tensors."""
        prepared = prepare_mxfp4_moe_weights(
            w13_u8=w13_u8,
            w13_scale=w13_scale,
            w_gate_u8=w_gate_u8,
            w_gate_scale=w_gate_scale,
            w_up_u8=w_up_u8,
            w_up_scale=w_up_scale,
            w2_u8=w2_u8,
            w2_scale=w2_scale,
            transpose_weights=transpose_weights,
        )
        self.w13_u8 = prepared.w13_u8
        self.w13_scale = prepared.w13_scale
        self.w2_u8 = prepared.w2_u8
        self.w2_scale = prepared.w2_scale
        self.weights_transposed = prepared.weights_transposed

    @torch.no_grad()
    def prepare_weights(
        self,
        *,
        transpose_mxfp4: bool = False,
        fuse_shared_gate_up: bool = True,
    ) -> None:
        """Prepare MXFP4 routed weights and shared expert fused gate_up weights."""
        if self.w13_u8 is not None and self.w2_u8 is not None:
            if transpose_mxfp4 and not self.weights_transposed:
                self.w13_u8 = self.w13_u8.transpose(1, 2).contiguous()
                self.w2_u8 = self.w2_u8.transpose(1, 2).contiguous()
                self.weights_transposed = True
            elif not transpose_mxfp4 and self.weights_transposed:
                self.w13_u8 = self.w13_u8.transpose(1, 2).contiguous()
                self.w2_u8 = self.w2_u8.transpose(1, 2).contiguous()
                self.weights_transposed = False
            else:
                self.w13_u8 = self.w13_u8.contiguous()
                self.w13_scale = self.w13_scale.contiguous()
                self.w2_u8 = self.w2_u8.contiguous()
                self.w2_scale = self.w2_scale.contiguous()
        self.shared_expert.prepare_weights(fuse_gate_up=fuse_shared_gate_up)

    def _mxfp4_weights(self) -> MXFP4MoEWeights:
        if (
            self.w13_u8 is None
            or self.w13_scale is None
            or self.w2_u8 is None
            or self.w2_scale is None
        ):
            raise RuntimeError(
                "MXFP4 routed expert weights are not initialized; call load_mxfp4_weights() first."
            )
        return MXFP4MoEWeights(
            w13_u8=self.w13_u8,
            w13_scale=self.w13_scale,
            w2_u8=self.w2_u8,
            w2_scale=self.w2_scale,
            weights_transposed=self.weights_transposed,
        )

    def route(
        self, routed: torch.Tensor, *, use_triton: Optional[bool] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute top-k expert IDs and normalized weights from latent activations.

        Exact specification (``router_precision="fp32_logits"``):
          1. ``logits = F.linear(routed, gate_weight).float()``
          2. ``scores = torch.sigmoid(logits)``
          3. ``selection = scores + expert_bias.float()``
          4. ``topk_ids = selection.topk(top_k, dim=-1).indices``
          5. ``topk_weights = scores.gather(-1, topk_ids)``
          6. ``topk_weights = topk_weights / (topk_weights.sum(-1, keepdim=True) + 1e-8)``
          7. ``topk_weights = topk_weights.to(routed.dtype).float()``
        """
        if self.router_precision == "fp32":
            logits = F.linear(routed.float(), self.gate.weight.float())
        else:
            # Default "fp32_logits": linear in input dtype, cast logits to FP32.
            logits = F.linear(routed, self.gate.weight)

        run_triton = (
            self.use_triton_mxfp4 if use_triton is None else bool(use_triton)
        )
        if (
            run_triton
            and logits.is_cuda
            and self.num_experts <= 2048
            and self.top_k <= 16
            and routed.dtype in (torch.bfloat16, torch.float32)
        ):
            if logits.stride(-1) != 1:
                logits = logits.contiguous()
            M = logits.shape[0]
            topk_ids = torch.empty((M, self.top_k), dtype=torch.int32, device=logits.device)
            topk_weights = torch.empty((M, self.top_k), dtype=torch.float32, device=logits.device)
            if M == 0:
                return topk_ids, topk_weights
            bias_t = self.expert_bias if self.expert_bias is not None else logits
            E_PAD = triton.next_power_of_2(self.num_experts)
            TOP_K_PAD = triton.next_power_of_2(max(self.top_k, 2))
            _qwen45_sigmoid_topk_router_kernel[(M,)](
                logits,
                bias_t,
                topk_ids,
                topk_weights,
                self.num_experts,
                logits.stride(0),
                E_PAD=E_PAD,
                TOP_K=self.top_k,
                TOP_K_PAD=TOP_K_PAD,
                HAS_BIAS=(self.expert_bias is not None),
                NORMALIZE=self.normalize_topk_scores,
                ROUND_BF16=(routed.dtype == torch.bfloat16),
                num_warps=4,
            )
            return topk_ids, topk_weights

        logits_f = logits.float()
        scores = torch.sigmoid(logits_f)
        if self.expert_bias is not None:
            selection = scores + self.expert_bias.float()
        else:
            selection = scores
        topk_ids = torch.topk(selection, k=self.top_k, dim=-1).indices
        topk_weights = scores.gather(-1, topk_ids)
        if self.normalize_topk_scores:
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-8
            )
        topk_weights = topk_weights.to(routed.dtype).float()
        return topk_ids, topk_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        use_triton_mxfp4: Optional[bool] = None,
        compute_mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Run Latent MoE + Shared Expert forward pass.

        Accepts 2D ``[M, hidden_size]`` or 3D ``[B, T, hidden_size]`` tensors.
        """
        orig_shape = hidden_states.shape
        x = hidden_states.reshape(-1, self.hidden_size)
        if x.shape[0] == 0:
            return torch.empty_like(hidden_states)

        use_triton = (
            self.use_triton_mxfp4 if use_triton_mxfp4 is None else bool(use_triton_mxfp4)
        )
        mode = self.compute_mode if compute_mode is None else compute_mode
        run_triton = use_triton and x.is_cuda and supports_triton_mxfp4(x.device)

        # 1. Latent down projection (5120 -> 2560)
        routed = F.linear(x, self.latent_down_proj.weight)

        # 2. Pre-MoE Gemma RMSNorm ((1.0 + weight) in FP32)
        routed = self.pre_moe_latent_layernorm(routed, use_triton=run_triton)

        # 3. Latent input gate (5120 -> 2560)
        routed = routed * torch.sigmoid(F.linear(x, self.latent_gate_proj.weight))

        # 4. Sigmoid + expert-bias top-k router
        topk_ids, topk_weights = self.route(routed, use_triton=run_triton)

        # 5. Routed MXFP4 experts (2560 -> 2560; EP AllReduce happens in 2560-dim latent space)
        weights = self._mxfp4_weights()
        if run_triton:
            routed = mxfp4_moe_forward(
                routed,
                topk_ids,
                topk_weights,
                weights,
                gate_beta=self.gate_beta,
                up_beta=self.up_beta,
                compute_mode=mode,
                use_triton=True,
                ep_start=self.ep_start,
                ep_size=self.ep_size,
            )
        else:
            routed = mxfp4_moe_forward_ref(
                routed,
                topk_ids,
                topk_weights,
                weights,
                gate_beta=self.gate_beta,
                up_beta=self.up_beta,
                ep_start=self.ep_start,
                ep_size=self.ep_size,
            )
        if self.ep_size > 1 and self.ep_group is not None:
            torch.distributed.all_reduce(
                routed, op=torch.distributed.ReduceOp.SUM, group=self.ep_group
            )

        # 6. Latent up projection (2560 -> 5120)
        routed = F.linear(routed, self.latent_up_proj.weight)

        # 7. Latent output scale gate ([M, 1])
        scale = torch.sigmoid(F.linear(x, self.latent_scale_gate.weight))
        routed = routed * scale

        # 8. Shared expert on original hidden_states ([M, 5120])
        shared = self.shared_expert(x, use_triton=run_triton)
        if self.shared_expert_gate is not None:
            shared = shared * torch.sigmoid(F.linear(x, self.shared_expert_gate.weight))

        # 9. Combine routed + shared (ffn_output_sconv + clamp(-1000, 1000) in DecoderLayer)
        return (routed + shared).reshape(orig_shape)
