"""Shared DSV4 utility functions used across BF16 and FP8 paths."""

import weakref

import torch
from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.model_loader.weight_memory_saver import (
    current_model_scope,
    feature_weights_region,
)
from rtp_llm.models_py.modules.factory.linear import LinearFactory

_V4_FP8_BLOCK_CFG = Fp8BlockWiseQuantConfig()
_FP8_LINEAR_REGISTRY = weakref.WeakSet()


class V41MXFP8Linear(torch.nn.Module):
    """Native 32x32 checkpoint blocks, expanded to per-row MXFP8 scales."""

    def __init__(self, weight: torch.Tensor, scales: torch.Tensor):
        super().__init__()
        self.weight = weight
        self.N, self.K = weight.shape
        if scales.shape != ((self.N + 31) // 32, (self.K + 31) // 32):
            raise ValueError(f"Invalid V4.1 FP8 scales: {weight.shape}, {scales.shape}")
        rows = torch.arange(self.N, device=scales.device) // 32
        self.weight_scales = get_mn_major_tma_aligned_packed_ue8m0_tensor(
            scales.float().index_select(0, rows)
        )

    def forward(self, x: torch.Tensor, out=None):
        import deep_gemm

        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        shape = (*x.shape[:-1], self.N)
        if x.numel() == 0:
            return out if out is not None else x.new_empty(shape)
        flat = x.reshape(-1, self.K).contiguous()
        quantized, scales = sgl_per_token_group_quant_fp8(
            flat,
            group_size=32,
            eps=torch.finfo(torch.float32).tiny,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        output = (
            out
            if out is not None
            else torch.empty(shape, device=x.device, dtype=torch.bfloat16)
        )
        deep_gemm.fp8_fp4_gemm_nt(
            (quantized, scales),
            (self.weight, self.weight_scales),
            output.reshape(-1, self.N),
            recipe=(1, 1, 32),
        )
        return output


def _is_v41_fp8_scale(w: torch.Tensor, s: torch.Tensor) -> bool:
    return s.dtype == torch.float8_e8m0fnu and s.shape == (
        (w.shape[0] + 31) // 32,
        (w.shape[1] + 31) // 32,
    )


def iter_fp8_linears() -> list:
    """Resident V4 projections, including model-level MTP e/h fusion."""
    return list(_FP8_LINEAR_REGISTRY)


def _repack_v4_fp8_scale_to_int32(scale: torch.Tensor) -> torch.Tensor:
    """V4 ckpt UE8M0 ``[N/128, K/128]`` to DeepGEMM int32-packed scale."""
    n_blk, _ = scale.shape
    n = n_blk * 128
    idx = torch.arange(n, device=scale.device) // 128
    scale_rep = scale.float().index_select(-2, idx)
    return get_mn_major_tma_aligned_packed_ue8m0_tensor(scale_rep)


def _v4_fp8_linear(w: torch.Tensor, s: torch.Tensor):
    """Build a CudaFp8DeepGEMMLinear from raw V4 FP8 weight + scale tensors."""
    if _is_v41_fp8_scale(w, s):
        return V41MXFP8Linear(w, s)
    raw_scale = s
    with feature_weights_region():
        if s.dtype == torch.float8_e8m0fnu:
            s = _repack_v4_fp8_scale_to_int32(s)
        linear = LinearFactory.create_linear_from_weights(
            {"_w": w, "_s": s}, "_w", "_s", quant_config=_V4_FP8_BLOCK_CFG
        )
    # The packed scale is not a checkpoint tensor. Retain its source and
    # owner so level-2 wake can reconstruct it at the CUDA-graph-baked address.
    linear._sleep_raw_weight_source = w
    linear._sleep_raw_scale_source = raw_scale
    linear._sleep_row_slice = None
    linear._sleep_col_slice = None
    linear._sleep_model_scope = current_model_scope()
    _FP8_LINEAR_REGISTRY.add(linear)
    return linear


def _v4_fp8_linear_from_dict(weights: dict, weight_key: str, scale_key: str):
    """Backwards-compat bridge over ``_v4_fp8_linear`` for flat dict callers."""
    w = weights[weight_key]
    s = weights[scale_key]
    if _is_v41_fp8_scale(w, s):
        return _v4_fp8_linear(w, s)
    if s.dtype == torch.float8_e8m0fnu:
        s = _repack_v4_fp8_scale_to_int32(s)
        weights[scale_key] = s
    return _v4_fp8_linear(w, s)


def _sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Reference PyTorch sparse attention with attention sink.

    Output: [B, S, H, D]
    """
    bsz, seqlen, n_heads, head_dim = q.size()
    valid = topk_idxs >= 0
    safe_idxs = topk_idxs.clamp_min(0)

    idx_expanded = safe_idxs.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    kv_exp = kv.unsqueeze(1).expand(-1, seqlen, -1, -1)
    selected = torch.gather(kv_exp, 2, idx_expanded)

    q_f = q.float()
    selected_f = selected.float()
    logits = torch.einsum("bshd,bskd->bshk", q_f, selected_f) * softmax_scale
    logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))

    scores_max = logits.amax(dim=-1, keepdim=True).clamp_min(-1e30)
    exp_logits = torch.exp(logits - scores_max)
    sink_logit = sink.view(1, 1, n_heads, 1).expand_as(scores_max)
    exp_sink = torch.exp(sink_logit - scores_max)
    sum_exp = exp_logits.sum(dim=-1, keepdim=True) + exp_sink

    acc_o = torch.einsum("bshk,bskd->bshd", exp_logits, selected_f)
    out = acc_o / sum_exp
    return out.to(q.dtype)
