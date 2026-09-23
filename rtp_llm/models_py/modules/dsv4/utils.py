"""Shared DSV4 utility functions used across BF16 and FP8 paths."""

import os
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
        self.weight_scales = self.pack_scales(scales)

    def pack_scales(self, scales: torch.Tensor) -> torch.Tensor:
        if scales.shape != ((self.N + 31) // 32, (self.K + 31) // 32):
            raise ValueError(
                f"Invalid V4.1 FP8 scales: {(self.N, self.K)}, {scales.shape}"
            )
        rows = torch.arange(self.N, device=scales.device) // 32
        return get_mn_major_tma_aligned_packed_ue8m0_tensor(
            scales.float().index_select(0, rows)
        )

    def _quantize_input(self, x: torch.Tensor):
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        flat = x.reshape(-1, self.K).contiguous()
        quantized, scales = sgl_per_token_group_quant_fp8(
            flat,
            group_size=32,
            eps=torch.finfo(torch.float32).tiny,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        return quantized.view(x.shape), scales

    def forward_quantized(
        self, quantized: torch.Tensor, scales: torch.Tensor, out=None
    ):
        import deep_gemm

        shape = (*quantized.shape[:-1], self.N)
        output = (
            out
            if out is not None
            else torch.empty(shape, device=quantized.device, dtype=torch.bfloat16)
        )
        if quantized.numel() == 0:
            return output
        deep_gemm.fp8_fp4_gemm_nt(
            (quantized.reshape(-1, self.K), scales),
            (self.weight, self.weight_scales),
            output.reshape(-1, self.N),
            recipe=(1, 1, 32),
        )
        return output

    def forward(self, x: torch.Tensor, out=None):
        if x.numel() == 0:
            return out if out is not None else x.new_empty((*x.shape[:-1], self.N))
        return self.forward_quantized(*self._quantize_input(x), out=out)


def _is_v41_fp8_scale(w: torch.Tensor, s: torch.Tensor) -> bool:
    return s.dtype == torch.float8_e8m0fnu and s.shape == (
        (w.shape[0] + 31) // 32,
        (w.shape[1] + 31) // 32,
    )


def merge_v41_qkv_weights(weights, q_weight, q_scale, kv_weight, kv_scale):
    """Merge replicated projections and make checkpoint entries share storage.

    The separate linears remain available for fallback and prefill, but their
    weight views must point at this allocation instead of retaining the two
    original allocations. Packed scales have an exact N-major leading
    dimension, so each fallback linear still packs its own small scale tensor.
    """
    if os.environ.get("DSV41_FUSED_QKV", "1") != "1":
        return None
    qw, qs, kw, ks = (weights[key] for key in (q_weight, q_scale, kv_weight, kv_scale))
    if not (
        qw.ndim == kw.ndim == 2
        and qw.dtype == kw.dtype == torch.float8_e4m3fn
        and qw.device == kw.device == qs.device == ks.device
        and qw.is_cuda
        and qw.shape[1] == kw.shape[1]
        and qw.shape[0] % 32 == kw.shape[0] % 32 == qw.shape[1] % 32 == 0
        and _is_v41_fp8_scale(qw, qs)
        and _is_v41_fp8_scale(kw, ks)
    ):
        return None
    with feature_weights_region():
        # Concatenate bytes: torch.cat support for E8M0 is backend-dependent.
        merged_w = torch.cat((qw.view(torch.uint8), kw.view(torch.uint8))).view(
            qw.dtype
        )
        merged_s = torch.cat((qs.view(torch.uint8), ks.view(torch.uint8))).view(
            qs.dtype
        )
        linear = V41MXFP8Linear(merged_w, merged_s)
    linear._sleep_raw_weight_source = merged_w
    linear._sleep_raw_scale_source = merged_s
    linear._sleep_row_slice = None
    linear._sleep_col_slice = None
    weights[q_weight], weights[kv_weight] = merged_w.split((qw.shape[0], kw.shape[0]))
    weights[q_scale], weights[kv_scale] = merged_s.split((qs.shape[0], ks.shape[0]))
    return linear


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
