"""Shared DSV4 utility functions used across BF16 and FP8 paths."""

import os
import weakref

import torch
from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.modules.base.common.embedding import Embedding
from rtp_llm.models_py.modules.factory.linear import LinearFactory

_V4_FP8_BLOCK_CFG = Fp8BlockWiseQuantConfig()
_FP8_LINEAR_REGISTRY = weakref.WeakSet()

# Pre-SM100 fallback caches an exact bf16 weight view only for linears at or
# below this element count (16MB bf16); larger (typically replicated) weights
# dequantize per call to keep the resident footprint at FP8 size.
# Weights up to this size keep a cached exact bf16 dequantization; larger
# ones dequantize per call. Set to 0 to free the cache entirely (memory-
# bound deployments trade dequant work for ~bytes-worth of headroom).
_V41_MXFP8_BF16_CACHE_MAX_NUMEL = int(
    os.environ.get("DSV41_MXFP8_BF16_CACHE_MAX_NUMEL", "8000000")
)
# Decode-sized batches dequantize weight-block-by-block instead of
# materializing the whole bf16 weight (graph-capture pools stay small).
_V41_MXFP8_CHUNKED_GEMM_MAX_ROWS = int(
    os.environ.get("DSV41_MXFP8_CHUNKED_GEMM_MAX_ROWS", "256")
)
# Chunk rows trade transient bytes against CUDA graph node count; see the
# matching knob in qlinear.py. 2048 bounds both for decode-graph capture.
_V41_MXFP8_DEQUANT_GEMM_CHUNK_ROWS = int(
    os.environ.get("DSV41_MXFP8_DEQUANT_CHUNK_ROWS", "128")
)


class V41Embedding(Embedding):
    """Main TP embedding with the target/draft token shape preserved."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden = super().forward(input.reshape(-1).to(torch.int32))
        return hidden.reshape(*input.shape, hidden.shape[-1])


class V41MXFP8Linear(torch.nn.Module):
    """Native 32x32 checkpoint blocks, expanded to per-row MXFP8 scales.

    SM100+ forwards through DeepGEMM's FP8xFP4 kernel. Pre-SM100 devices
    (e.g. Ada sm89) cannot JIT that kernel; they keep the original FP8
    weight plus an exact bf16 dequantization (e4m3 × 2^k is precise in
    bf16) and run cublas GEMMs — same checkpoint semantics, no new
    quantization.
    """

    def __init__(self, weight: torch.Tensor, scales: torch.Tensor):
        super().__init__()
        from rtp_llm.models_py.modules.dsv41.qlinear import native_fp4_gemm_supported

        self.N, self.K = weight.shape
        self.native_fp4_gemm = native_fp4_gemm_supported(
            weight.device if weight.is_cuda else None
        )
        if self.native_fp4_gemm:
            self.weight = weight
            self.weight_scales = self.pack_scales(scales)
            self._dequant_bf16 = None
        else:
            self.weight_scales = scales
            if weight.numel() <= _V41_MXFP8_BF16_CACHE_MAX_NUMEL:
                # Small (TP-sharded) linears: keep an exact bf16 view as the
                # canonical weight so decode steps never re-dequantize.
                self._dequant_bf16 = self._dequant_weight_bf16(weight, scales)
                self.weight = self._dequant_bf16
            else:
                # Large replicated weights (e.g. the sp_id-replicated shared
                # expert w13/w2, ~2.8GB/rank of bf16 copies across layers):
                # keep only the FP8 checkpoint bytes resident and dequantize
                # per call through the chunked exact path. The temporary is
                # recycled by the caching allocator.
                self.weight = weight
                self._dequant_bf16 = None

    @staticmethod
    def _dequant_weight_bf16(weight: torch.Tensor, scales: torch.Tensor):
        """Exact bf16 view of FP8 e4m3 weights with UE8M0 32x32 scales.

        Chunked over scale-row blocks so peak temporaries stay at one
        [32, K] bf16 block: e4m3 -> bf16 is exact and multiplying by a
        power-of-two scale keeps <=4 significand bits, so the bf16 product
        is exact (no fp32 intermediate, which OOMs a nearly-full card).
        """
        if scales.dtype != torch.float8_e8m0fnu:
            raise ValueError(f"unsupported V4.1 FP8 scale dtype {scales.dtype}")
        n, k = weight.shape
        if scales.shape != ((n + 31) // 32, (k + 31) // 32):
            raise ValueError(f"Invalid V4.1 FP8 scales: {(n, k)}, {scales.shape}")
        out = torch.empty((n, k), dtype=torch.bfloat16, device=weight.device)
        scales_f32 = scales.to(torch.float32)
        for i in range(scales_f32.shape[0]):
            rows = slice(i * 32, min((i + 1) * 32, n))
            col_scale = (
                scales_f32[i].repeat_interleave(32)[:k].to(torch.bfloat16)
            )
            out[rows] = weight[rows].to(torch.bfloat16) * col_scale
        return out

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

        flat = (
            x
            if x.ndim == 2 and x.shape[1] == self.K and not x.requires_grad
            else x.reshape(-1, self.K)
        ).contiguous()
        quantized, scales = sgl_per_token_group_quant_fp8(
            flat,
            group_size=32,
            eps=torch.finfo(torch.float32).tiny,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        return (
            quantized
            if quantized.shape == x.shape and not quantized.requires_grad
            else quantized.view(x.shape)
        ), scales

    def forward_quantized(self, x_q: torch.Tensor, x_s: torch.Tensor, out=None):
        """Consume group32 E4M3/packed UE8M0 without another quantization.

        Deliberately no quantize_input method: the V4 input-reuse gate must
        remain closed for this linear. The caller owns the BF16 norm tensor.
        Input layout and disjoint BF16 output storage are caller preconditions.
        """
        if not self.native_fp4_gemm:
            raise RuntimeError(
                "V41MXFP8Linear pre-SM100 fallback does not consume pre-quantized "
                "inputs; packed-lookup/fused callers must gate on native_fp4_gemm"
            )
        m = x_q.numel() // self.K
        shape = (*x_q.shape[:-1], self.N)
        output = (
            out
            if out is not None
            else torch.empty(shape, device=x_q.device, dtype=torch.bfloat16)
        )
        if m:
            import deep_gemm

            deep_gemm.fp8_fp4_gemm_nt(
                (
                    (
                        x_q
                        if x_q.shape == (m, self.K) and not x_q.requires_grad
                        else x_q.reshape(m, self.K)
                    ),
                    x_s,
                ),
                (self.weight, self.weight_scales),
                (
                    output
                    if output.shape == (m, self.N) and not output.requires_grad
                    else output.reshape(m, self.N)
                ),
                recipe=(1, 1, 32),
            )
        return output

    def _forward_chunked_dequant_gemm(self, x_flat_bf16: torch.Tensor):
        """Bounded-transient exact GEMM for decode-sized batches.

        Dequantizes one 128-row block of the FP8 weight at a time (e4m3 x
        2^k stays exact in bf16) and GEMMs it into the output slice, so the
        full dequantized weight is never materialized. Decode CUDA graph
        captures previously OOM'd on the 314MB whole-weight temporary for
        the Engram wkv projection; this keeps per-call transients at a few
        MB per captured batch size while reading the same checkpoint bytes.
        """
        m = x_flat_bf16.shape[0]
        output = torch.empty(
            (m, self.N), dtype=torch.bfloat16, device=x_flat_bf16.device
        )
        chunk = _V41_MXFP8_DEQUANT_GEMM_CHUNK_ROWS
        scales = self.weight_scales.to(torch.float32)
        for begin in range(0, self.N, chunk):
            end = min(begin + chunk, self.N)
            col_scale = (
                scales[begin // 32 : (end + 31) // 32]
                .repeat_interleave(32, 0)[: end - begin]
                .repeat_interleave(32, 1)[:, : self.K]
                .to(torch.bfloat16)
            )
            w_chunk = self.weight[begin:end].to(torch.bfloat16) * col_scale
            output[:, begin:end] = torch.nn.functional.linear(x_flat_bf16, w_chunk)
            del col_scale, w_chunk
        return output

    def forward(self, x: torch.Tensor, out=None):
        if x.numel() == 0:
            return out if out is not None else x.new_empty((*x.shape[:-1], self.N))
        if not self.native_fp4_gemm:
            # Exact pre-SM100 path: bf16 activations against the exact bf16
            # dequantization of the FP8 checkpoint weights (cached for small
            # linears, per-call chunked for large replicated ones).
            w_bf16 = self._dequant_bf16
            if w_bf16 is None and x.shape[-1] == self.K:
                rows = x.numel() // self.K
                if rows <= _V41_MXFP8_CHUNKED_GEMM_MAX_ROWS:
                    y = self._forward_chunked_dequant_gemm(
                        x.reshape(-1, self.K).to(torch.bfloat16)
                    ).view(*x.shape[:-1], self.N)
                    if out is None:
                        return y
                    out.copy_(y)
                    return out
            temporary = w_bf16 is None
            if temporary:
                w_bf16 = self._dequant_weight_bf16(self.weight, self.weight_scales)
            y = torch.nn.functional.linear(x.to(torch.bfloat16), w_bf16)
            if temporary:
                del w_bf16
            if out is None:
                return y
            out.copy_(y)
            return out
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
    qw, qs, kw, ks = (weights[key] for key in (q_weight, q_scale, kv_weight, kv_scale))
    if not (
        qw.ndim == kw.ndim == 2
        and qw.dtype == kw.dtype == torch.float8_e4m3fn
        and (qw.device == kw.device == qs.device == ks.device)
        and qw.is_cuda
        and (qw.shape[1] == kw.shape[1])
        and (qw.shape[0] % 32 == kw.shape[0] % 32 == qw.shape[1] % 32 == 0)
        and _is_v41_fp8_scale(qw, qs)
        and _is_v41_fp8_scale(kw, ks)
    ):
        return None
    merged_w = torch.cat((qw.view(torch.uint8), kw.view(torch.uint8))).view(qw.dtype)
    merged_s = torch.cat((qs.view(torch.uint8), ks.view(torch.uint8))).view(qs.dtype)
    linear = V41MXFP8Linear(merged_w, merged_s)
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
    if s.dtype == torch.float8_e8m0fnu:
        s = _repack_v4_fp8_scale_to_int32(s)
    linear = LinearFactory.create_linear_from_weights(
        {"_w": w, "_s": s}, "_w", "_s", quant_config=_V4_FP8_BLOCK_CFG
    )
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
