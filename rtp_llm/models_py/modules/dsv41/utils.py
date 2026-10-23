"""Shared DSV4 utility functions used across BF16 and FP8 paths."""

import os
import weakref
from functools import lru_cache

import torch
from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.modules.base.common.embedding import Embedding
from rtp_llm.models_py.modules.factory.linear import LinearFactory

_V4_FP8_BLOCK_CFG = Fp8BlockWiseQuantConfig()
_FP8_LINEAR_REGISTRY = weakref.WeakSet()


@lru_cache(maxsize=None)
def _dense_fp4_device_supported(device_index: int) -> bool:
    if torch.version.hip is not None or not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability(device_index)
    except (AssertionError, RuntimeError, ValueError):
        return False
    if major != 10:
        return False
    try:
        from rtp_llm.models_py.kernels.cuda import deepgemm_wrapper
    except Exception:
        return False
    return deepgemm_wrapper.has_deep_gemm()


def dense_fp4_decode_supported(device: torch.device) -> bool:
    """SM100+ packed-FP4 B operand, opt-out via ``DSV41_DENSE_FP4=0``."""
    if os.environ.get("DSV41_DENSE_FP4", "1") != "1":
        return False
    if device.type != "cuda":
        return False
    return _dense_fp4_device_supported(
        device.index if device.index is not None else torch.cuda.current_device()
    )


def _unpack_ue8m0_row_scales(packed: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Invert the MN-major packed UE8M0 layout back to one scale per 32 K.

    ``packed`` is the DeepGEMM int32 tensor with logical shape ``[N, K/128]``,
    physically N-contiguous. Four consecutive K-block exponents share each
    int32, so a logical row-major copy exposes them in increasing K order.
    """
    if (
        packed.dtype != torch.int32
        or packed.shape != (n, k // 128)
        or packed.stride(0) != 1
        or packed.stride(1) < n
        or packed.stride(1) % 4
    ):
        raise ValueError(f"Invalid packed V4.1 scale layout: {packed.shape}")
    if k % 128 or (k // 32) % 4:
        raise ValueError(f"Invalid V4.1 FP4 scale K: {k}")
    ue8m0 = packed.contiguous().view(torch.uint8).view(n, k // 32)
    return (ue8m0.to(torch.int32).clamp_min(1) << 23).view(torch.float32)


def quantize_fp8_weight_to_fp4(
    weight: torch.Tensor, packed_scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Requantize an MXFP8 weight to DeepGEMM's packed MXFP4 B operand.

    Returns ``([N, K/2] int8 K-major codes, [N, K/128] int32 MN-major packed
    UE8M0 scales)`` or ``None`` when the geometry or device cannot use the
    dense FP4 kernel. The source FP8 weight/scales are never modified.
    """
    n, k = (int(dim) for dim in weight.shape)
    if (
        weight.dtype != torch.float8_e4m3fn
        or not weight.is_cuda
        or not weight.is_contiguous()
        or n <= 0
        or k <= 0
        or n % 128
        or k % 128
        or (k // 32) % 4
        or not dense_fp4_decode_supported(weight.device)
    ):
        return None
    if packed_scales.shape != (n, k // 128):
        return None

    scale = _unpack_ue8m0_row_scales(packed_scales, n, k)
    # Exact dequantization of the FP8 payload before the MXFP4 rounding.
    dequant = weight.float().view(n, k // 32, 32)
    dequant.mul_(scale.unsqueeze(-1))
    dequant = dequant.view(n, k)
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import per_token_cast_to_fp4

    codes, scales = per_token_cast_to_fp4(dequant, use_ue8m0=True, gran_k=32)
    del dequant
    return codes, get_mn_major_tma_aligned_packed_ue8m0_tensor(scales)


class V41Embedding(Embedding):
    """Main TP embedding with the target/draft token shape preserved."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden = super().forward(input.reshape(-1).to(torch.int32))
        return hidden.reshape(*input.shape, hidden.shape[-1])


class V41MXFP8Linear(torch.nn.Module):
    """Native 32x32 checkpoint blocks, expanded to per-row MXFP8 scales."""

    def __init__(self, weight: torch.Tensor, scales: torch.Tensor):
        super().__init__()
        self.weight = weight
        self.N, self.K = weight.shape
        self.weight_scales = self.pack_scales(scales)
        # Decode-only packed MXFP4 B operand. ``weight``/``weight_scales`` stay
        # authoritative for prefill, reload and the fused FP8 paths.
        self.weight_fp4: torch.Tensor | None = None
        self.weight_fp4_scales: torch.Tensor | None = None
        self._dense_fp4_eligible = False

    def mark_dense_fp4(self) -> bool:
        """Opt this production projection into packed-FP4 decode weights."""
        if self._dense_fp4_eligible:
            return True
        if (
            self.weight.dtype != torch.float8_e4m3fn
            or self.weight_scales.dtype != torch.int32
            or self.N % 128
            or self.K % 128
            or (self.K // 32) % 4
        ):
            return False
        self._dense_fp4_eligible = True
        return True

    def enable_dense_fp4(self, force: bool = False) -> bool:
        """Build the FP4 copy after checkout/load; refresh it in place on reload."""
        if not self._dense_fp4_eligible and not self.mark_dense_fp4():
            return False
        if self.weight_fp4 is not None and not force:
            return True
        built = quantize_fp8_weight_to_fp4(self.weight, self.weight_scales)
        if built is None:
            if force and self.weight_fp4 is not None:
                raise RuntimeError("V4.1 FP4 weight reload is unsupported")
            return False
        codes, scales = built
        if self.weight_fp4 is None:
            self.weight_fp4 = codes
            self.weight_fp4_scales = scales
        else:
            if self.weight_fp4.shape != codes.shape:
                raise RuntimeError(
                    f"V4.1 FP4 weight geometry changed: {self.weight_fp4.shape}"
                )
            if self.weight_fp4_scales.shape != scales.shape:
                raise RuntimeError(
                    f"V4.1 FP4 scale geometry changed: {self.weight_fp4_scales.shape}"
                )
            # Keep the buffer address captured by existing CUDA graphs.
            self.weight_fp4.copy_(codes)
            self.weight_fp4_scales.copy_(scales)
        return True

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

    def forward_quantized(
        self,
        x_q: torch.Tensor,
        x_s: torch.Tensor,
        out=None,
        *,
        use_fp4: bool = False,
    ):
        """Consume group32 E4M3/packed UE8M0 without another quantization.

        Deliberately no quantize_input method: the V4 input-reuse gate must
        remain closed for this linear. The caller owns the BF16 norm tensor.
        Input layout and disjoint BF16 output storage are caller preconditions.
        Decode consumers opt into the FP4 copy; prefill and fallbacks keep FP8.
        """
        m = x_q.numel() // self.K
        shape = (*x_q.shape[:-1], self.N)
        output = (
            out
            if out is not None
            else torch.empty(shape, device=x_q.device, dtype=torch.bfloat16)
        )
        if m:
            import deep_gemm

            weight, weight_scales = (
                (self.weight_fp4, self.weight_fp4_scales)
                if use_fp4 and self.weight_fp4 is not None
                else (self.weight, self.weight_scales)
            )
            deep_gemm.fp8_fp4_gemm_nt(
                (
                    (
                        x_q
                        if x_q.shape == (m, self.K) and not x_q.requires_grad
                        else x_q.reshape(m, self.K)
                    ),
                    x_s,
                ),
                (weight, weight_scales),
                (
                    output
                    if output.shape == (m, self.N) and not output.requires_grad
                    else output.reshape(m, self.N)
                ),
                recipe=(1, 1, 32),
            )
        return output

    def forward(self, x: torch.Tensor, out=None, *, use_fp4: bool = False):
        if x.numel() == 0:
            return out if out is not None else x.new_empty((*x.shape[:-1], self.N))
        return self.forward_quantized(
            *self._quantize_input(x), out=out, use_fp4=use_fp4
        )


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
