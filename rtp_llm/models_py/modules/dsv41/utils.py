"""Shared DSV4 utility functions used across BF16 and FP8 paths."""

import weakref

import torch

try:  # DeepGEMM is a platform dependency, not an import-time requirement.
    from deep_gemm.utils.layout import (
        get_mn_major_tma_aligned_packed_ue8m0_tensor,
    )
except Exception:  # pragma: no cover - CPU/import-only environments
    get_mn_major_tma_aligned_packed_ue8m0_tensor = None

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.modules.base.common.embedding import Embedding
from rtp_llm.models_py.modules.factory.linear import LinearFactory

_V4_FP8_BLOCK_CFG = Fp8BlockWiseQuantConfig()
_FP8_LINEAR_REGISTRY = weakref.WeakSet()


class V41Embedding(Embedding):
    """Main TP embedding with the target/draft token shape preserved."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden = super().forward(input.reshape(-1).to(torch.int32))
        return hidden.reshape(*input.shape, hidden.shape[-1])


def _native_fp8_gemm_available() -> bool:
    """DeepGEMM FP8 kernels need SM89+; A100 (SM80) uses the reference path."""
    if getattr(torch.version, "hip", None) is not None:
        return False
    if not torch.cuda.is_available():
        return False
    if get_mn_major_tma_aligned_packed_ue8m0_tensor is None:
        return False
    return tuple(torch.cuda.get_device_capability()) >= (8, 9)


class V41MXFP8Linear(torch.nn.Module):
    """Native 32x32 checkpoint blocks, expanded to per-row MXFP8 scales."""

    def __init__(self, weight: torch.Tensor, scales: torch.Tensor):
        super().__init__()
        self.weight = weight
        self.N, self.K = weight.shape
        self._fallback = not _native_fp8_gemm_available()
        self._raw_scales = scales if self._fallback else None
        self._fallback_weight = None
        self.weight_scales = None if self._fallback else self.pack_scales(scales)

    def _dequant_weight(self) -> torch.Tensor:
        """Decode the original 32x32 FP8 blocks to BF16 once, on device.

        This is only reachable on pre-SM89 devices where the weights are
        already resident; the Engram tables never pass through this class.
        """
        if self._fallback_weight is None:
            rows = torch.arange(self.N, device=self.weight.device) // 32
            cols = torch.arange(self.K, device=self.weight.device) // 32
            scale = self._raw_scales.float().index_select(0, rows)
            scale = scale.index_select(1, cols)
            decoded = self.weight.float() * scale
            self._fallback_weight = decoded.to(torch.bfloat16)
            self._raw_scales = None
        return self._fallback_weight

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
        if self._fallback:
            # Keep the caller's BF16 tensor; forward_quantized detects this
            # sentinel pair and decodes the resident FP8 blocks instead.
            flat = (
                x
                if x.ndim == 2 and x.shape[1] == self.K and not x.requires_grad
                else x.reshape(-1, self.K)
            )
            return flat.contiguous(), None
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
        if self._fallback:
            if x_q.dtype == torch.float8_e4m3fn:
                raise RuntimeError(
                    "A100 reference V41MXFP8Linear cannot consume a foreign FP8 "
                    "activation; callers must use _quantize_input on this module"
                )
            shape = (*x_q.shape[:-1], self.N)
            result = torch.nn.functional.linear(
                x_q.reshape(-1, self.K),
                self._dequant_weight().to(x_q.dtype),
            ).reshape(shape)
            if out is not None:
                out.copy_(result)
                return out
            return result
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

    def forward(self, x: torch.Tensor, out=None):
        if x.numel() == 0:
            return out if out is not None else x.new_empty((*x.shape[:-1], self.N))
        if self._fallback:
            result = torch.nn.functional.linear(
                x, self._dequant_weight().to(x.dtype)
            )
            if out is not None:
                out.copy_(result)
                return out
            return result
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
