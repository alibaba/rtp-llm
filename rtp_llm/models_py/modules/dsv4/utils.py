"""Shared DSV4 utility functions used across BF16 and FP8 paths."""

import os
from types import MethodType

import torch
from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.utils.arch import is_sm120

_V4_FP8_BLOCK_CFG = Fp8BlockWiseQuantConfig()


class V41MXFP8Linear(torch.nn.Module):
    """Upstream group32/packed-int32 interface with an SM120 GEMM adapter.

    Ported from e4a9162f6f. Activation quantization remains upstream's native
    recipe; the SM120 provider is called only with already quantized values.
    Its standalone forward uses a different activation floor.
    """

    def __init__(self, weight: torch.Tensor, scales: torch.Tensor):
        super().__init__()
        self.weight = weight
        self.N, self.K = weight.shape
        self._sm120_linear = None
        if weight.is_cuda and is_sm120(weight.device):
            from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import (
                V41MXFP8Linear as SM120MXFP8Linear,
            )

            self._sm120_linear = SM120MXFP8Linear(weight, scales)
            self.weight_scales = self._sm120_linear._packed_scales
        else:
            self.weight_scales = self.pack_scales(scales)

    def pack_scales(self, scales: torch.Tensor) -> torch.Tensor:
        if scales.shape != ((self.N + 31) // 32, (self.K + 31) // 32):
            raise ValueError(
                f"Invalid V4.1 FP8 scales: {(self.N, self.K)}, {scales.shape}"
            )
        if self._sm120_linear is not None:
            return self._sm120_linear._interleave(
                scales.view(torch.uint8).repeat_interleave(32, dim=0)
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

    def forward_quantized(self, x_q: torch.Tensor, x_s: torch.Tensor, out=None):
        """Consume group32 E4M3/packed UE8M0 without another quantization.

        Deliberately no quantize_input method: the V4 input-reuse gate must
        remain closed for this linear. The caller owns the BF16 norm tensor.
        """
        if (
            x_q.ndim < 1
            or x_q.shape[-1] != self.K
            or x_q.dtype != torch.float8_e4m3fn
            or not x_q.is_cuda
            or x_q.device != self.weight.device
            or not x_q.is_contiguous()
        ):
            raise ValueError("expected contiguous CUDA E4M3 [...,K] input")
        m = x_q.numel() // self.K
        if (
            x_s.dtype != torch.int32
            or x_s.device != x_q.device
            or x_s.shape != (m, (self.K + 127) // 128)
            or x_s.stride() != (1, max(1, (m + 3) // 4 * 4))
        ):
            raise ValueError("expected group32 column-major TMA packed UE8M0 scales")
        shape = (*x_q.shape[:-1], self.N)
        if out is not None and any(
            torch._C._overlaps(out, source)
            for source in (x_q, x_s, self.weight, self.weight_scales)
        ):
            raise ValueError(
                "out must not share storage with quantized inputs or weights"
            )
        if out is not None and (
            out.shape != shape
            or out.dtype != torch.bfloat16
            or out.device != x_q.device
            or not out.is_contiguous()
        ):
            raise ValueError("out must be contiguous BF16 [...,N] on the input device")
        output = (
            out
            if out is not None
            else torch.empty(shape, device=x_q.device, dtype=torch.bfloat16)
        )
        if m:
            if self._sm120_linear is not None:
                # Each int32 contains four consecutive group32 exponents.
                # Force dense strides even for a singleton packed column:
                # contiguous() may retain its padded last stride.
                linear_scales = (
                    x_s.clone(memory_format=torch.contiguous_format)
                    .view(torch.uint8)[:, : self.K // 32]
                    .contiguous()
                )
                self._sm120_linear.forward_quantized(
                    x_q.reshape(m, self.K),
                    linear_scales,
                    out=output.reshape(m, self.N),
                )
                return output
            import deep_gemm

            deep_gemm.fp8_fp4_gemm_nt(
                (x_q.reshape(m, self.K), x_s),
                (self.weight, self.weight_scales),
                output.reshape(m, self.N),
                recipe=(1, 1, 32),
            )
        return output

    def forward(self, x: torch.Tensor, out=None):
        return self.forward_quantized(*self._quantize_input(x), out=out)


def _is_v41_fp8_scale(w: torch.Tensor, s: torch.Tensor) -> bool:
    return s.dtype == torch.float8_e8m0fnu and s.shape == (
        (w.shape[0] + 31) // 32,
        (w.shape[1] + 31) // 32,
    )


def _decode_ue8m0(scale: torch.Tensor, groups: int) -> torch.Tensor:
    if scale.dtype != torch.int32:
        return scale.float().contiguous()
    raw = (
        scale.contiguous().reshape(-1).view(torch.uint8).reshape(*scale.shape[:-1], -1)
    )
    return (raw[..., :groups].to(torch.int32) - 127).float().exp2()


def _sm120_forward_quantized(
    self,
    input_fp8: torch.Tensor,
    input_scales: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    from flashinfer.gemm import gemm_fp8_nt_groupwise

    rows, _ = self._validate_input(input_fp8)
    output = self._prepare_output(input_fp8, rows, out)
    if rows == 0:
        return output
    padded = (rows + 3) & ~3
    groups = (self.K + 127) // 128
    a_scale = _decode_ue8m0(input_scales, groups)
    if padded == rows:
        a, gemm_out = input_fp8.contiguous(), output
    else:
        a = torch.zeros(
            (padded, self.K), dtype=input_fp8.dtype, device=input_fp8.device
        )
        a[:rows].copy_(input_fp8)
        padded_scale = torch.ones(
            (padded, groups), dtype=torch.float32, device=input_fp8.device
        )
        padded_scale[:rows].copy_(a_scale)
        a_scale, gemm_out = padded_scale, None
    result = gemm_fp8_nt_groupwise(
        a,
        self.weight,
        a_scale,
        self._dsv4_sm120_weight_scale_fp32,
        scale_granularity_mnk=(1, 128, 128),
        scale_major_mode="K",
        out=gemm_out,
        out_dtype=torch.bfloat16,
    )
    if padded != rows:
        output.copy_(result[:rows])
    if self.bias is not None:
        output.add_(self.bias.to(output.dtype))
    return output


def _enable_sm120_cached_weight_scale(linear):
    weight = getattr(linear, "weight", None)
    if (
        os.environ.get("DSV4_SM120_CACHE_FP8_SCALES", "1") == "0"
        or weight is None
        or not weight.is_cuda
        or not is_sm120(weight.device)
    ):
        return linear
    weight_scale = _decode_ue8m0(linear.weight_scales, (linear.K + 127) // 128)
    if weight_scale.size(0) == linear.N:
        weight_scale = weight_scale[::128]
    linear._dsv4_sm120_weight_scale_fp32 = weight_scale.contiguous()
    linear.forward_quantized = MethodType(_sm120_forward_quantized, linear)
    return linear


def _repack_v4_fp8_scale_to_int32(scale: torch.Tensor) -> torch.Tensor:
    """V4 ckpt UE8M0 ``[N/128, K/128]`` to DeepGEMM int32-packed scale."""
    assert scale.dtype == torch.float8_e8m0fnu, f"unexpected scale dtype {scale.dtype}"
    assert scale.dim() == 2, f"unexpected scale dim {scale.dim()}"

    n_blk, _ = scale.shape
    n = n_blk * 128
    idx = torch.arange(n, device=scale.device) // 128
    scale_rep = scale.float().index_select(-2, idx)
    return get_mn_major_tma_aligned_packed_ue8m0_tensor(scale_rep)


def _v4_fp8_linear(w: torch.Tensor, s: torch.Tensor):
    """Build a CudaFp8DeepGEMMLinear from raw V4 FP8 weight + scale tensors."""
    assert s is not None, "expected non-null FP8 scale"
    if _is_v41_fp8_scale(w, s):
        return V41MXFP8Linear(w, s)
    # DeepGEMM's packed int32 UE8M0 contract is unsupported on consumer
    # Blackwell.  Keep the compact checkpoint scale for the SM120 CUTLASS
    # blockwise backend; only data-center architectures use the DeepGEMM
    # TMA-aligned packed representation.
    from rtp_llm.models_py.utils.arch import is_sm120

    if s.dtype == torch.float8_e8m0fnu and not is_sm120(s.device):
        s = _repack_v4_fp8_scale_to_int32(s)
    local = {"_w": w, "_s": s}
    linear = LinearFactory.create_linear_from_weights(
        local,
        "_w",
        "_s",
        quant_config=_V4_FP8_BLOCK_CFG,
    )
    return _enable_sm120_cached_weight_scale(linear)


def _v4_fp8_linear_from_dict(weights: dict, weight_key: str, scale_key: str):
    """Backwards-compat bridge over ``_v4_fp8_linear`` for flat dict callers."""
    w = weights[weight_key]
    s = weights[scale_key]
    if _is_v41_fp8_scale(w, s):
        return _v4_fp8_linear(w, s)
    from rtp_llm.models_py.utils.arch import is_sm120

    if s.dtype == torch.float8_e8m0fnu and not is_sm120(s.device):
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
