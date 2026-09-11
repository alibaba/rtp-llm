"""Shared DSV4 utility functions used across BF16 and FP8 paths."""
import os
from types import MethodType

import torch
from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.modules.factory.linear import LinearFactory

_V4_FP8_BLOCK_CFG = Fp8BlockWiseQuantConfig()
def _decode_ue8m0(scale: torch.Tensor, groups: int) -> torch.Tensor:
    if scale.dtype != torch.int32:
        return scale.float().contiguous()
    # The `.reshape(-1)` before the dtype view is load-bearing, not decoration.
    #
    # DeepGEMM's packed UE8M0 scales are MN-major: `as_strided((N, k_packed),
    # (1, aligned_rows))`, so stride(-1) is the LARGE one. When k_packed > 1 the
    # tensor reports is_contiguous() == False and `.contiguous()` copies it into
    # stride(-1) == 1, so the old `scale.contiguous().view(torch.uint8)` worked.
    # But when **k_packed == 1** (i.e. K <= 512) the LAST DIM HAS SIZE 1, and
    # PyTorch ignores size-1 dims for contiguity: is_contiguous() returns True,
    # `.contiguous()` is a NO-OP returning the same object, and the view then dies
    # with `self.stride(-1) must be 1 to view Int as Byte ... but got <aligned_rows>`.
    # DSV4 production only ever escaped this because K is 4096/2048 (k_packed 8/4);
    # any small-K fp8 linear on SM120, and the shared-expert unit-test fixture
    # (dim=256 -> k_packed=1), hit it. `.reshape(-1)` collapses to a 1-D view whose
    # stride is 1 by construction, so the reinterpret is always legal.
    #
    # Verified semantics-preserving: on the production shape (N=2048, k_packed=8)
    # the decoded result is torch.equal to the old expression; on the size-1 case
    # it decodes correctly where the old one raised.
    raw = scale.contiguous().reshape(-1).view(torch.uint8).reshape(*scale.shape[:-1], -1)
    # P1b: uint8 -> float32 directly, then in-place sub/exp2 — identical
    # values (exponents <= 255 are exact in fp32), 3 kernels instead of 5.
    return raw[..., :groups].float().sub_(127).exp2_()
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
        # P1a: pad rows are computed per-row and discarded by result[:rows] —
        # init only the <=3-row pad tail instead of full-buffer zeros/ones.
        a = torch.empty((padded, self.K), dtype=input_fp8.dtype, device=input_fp8.device)
        a[:rows].copy_(input_fp8)
        a[rows:].zero_()
        padded_scale = torch.empty(
            (padded, groups), dtype=torch.float32, device=input_fp8.device
        )
        padded_scale[:rows].copy_(a_scale)
        padded_scale[rows:].fill_(1.0)
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
        or torch.cuda.get_device_capability(weight.device)[0] != 12
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


class _V4Fp8ScaledMMLinear(torch.nn.Module):
    """Rowwise-FP8 linear via torch._scaled_mm (cuBLASLt) — DSV4_DENSE_SCALEDMM=1.

    Weights are dequantized from the per-128-block UE8M0 checkpoint layout and
    requantized per-output-row at construction; activations quantize per-row
    at forward. Numerics differ from the per-128-block DeepGEMM incumbent
    (rowwise is coarser along K) — gate on the in-engine logit-drift check.
    Offline race: -26.6% weighted on the dense GEMM pool
    (results_20260904_p2_shapes/race_dense.json).
    """

    def __init__(self, w_fp8: torch.Tensor, s_e8m0: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        N, K = w_fp8.shape
        sf = s_e8m0.float()  # [N/128, K/128], exact powers of two
        w_deq = w_fp8.float() * sf.repeat_interleave(128, dim=0)[:N,].repeat_interleave(128, dim=1)[:, :K]
        row_max = w_deq.abs().amax(dim=1).clamp(min=1e-12)
        w_row = (w_deq * (448.0 / row_max).unsqueeze(1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        self.register_buffer("_w_t", w_row.t())  # (K, N) column-major for _scaled_mm
        self.register_buffer("_scale_b", (row_max / 448.0).view(1, N).float())
        self._bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1])
        xf = x2.float()
        amax = xf.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        xq = (xf * (448.0 / amax)).clamp(-448, 448).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            xq, self._w_t, scale_a=(amax / 448.0).float(), scale_b=self._scale_b,
            out_dtype=torch.bfloat16,
        )
        out = out.reshape(*x.shape[:-1], out.shape[-1])
        return out + self._bias if self._bias is not None else out


def _v4_fp8_linear(w: torch.Tensor, s: torch.Tensor):
    """Build a CudaFp8DeepGEMMLinear from raw V4 FP8 weight + scale tensors."""
    assert s is not None, "expected non-null FP8 scale"
    if os.environ.get("DSV4_DENSE_SCALEDMM") == "1" and s.dtype == torch.float8_e8m0fnu:
        return _V4Fp8ScaledMMLinear(w, s).to(w.device)
    if s.dtype == torch.float8_e8m0fnu:
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
