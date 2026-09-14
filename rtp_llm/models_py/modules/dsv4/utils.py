"""Shared DSV4 utility functions used across BF16 and FP8 paths."""
import os

import torch
from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.modules.factory.linear import LinearFactory

_V4_FP8_BLOCK_CFG = Fp8BlockWiseQuantConfig()


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
    return LinearFactory.create_linear_from_weights(
        local,
        "_w",
        "_s",
        quant_config=_V4_FP8_BLOCK_CFG,
    )


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
