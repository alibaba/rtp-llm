"""Shared DSV4 utility functions used across BF16 and FP8 paths."""

import torch
from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
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


def _v4_fp8_scale_relabelled_fp32(
    w: torch.Tensor, scale: torch.Tensor
) -> tuple:
    """SM90 counterpart of ``_repack_v4_fp8_scale_to_int32``.

    The int32-packed UE8M0 scale layout is a Blackwell recipe:
    ``CudaFp8DeepGEMMLinear.__init__`` only interprets ``(weight, scale)``
    under that convention when ``is_deep_gemm_e8m0_used()`` — i.e. device
    capability major in {10, 12}. On Hopper it instead takes the fp32
    branch, which reads the shapes *transposed* and then reshapes them
    back::

        self.K, self.N = self.weight.shape
        self.weight   = self.weight.reshape(self.N, self.K)

    So handing it a packed scale on H20 raises "int32 e8m0 scale N: …" even
    though the tensors are internally consistent — the tensors are fine,
    the branch is wrong. SM90 ``fp8_gemm_nt`` wants exactly
    ``([N, K] fp8, [N/128, K/128] fp32)``, the same recipe
    ``GroupedFP8Strategy`` already drives
    ``m_grouped_fp8_gemm_nt_contiguous`` with, so all that is needed is the
    plain fp32 cast plus a pre-relabel that the constructor's reshape
    undoes.

    Returns ``(w_relabelled, s_relabelled)`` shaped ``[K, N]`` and
    ``[K/128, N/128]``. Both relabels are pure views of contiguous data,
    not transposes, so the constructor's ``reshape`` restores the original
    layout bit-for-bit; it also leaves ``scale_ue8m0`` False, which
    correctly disables the UE8M0 activation-quant fast paths that have no
    SM90 kernel.
    """
    assert scale.dtype == torch.float8_e8m0fnu, f"unexpected scale dtype {scale.dtype}"
    assert scale.dim() == 2 and w.dim() == 2, "expected 2D weight and scale"
    n, k = w.shape
    n_blk, k_blk = scale.shape
    assert n_blk == n // 128 and k_blk == k // 128, (
        f"scale {tuple(scale.shape)} is not the 128x128 block grid of "
        f"weight {tuple(w.shape)}"
    )
    return w.reshape(k, n), scale.float().reshape(k_blk, n_blk)


def _v4_fp8_linear(w: torch.Tensor, s: torch.Tensor):
    """Build a CudaFp8DeepGEMMLinear from raw V4 FP8 weight + scale tensors."""
    assert s is not None, "expected non-null FP8 scale"
    if s.dtype == torch.float8_e8m0fnu:
        if is_deep_gemm_e8m0_used():
            s = _repack_v4_fp8_scale_to_int32(s)
        else:
            w, s = _v4_fp8_scale_relabelled_fp32(w, s)
    local = {"_w": w, "_s": s}
    return LinearFactory.create_linear_from_weights(
        local,
        "_w",
        "_s",
        quant_config=_V4_FP8_BLOCK_CFG,
    )


def _v4_fp8_linear_from_dict(weights: dict, weight_key: str, scale_key: str):
    """Backwards-compat bridge over ``_v4_fp8_linear`` for flat dict callers.

    The write-back is confined to the SM100 packed layout: the SM90 form is
    a *per-weight* relabel derived from that weight's N and K, so caching it
    under the raw scale key would be wrong for any other consumer."""
    w = weights[weight_key]
    s = weights[scale_key]
    if s.dtype == torch.float8_e8m0fnu and is_deep_gemm_e8m0_used():
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
