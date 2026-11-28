"""Shared DSV4 utility functions used across BF16 and FP8 paths."""

from typing import Optional

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.kernels.cuda.quant_layouts import is_e8m0_scale
from rtp_llm.models_py.modules.factory.linear import LinearFactory

_V4_FP8_BLOCK_CFG = Fp8BlockWiseQuantConfig()


def _repack_v4_fp8_scale_to_int32(
    scale: torch.Tensor,
    n_rows: Optional[int] = None,
    k_cols: Optional[int] = None,
) -> torch.Tensor:
    """V4 ckpt UE8M0 ``[ceil(N/128), K/128]`` to DeepGEMM int32-packed scale."""
    from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

    if not is_e8m0_scale(scale):
        raise ValueError(f"unexpected compact V4 scale dtype {scale.dtype}")
    if scale.dim() != 2:
        raise ValueError(f"unexpected compact V4 scale dim {scale.dim()}")

    if n_rows is not None and k_cols is not None:
        if k_cols % 128 != 0:
            raise ValueError(f"V4 FP8 linear requires K % 128 == 0, got K={k_cols}")
        expected_shape = ((n_rows + 127) // 128, k_cols // 128)
        if tuple(scale.shape) != expected_shape:
            raise ValueError(
                f"V4 compact scale shape mismatch: expected {expected_shape} "
                f"for weight {(n_rows, k_cols)}, got {tuple(scale.shape)}"
            )

    n = n_rows if n_rows is not None else scale.shape[0] * 128
    idx = torch.arange(n, device=scale.device) // 128
    scale_rep = scale.float().index_select(-2, idx)
    return get_mn_major_tma_aligned_packed_ue8m0_tensor(scale_rep)


def _v4_fp8_linear(w: torch.Tensor, s: torch.Tensor):
    """Build an FP8 PER_BLOCK linear from raw V4 FP8 weight + scale tensors."""
    if w.dim() != 2:
        raise ValueError(f"V4 FP8 weight must be 2D, got {tuple(w.shape)}")
    if s is None:
        raise ValueError("expected non-null FP8 scale")
    if w.device != s.device:
        raise ValueError(
            f"V4 FP8 weight device {w.device} != scale device {s.device}"
        )
    if s.dtype == torch.int32:
        if w.shape[1] % 128 != 0:
            raise ValueError(
                f"V4 FP8 linear requires K % 128 == 0, got K={w.shape[1]}"
            )
        expected_packed = (w.shape[0], (w.shape[1] + 511) // 512)
        if tuple(s.shape) != expected_packed:
            raise ValueError(
                f"V4 packed scale shape mismatch: expected {expected_packed} "
                f"for weight {tuple(w.shape)}, got {tuple(s.shape)}"
            )
    elif is_e8m0_scale(s):
        s = _repack_v4_fp8_scale_to_int32(s, n_rows=w.shape[0], k_cols=w.shape[1])
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
    if is_e8m0_scale(s):
        if w.dim() != 2:
            raise ValueError(f"V4 FP8 weight must be 2D, got {tuple(w.shape)}")
        if w.device != s.device:
            raise ValueError(
                f"V4 FP8 weight device {w.device} != scale device {s.device}"
            )
        s = _repack_v4_fp8_scale_to_int32(s, n_rows=w.shape[0], k_cols=w.shape[1])
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
