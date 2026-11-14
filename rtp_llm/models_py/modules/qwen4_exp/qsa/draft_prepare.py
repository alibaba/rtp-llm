"""Fuse draft RoPE generation/application, packed layout and score metadata.

Uses the same libdevice trig and two BF16 product rounding boundaries as the
qualified target preparation and rope_triton kernels. Read-only preparation;
the separately validated writer still owns side-cache mutation and undo.
"""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from rtp_llm.models_py.kernels.cuda.qsa.draft_prepare_triton import _prepare_draft_qsa


def try_prepare(q, positions, sources, visible, frequency, *, rope_config):
    from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import is_qsa_rope_style

    scale = float(getattr(rope_config, "scale", 1.0))
    # The caller retains graph metadata/ownership validation before mutation.
    if not (
        q.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(q.device) == (12, 0)
        and q.dtype == torch.bfloat16
        and q.ndim == 3
        and q.shape[1:] == (4, 128)
        and q.stride(-1) == 1
        and q.stride(-2) == 128
        and q.stride(0) >= 512
        and 1 <= q.shape[0] <= 32
        and positions.shape == sources.shape == (q.shape[0],)
        and positions.dtype == sources.dtype == torch.int64
        and visible.dtype == torch.int32
        and visible.ndim == 2
        and visible.shape[1] == 4
        and visible.numel() == q.shape[0]
        and isinstance(frequency, torch.Tensor)
        and frequency.dtype == torch.float32
        and frequency.shape == (32,)
        and all(
            x.device == q.device and x.is_contiguous()
            for x in (positions, sources, visible, frequency)
        )
        and math.isfinite(scale)
        and scale > 0
        and int(rope_config.dim) == 64
        and int(rope_config.index_factor) in (1, 3)
        and is_qsa_rope_style(rope_config, "Base")
        and bool(getattr(rope_config, "indexer_is_neox_style", True))
        and not q.requires_grad
    ):
        return None
    padded = torch.empty((*visible.shape, 4, 128), dtype=q.dtype, device=q.device)
    cosine = torch.empty((q.shape[0], 64), dtype=q.dtype, device=q.device)
    sine = torch.empty_like(cosine)
    weights = torch.empty((q.shape[0], 4), dtype=torch.float32, device=q.device)
    lengths = torch.empty_like(visible)
    _prepare_draft_qsa[(q.shape[0], 4)](
        q,
        positions,
        sources,
        visible,
        frequency,
        padded,
        cosine,
        sine,
        weights,
        lengths,
        QS=q.stride(0),
        QH=q.stride(1),
        H=4,
        INV_SCALE=1.0 / scale,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return padded, cosine, sine, weights, lengths


def try_prepare_metadata(
    q, row_lengths, prefixes, cu, frequency, *, rope_config, q_gamma=None, norm_eps=1e-6
):
    """Prepare fixed four-lane queries directly from owner-validated metadata.

    If q_gamma is supplied, consume raw Q and preserve the exact RMSNorm
    reduction and BF16 boundary before RoPE. Unsupported inputs return None
    before any writes; the owner normalizes Q before using its original path.
    """
    from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import is_qsa_rope_style

    scale = float(getattr(rope_config, "scale", 1.0))
    if not (
        q.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(q.device) == (12, 0)
        and q.dtype == torch.bfloat16
        and q.ndim == 3
        and q.shape[1:] == (4, 128)
        and q.stride(-1) == 1
        and q.stride(-2) == 128
        and q.stride(0) >= 512
        and row_lengths.ndim == prefixes.ndim == cu.ndim == 1
        and 1 <= row_lengths.numel() <= 8
        and q.shape[0] == row_lengths.numel() * 4
        and prefixes.shape == row_lengths.shape
        and cu.numel() == row_lengths.numel() + 1
        and all(
            t.dtype == torch.int32 and t.device == q.device and t.is_contiguous()
            for t in (row_lengths, prefixes, cu)
        )
        and isinstance(frequency, torch.Tensor)
        and frequency.shape == (32,)
        and frequency.dtype == torch.float32
        and frequency.device == q.device
        and frequency.is_contiguous()
        and not frequency.requires_grad
        and not q.requires_grad
        and math.isfinite(scale)
        and scale > 0
        and int(rope_config.dim) == 64
        and int(rope_config.index_factor) in (1, 3)
        and is_qsa_rope_style(rope_config, "Base")
        and bool(getattr(rope_config, "indexer_is_neox_style", True))
        and (
            q_gamma is None
            or (
                q_gamma.shape == (128,)
                and q_gamma.dtype in (torch.bfloat16, torch.float32)
                and q_gamma.device == q.device
                and q_gamma.is_contiguous()
                and not q_gamma.requires_grad
                and isinstance(norm_eps, (float, int))
                and math.isfinite(norm_eps)
                and norm_eps >= 0
            )
        )
    ):
        return None
    batch, rows = row_lengths.numel(), q.shape[0]
    padded = torch.empty((batch, 4, 4, 128), dtype=q.dtype, device=q.device)
    cosine = torch.empty((rows, 64), dtype=q.dtype, device=q.device)
    sine = torch.empty_like(cosine)
    weights = torch.empty((rows, 4), dtype=torch.float32, device=q.device)
    lengths = torch.empty((batch, 4), dtype=torch.int32, device=q.device)
    visible = torch.empty_like(lengths)
    _prepare_draft_qsa[(rows, 4)](
        q,
        None,
        None,
        visible,
        frequency,
        padded,
        cosine,
        sine,
        weights,
        lengths,
        QS=q.stride(0),
        QH=q.stride(1),
        H=4,
        INV_SCALE=1.0 / scale,
        FROM_METADATA=True,
        ROW_LENGTHS=row_lengths,
        PREFIX=prefixes,
        CU=cu,
        BATCH=batch,
        RB=triton.next_power_of_2(batch),
        RAW_QUERY=q_gamma is not None,
        QG=q_gamma,
        EPS=float(norm_eps) if q_gamma is not None else 1e-6,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return padded, cosine, sine, weights, lengths, visible
