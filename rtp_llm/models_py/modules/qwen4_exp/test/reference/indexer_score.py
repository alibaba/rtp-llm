"""Native 4-head scoring kernel for the QSA indexer (prefill shape).

Why this kernel exists
----------------------
The DSv4 indexer scores through DeepGEMM's non-paged ``fp8_mqa_logits``, whose
tile is a fixed ``block_qh = 128`` query-head rows, i.e.
``block_q = 128 / num_heads``.  In the DeepGEMM build this repo pins
(2.2.0 / ``9b680f4``) the host side asserts ``seq_len_alignment % block_q == 0``
with ``seq_len_alignment = 4`` -- so ``num_heads = 4`` (``block_q = 32``) is
rejected outright; later builds additionally assert
``num_heads == 32 or num_heads == 64`` on SM90.  The qwen4 geometry cannot use
that kernel on H20/H100, so prefill scoring lives here instead: one K head
(MQA) scored against the 4 Q heads natively, no head padding, no DeepGEMM.

Contract -- mirrors ``fp8_mqa_indexer_score`` so it can drop into the same
call sites:

    q_fp8   [M, H, D] float8_e4m3fn   (H = 4 for this model, D = 128)
    weight  [M, H]    fp32            per-(token, head) weight.  qwen4 folds
                                      the q dequant scale into it and passes
                                      ``q_scale * (1 / sqrt(D))`` -- the
                                      equal-weight sum of DSv4's weighted form.
    k_fp8   [N, D]    float8_e4m3fn   pooled block keys
    k_scale [N]       fp32            per-entry dequant scale
    ks, ke  [M]       int32           K window ``[ks, ke)`` per query row

    -> logits [M, N] fp32, ``-inf`` outside each row's window.

Per row::

    logits[m, n] = sum_h weight[m, h] * relu(dot(q[m,h], k[n]) * k_scale[n])

Numerics: every ``float8_e4m3fn`` value is exactly representable in bf16, so
the kernel dequantizes by a plain cast and dots in bf16 with fp32 accumulation
-- products and sums are exact with respect to the quantized inputs (checked
bit-equal against an fp64 reference in the tests).

Window semantics: the kernel writes the **entire** ``[M, N]`` rectangle,
filling ``-inf`` outside the window.  DeepGEMM leaves those entries untouched
(``clean_logits=False``); filling is strictly safer for the topk consumer and
matches the reference implementation's ``-inf`` sentinel.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

_BLOCK_M = 64
_BLOCK_N = 64


@triton.jit
def _qsa_indexer_score_kernel(
    q_ptr,
    w_ptr,
    k_ptr,
    k_scale_ptr,
    ks_ptr,
    ke_ptr,
    out_ptr,
    M,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rd = tl.arange(0, D)
    m_mask = rm < M

    ks = tl.load(ks_ptr + rm, mask=m_mask, other=0)
    ke = tl.load(ke_ptr + rm, mask=m_mask, other=0)
    # Skip the head dots for tiles that touch no row's window; the store below
    # still runs, so those cells get their -inf.
    window_lo = tl.min(ks, axis=0)
    window_hi = tl.max(ke, axis=0)
    needs_compute = (pid_n * BN < window_hi) and (pid_n * BN + BN > window_lo)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if needs_compute:
        k_tile = tl.load(
            k_ptr + rn[None, :] * D + rd[:, None],
            mask=rn[None, :] < N,
            other=0.0,
        ).to(tl.bfloat16)
        k_scale = tl.load(k_scale_ptr + rn, mask=rn < N, other=0.0)
        for h in tl.static_range(H):
            q_tile = tl.load(
                q_ptr + (rm[:, None] * H + h) * D + rd[None, :],
                mask=m_mask[:, None],
                other=0.0,
            ).to(tl.bfloat16)
            head = tl.dot(q_tile, k_tile, out_dtype=tl.float32)
            head = tl.maximum(head * k_scale[None, :], 0.0)
            w_head = tl.load(w_ptr + rm * H + h, mask=m_mask, other=0.0)
            acc += head * w_head[:, None]

    valid = (rn[None, :] >= ks[:, None]) & (rn[None, :] < ke[:, None])
    tl.store(
        out_ptr + rm[:, None] * N + rn[None, :],
        tl.where(valid, acc, float("-inf")),
        mask=m_mask[:, None] & (rn[None, :] < N),
    )


def qsa_indexer_score(
    q_fp8: torch.Tensor,
    weight: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    *,
    block_m: int = _BLOCK_M,
    block_n: int = _BLOCK_N,
) -> torch.Tensor:
    """Score every query row against the pooled keys, ``[M, N]`` fp32.

    Args mirror the module docstring.  ``-inf`` marks positions outside a
    row's ``[ks, ke)`` window; callers feed the result straight to topk.
    """
    if not q_fp8.is_cuda:
        raise ValueError("qsa_indexer_score expects CUDA tensors")
    if q_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError(f"q_fp8 must be float8_e4m3fn, got {q_fp8.dtype}")
    if q_fp8.dim() != 3:
        raise ValueError(f"q_fp8 must be [M, H, D], got {tuple(q_fp8.shape)}")
    if k_fp8.dtype != torch.float8_e4m3fn or k_fp8.dim() != 2:
        raise ValueError(f"k_fp8 must be [N, D] float8_e4m3fn, got {k_fp8.dtype}")
    if k_fp8.shape[1] != q_fp8.shape[2] or k_fp8.shape[0] != k_scale.numel():
        raise ValueError(
            f"shape mismatch: q {tuple(q_fp8.shape)}, k {tuple(k_fp8.shape)}, "
            f"k_scale {tuple(k_scale.shape)}"
        )
    if weight.shape != q_fp8.shape[:2] or weight.dtype != torch.float32:
        raise ValueError(
            f"weight must be fp32 [M, H]={tuple(q_fp8.shape[:2])}, "
            f"got {weight.dtype} {tuple(weight.shape)}"
        )
    if k_scale.dtype != torch.float32:
        raise ValueError(f"k_scale must be fp32, got {k_scale.dtype}")
    M, N = q_fp8.shape[0], k_fp8.shape[0]
    for name, tensor in (("ks", ks), ("ke", ke)):
        if tensor.dtype != torch.int32 or tensor.shape != (M,):
            raise ValueError(
                f"{name} must be int32 [{M}], got {tensor.dtype} "
                f"{tuple(tensor.shape)}"
            )

    q_fp8 = q_fp8.contiguous()
    weight = weight.contiguous()
    k_fp8 = k_fp8.contiguous()
    k_scale = k_scale.contiguous()
    ks = ks.contiguous()
    ke = ke.contiguous()

    logits = torch.empty((M, N), device=q_fp8.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return logits

    H = q_fp8.shape[1]
    D = q_fp8.shape[2]
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _qsa_indexer_score_kernel[grid](
        q_fp8,
        weight,
        k_fp8,
        k_scale,
        ks,
        ke,
        logits,
        M,
        N,
        H=H,
        D=D,
        BM=block_m,
        BN=block_n,
    )
    return logits


def quantize_per_head(
    x: torch.Tensor, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp8-quantize a ``[..., D]`` tensor with a per-row absmax scale.

    Returns ``(x_fp8, scale)`` with ``x_fp8 * scale ~= x`` elementwise
    (``float8_e4m3fn`` max is 448).  One scale per leading row, so a
    ``[M, H, D]`` input yields ``[M, H]`` -- exactly the shape the score
    kernel folds into ``weight``.
    """
    flat = x.float()
    scale = flat.abs().amax(dim=-1, keepdim=True).clamp(min=eps) / 448.0
    return (flat / scale).to(torch.float8_e4m3fn), scale.squeeze(-1)
