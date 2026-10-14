"""Xing4.0 hyper-connection mix/sinkhorn Triton kernel (HC fusion lane K2).

The math authority is ``Xing4HCUnit._pre_impl`` in
``rtp_llm/models_py/model_desc/xing4_0_model.py`` (L130-158): the
``F.linear`` GEMM produces ``mixes[T, 24]`` fp32, and everything downstream
of it — the sigmoid pre/post gates and the clamp + exp + 20-round
(row, col) sinkhorn comb — is a chain of ~130 tiny elementwise/reduction
launches per sublayer forward.  This module fuses that whole per-token
chain into ONE Triton launch (grid=T, one program per token; hc=4 keeps
the 4x4 sinkhorn entirely in registers), leaving the GEMM itself untouched
(its >60% device-time share must stay out of the fusion numerator).

Xing4-exact contract (the differences vs the DeepSeek-V4 vendored mhc
kernels are deliberate — see xing4_0_model.py):

  * ``pre  = sigmoid(pre_w * pre_scale + pre_b)`` — no eps term
    (xing4_0_model.py:134);
  * ``post = 2 * sigmoid(post_w * post_scale + post_b)``;
  * comb logits are clamped to ``[clamp_min, clamp_max]`` (default ±30,
    ``mhc_h_res_clamp``) AFTER scale+bias and BEFORE the row max
    (xing4_0_model.py:137-141);
  * ``comb = exp(logits - rowmax)`` then ``sinkhorn_iters`` rounds of
    row (dim=-1) then col (dim=-2) normalization, with eps added ONLY to
    the denominators (xing4_0_model.py:141-144);
  * all math runs in fp32 registers; the single bf16 cast happens at the
    store, at the same point the reference casts ``pre/post/comb``
    (xing4_0_model.py:158).

``sinkhorn_iters``/``eps``/``clamp_min``/``clamp_max`` stay runtime kernel
arguments (no constexpr specialization fan-out).  Indexing discipline
follows ``_fused_rmsnorm_rope_triton.py``: the token program id is cast to
int64 BEFORE the row-stride multiply.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_HC = 4
_MIX_WIDTH = _HC * (_HC + 2)  # 24 = pre(4) + post(4) + comb(16)

_OUT_TL_DTYPE = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}


@triton.jit
def _xing4_hc_mix_kernel(
    mixes_ptr,  # fp32 [T, 24], row-major contiguous
    base_ptr,  # fp32 [24]
    scale_ptr,  # fp32 [3]
    pre_ptr,  # out [T, 4]
    post_ptr,  # out [T, 4]
    comb_ptr,  # out [T, 4, 4]
    sinkhorn_iters,  # runtime i32 loop bound (not constexpr)
    eps,  # runtime f32, added to row/col denominators only
    clamp_min,  # runtime f32 (mhc_h_res_clamp_min)
    clamp_max,  # runtime f32 (mhc_h_res_clamp_max)
    OUT_DTYPE: tl.constexpr,
):
    # int64 BEFORE the row-stride multiply — same pitfall class as
    # _fused_rmsnorm_rope_triton.py (pid * stride silently overflowing
    # int32 on large flattened token counts).
    pid = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, 4)
    c = tl.arange(0, 4)

    mixes_row = mixes_ptr + pid * 24
    # .to(tl.float32) is a no-op for fp32 mixes and a lossless widen for the
    # bf16 GEMM output: the model feeds the hc_fn F.linear output directly
    # (no separate widen launch), keeping the math bitwise identical to the
    # eager path's .float()-then-compute.
    pre_w = tl.load(mixes_row + r).to(tl.float32)
    post_w = tl.load(mixes_row + 4 + r).to(tl.float32)
    comb_w = tl.load(mixes_row + 8 + r[:, None] * 4 + c[None, :]).to(tl.float32)

    pre_b = tl.load(base_ptr + r)
    post_b = tl.load(base_ptr + 4 + r)
    comb_b = tl.load(base_ptr + 8 + r[:, None] * 4 + c[None, :])
    pre_s = tl.load(scale_ptr)
    post_s = tl.load(scale_ptr + 1)
    comb_s = tl.load(scale_ptr + 2)

    # Gates: sigmoid(pre_w * pre_scale + pre_b) with no eps (xing4_0_model.py:134).
    pre = tl.sigmoid(pre_w * pre_s + pre_b)
    post = 2.0 * tl.sigmoid(post_w * post_s + post_b)

    # comb: scale+bias -> clamp -> exp(x - rowmax) -> (row, col) sinkhorn.
    logits = comb_w * comb_s + comb_b
    logits = tl.minimum(tl.maximum(logits, clamp_min), clamp_max)
    row_max = tl.max(logits, axis=1)  # amax(dim=-1)
    comb = tl.exp(logits - row_max[:, None])

    for _ in range(sinkhorn_iters):
        row_sum = tl.sum(comb, axis=1)  # dim=-1 first
        comb = comb / (row_sum[:, None] + eps)
        col_sum = tl.sum(comb, axis=0)  # then dim=-2
        comb = comb / (col_sum[None, :] + eps)

    # Single cast point, matching the reference's pre/post/comb .to(dtype).
    tl.store(pre_ptr + pid * 4 + r, pre.to(OUT_DTYPE))
    tl.store(post_ptr + pid * 4 + r, post.to(OUT_DTYPE))
    tl.store(comb_ptr + pid * 16 + r[:, None] * 4 + c[None, :], comb.to(OUT_DTYPE))


def hc_mix_triton(
    mixes: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    *,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
    clamp_min: float = -30.0,
    clamp_max: float = 30.0,
    out_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused per-token HC mixer: ``mixes[T, 24]`` fp32 -> (pre, post, comb).

    Mirrors ``Xing4HCUnit._pre_impl`` (xing4_0_model.py:130-158) strictly
    downstream of the ``F.linear`` GEMM:

      * ``pre``  [T, 4]  = ``sigmoid(pre_w * pre_scale + pre_b)``
      * ``post`` [T, 4]  = ``2 * sigmoid(post_w * post_scale + post_b)``
      * ``comb`` [T, 4, 4] = clamp + exp(x - rowmax) + ``sinkhorn_iters``
        rounds of (row +eps, col +eps) normalization

    All math is fp32; ``out_dtype`` only selects the single exit cast
    (bf16 for the model path, fp32 for equivalence debugging).  The caller
    owns the ``post.unsqueeze(-1)`` reshaping done by ``_pre_impl``.

    Args:
        mixes: fp32 ``[T, 24]`` CUDA tensor (contiguous or not; a
            non-contiguous input is copied once).
        base: fp32 ``[24]`` CUDA tensor (``hc`` pre/post/comb biases).
        scale: fp32 ``[3]`` CUDA tensor (pre/post/comb scales).
        sinkhorn_iters: rounds of (row, col) normalization (20 in config).
        eps: denominator epsilon (``hc_eps``, 1e-6 in config).
        clamp_min/clamp_max: comb logit clamp (``mhc_h_res_clamp``, ±30).
        out_dtype: torch.bfloat16 / torch.float16 / torch.float32.

    Returns:
        (pre [T, 4], post [T, 4], comb [T, 4, 4]) tensors in ``out_dtype``.
    """
    if out_dtype not in _OUT_TL_DTYPE:
        raise ValueError(
            f"unsupported out_dtype {out_dtype}; expected one of "
            f"{sorted(str(d) for d in _OUT_TL_DTYPE)}"
        )
    assert mixes.is_cuda and base.is_cuda and scale.is_cuda, "hc_mix_triton is CUDA-only"
    assert mixes.dim() == 2 and mixes.shape[-1] == _MIX_WIDTH, (
        f"mixes must be [T, {_MIX_WIDTH}], got {tuple(mixes.shape)}"
    )
    assert base.shape == (_MIX_WIDTH,), f"base must be [{_MIX_WIDTH}], got {tuple(base.shape)}"
    assert scale.shape == (3,), f"scale must be [3], got {tuple(scale.shape)}"
    assert mixes.dtype in (torch.float32, torch.bfloat16), (
        "mixes must be fp32 or the bf16 hc_fn GEMM output (widened losslessly in-kernel)"
    )
    assert base.dtype == torch.float32 and scale.dtype == torch.float32, (
        "base/scale must be fp32"
    )
    iters = int(sinkhorn_iters)
    assert iters >= 0, f"sinkhorn_iters must be >= 0, got {sinkhorn_iters}"

    mixes = mixes.contiguous()
    base = base.contiguous()
    scale = scale.contiguous()

    T = mixes.shape[0]
    pre = torch.empty((T, _HC), dtype=out_dtype, device=mixes.device)
    post = torch.empty((T, _HC), dtype=out_dtype, device=mixes.device)
    comb = torch.empty((T, _HC, _HC), dtype=out_dtype, device=mixes.device)
    if T == 0:
        return pre, post, comb

    _xing4_hc_mix_kernel[(T,)](
        mixes,
        base,
        scale,
        pre,
        post,
        comb,
        iters,
        eps,
        clamp_min,
        clamp_max,
        OUT_DTYPE=_OUT_TL_DTYPE[out_dtype],
        num_warps=1,
    )
    return pre, post, comb
