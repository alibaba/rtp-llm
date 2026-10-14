"""Fused Xing4 hyper-connection post-combination Triton kernel (fusion K4).

Numerical authority (do not edit): ``Xing4HCUnit._post_impl`` in
``rtp_llm/models_py/model_desc/xing4_0_model.py`` (L160-170), the eager tail
of every Xing4 HC sublayer::

    first  = post.to(dt).unsqueeze(-1) * x.unsqueeze(-2)   # [T, hc, H] bf16
    second = torch.matmul(comb.to(dt), residual.to(dt))    # comb untransposed
    out    = (first + second).to(dt)

This module replaces that tail — broadcast mul + 4x4xH micro-GEMM bmm + add
plus up to three dtype casts (~6 eager launches) — with ONE Triton kernel::

    out[t, i, h] = bf16( bf16(post[t, i] * x[t, h])
                       + bf16(sum_j fp32(comb[t, i, j]) * fp32(residual[t, j, h])) )

``comb`` is applied UNTRANSPOSED: ``second[t, i, h] = sum_j comb[t, i, j] *
residual[t, j, h]`` (``torch.matmul(comb, residual)`` semantics). Swapping the
comb indices would silently exchange the sinkhorn row/col normalization.

Rounding structure — kept identical to the eager ops so the fused result is
bit-exact against them whenever the cuBLAS accumulation itself is exact
(zero/identity comb, exact ties; the UT gates those bit-exact), and within
one bf16 ulp of the accumulated stream magnitude otherwise; the UT
(test_hc_post_triton.py) gates the sinkhorn-domain cases with rtol=2**-7
plus that ulp-scaled atol (see its docstring for the full scheme):

* ``first`` — the eager bf16 broadcast mul runs in fp32 opmath and rounds the
  product to bf16; the kernel multiplies in fp32 and rounds to bf16 (a product
  of two bf16 values is exact in fp32, so this leg is bit-exact).
* ``second`` — the eager bf16 bmm (cuBLAS) accumulates in fp32 and rounds each
  completed element to bf16 once; the kernel accumulates the hc products in
  fp32 in ascending-j order — the same sequential order as the K=4 GEMM — and
  rounds once after the full sum. Only a cuBLAS reordering of the 4-term fp32
  sum can differ, bounded by one bf16 ulp after rounding.
* ``out`` — the eager bf16 add runs in fp32 opmath and rounds to bf16; the
  kernel adds the two bf16-rounded values in fp32 and rounds to bf16.

Triton pitfall (verified against the generated PTX on triton 3.6/SM103): a
literal ``x.to(tl.bfloat16).to(tl.float32)`` round-trip is canonicalized away
(``extf(truncf(x)) -> x``) and the mul+add tail then contracts into a single
``fma`` — silently dropping BOTH intermediate bf16 roundings above. The
roundings are therefore forced through an opaque int-bitcast
round-to-nearest-even (the c10::BFloat16 algorithm) in ``_rtne_round_bf16``.

Kernel rules honored: elementwise/reduction fusion only (no tl.dot, no new
GEMM/attention core — the hc_fn ``F.linear`` GEMM is untouched); grid=(T,)
with an in-kernel H block loop; ``pid`` and every T-dependent offset are
computed in int64 because out/residual hold T*hc*H elements and large decode
batches push ``pid * stride`` past INT32_MAX (same pitfall as
``modules/dsv4/_fused_rmsnorm_rope_triton.py``).

``out`` may alias ``residual``: each program fully reads its own
``residual[t, :, h_block]`` slice before storing ``out[t, :, h_block]`` (the
h-block loop reads every j row of the current block before the block store),
and no other program touches that slice.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_DEFAULT_BLOCK_H = 512
_DEFAULT_NUM_WARPS = 4


@triton.jit
def _rtne_round_bf16(x):
    """RTNE-round an fp32 tile to bf16-representable values, kept as fp32.

    ``x.to(tl.bfloat16).to(tl.float32)`` cannot be used here: Triton 3.6
    canonicalizes the ``extf(truncf(x))`` round-trip back to ``x`` (see the
    module docstring), which deletes exactly the intermediate rounding this
    kernel must reproduce. The bit-manipulation form below is the standard
    fp32->bf16 round-to-nearest-even (same algorithm as c10::BFloat16):
    add half an ulp (plus the target LSB, for ties-to-even) and clear the low
    16 bits. NaN payloads are not preserved; this path never sees NaN.
    """
    u = x.to(tl.int32, bitcast=True)
    u = u + 0x7FFF + ((u >> 16) & 1)
    u = u & -65536  # 0xFFFF0000: sign+exponent+top-7 mantissa bits of bf16
    return u.to(tl.float32, bitcast=True)


@triton.jit
def _hc_post_kernel(
    post_ptr,  # [T, HC] bf16, already cast to the stream dtype
    comb_ptr,  # [T, HC, HC] bf16, applied UNTRANSPOSED
    x_ptr,  # [T, H] bf16 sublayer output
    residual_ptr,  # [T, HC, H] bf16 streams
    out_ptr,  # [T, HC, H] bf16
    H,
    stride_post_t,
    stride_comb_t,
    stride_comb_i,
    stride_x_t,
    stride_res_t,
    stride_res_i,
    stride_out_t,
    stride_out_i,
    BLOCK_H: tl.constexpr,
    HC: tl.constexpr,
):
    # int64 BEFORE the row-stride multiply: the flattened T*HC*H element
    # offset must not silently overflow int32 for large decode batches (same
    # pitfall as _fused_rmsnorm_rope_kernel in modules/dsv4).
    pid_t = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1)
    offs_i = tl.arange(0, HC)
    post = tl.load(post_ptr + pid_t * stride_post_t + offs_i).to(tl.float32)

    # 2D grid (T, cdiv(H, BLOCK_H)): each program owns one H block, so the
    # elementwise math is identical per element (no cross-block reduction) —
    # bitwise unchanged vs the serial in-program loop, while decode (T=1)
    # gains cdiv(H, BLOCK_H) CTAs instead of one and prefill fills the GPU.
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < H
    x = tl.load(
        x_ptr + pid_t * stride_x_t + offs_h, mask=h_mask, other=0.0
    ).to(tl.float32)
    # first: fp32 product rounded to bf16 (eager broadcast-mul opmath).
    first = _rtne_round_bf16(post[:, None] * x[None, :])

    # second: fp32 accumulation over j in ascending order, rounded to
    # bf16 once per element after the full sum (cuBLAS bf16 bmm structure).
    acc = tl.zeros((HC, BLOCK_H), dtype=tl.float32)
    for j in tl.static_range(HC):
        comb_j = tl.load(
            comb_ptr + pid_t * stride_comb_t + offs_i * stride_comb_i + j
        ).to(tl.float32)
        res_j = tl.load(
            residual_ptr + pid_t * stride_res_t + j * stride_res_i + offs_h,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        acc += comb_j[:, None] * res_j[None, :]
    second = _rtne_round_bf16(acc)

    # out: fp32 add of the two bf16-rounded values, rounded to bf16.
    out = (first + second).to(tl.bfloat16)
    out_ptrs = (
        out_ptr
        + pid_t * stride_out_t
        + offs_i[:, None] * stride_out_i
        + offs_h[None, :]
    )
    tl.store(out_ptrs, out, mask=h_mask[None, :])


def hc_post_triton(
    post: torch.Tensor,
    comb: torch.Tensor,
    x: torch.Tensor,
    residual: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    block_h: int = _DEFAULT_BLOCK_H,
) -> torch.Tensor:
    """Fused Xing4 HC post combination (see module docstring for the math).

    Args:
        post: ``[..., HC]`` or ``[..., HC, 1]`` bf16 — already cast to the
            stream dtype by the caller (reference domain: (0, 2)).
        comb: ``[..., HC, HC]`` bf16, applied UNTRANSPOSED.
        x: ``[..., H]`` bf16 sublayer output.
        residual: ``[..., HC, H]`` bf16 streams.
        out: optional preallocated ``[..., HC, H]`` bf16 output; may alias
            ``residual`` (see module docstring).
        block_h: H chunk size of the in-kernel loop (power of two).

    Returns:
        ``out`` (freshly allocated when ``out is None``), same shape and dtype
        as ``residual``.
    """
    if x.dtype != torch.bfloat16:
        raise ValueError(f"x (sublayer output) must be bfloat16, got {x.dtype}")
    if residual.dtype != torch.bfloat16:
        raise ValueError(f"residual must be bfloat16, got {residual.dtype}")
    if post.dtype != torch.bfloat16 or comb.dtype != torch.bfloat16:
        raise ValueError(
            "post/comb must already be cast to bfloat16: the reference rounds "
            "post*x in bf16, so casting inside the kernel would diverge; use "
            "xing4_hc_post() for not-yet-cast tensors"
        )
    if post.dim() == residual.dim() and post.shape[-1] == 1:
        post = post.squeeze(-1)

    hc = residual.shape[-2]
    H = residual.shape[-1]
    if x.shape[:-1] != residual.shape[:-2] or x.shape[-1] != H:
        raise ValueError(
            f"x shape {tuple(x.shape)} incompatible with residual shape "
            f"{tuple(residual.shape)}"
        )
    if post.shape != (*x.shape[:-1], hc):
        raise ValueError(f"post shape {tuple(post.shape)} != {(*x.shape[:-1], hc)}")
    if comb.shape != (*x.shape[:-1], hc, hc):
        raise ValueError(
            f"comb shape {tuple(comb.shape)} != {(*x.shape[:-1], hc, hc)}"
        )
    if hc < 1 or (hc & (hc - 1)) != 0:
        raise ValueError(f"hc must be a power of two for tl.arange, got {hc}")
    if block_h < 1 or (block_h & (block_h - 1)) != 0:
        raise ValueError(f"block_h must be a power of two, got {block_h}")
    if not x.is_cuda:
        raise ValueError("hc_post_triton requires CUDA tensors")
    for name, tensor in (
        ("post", post),
        ("comb", comb),
        ("x", x),
        ("residual", residual),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous, strides={tensor.stride()}")
        if tensor.device != x.device:
            raise ValueError(f"{name} on {tensor.device}, expected {x.device}")

    if out is None:
        out = torch.empty_like(residual)
    else:
        if out.shape != residual.shape:
            raise ValueError(
                f"out shape {tuple(out.shape)} != {tuple(residual.shape)}"
            )
        if out.dtype != torch.bfloat16:
            raise ValueError(f"out must be bfloat16, got {out.dtype}")
        if out.device != x.device or not out.is_contiguous():
            raise ValueError("out must be contiguous and on the same CUDA device")

    T = residual.numel() // (hc * H)
    if T == 0:
        return out

    post_f = post.view(T, hc)
    comb_f = comb.view(T, hc, hc)
    x_f = x.view(T, H)
    res_f = residual.view(T, hc, H)
    out_f = out.view(T, hc, H)

    import triton as _triton

    _hc_post_kernel[(T, _triton.cdiv(H, block_h))](
        post_f,
        comb_f,
        x_f,
        res_f,
        out_f,
        H,
        post_f.stride(0),
        comb_f.stride(0),
        comb_f.stride(1),
        x_f.stride(0),
        res_f.stride(0),
        res_f.stride(1),
        out_f.stride(0),
        out_f.stride(1),
        BLOCK_H=block_h,
        HC=hc,
        num_warps=_DEFAULT_NUM_WARPS,
    )
    return out


def xing4_hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Drop-in fused replacement for ``Xing4HCUnit._post_impl``.

    Mirrors the reference argument order ``(x, residual, post, comb)`` and its
    dtype casts: post/comb/residual are cast to ``x.dtype`` exactly like the
    eager path (the casts are no-ops for tensors that already carry the stream
    dtype and device, as they do straight out of ``_pre_impl``). Accepts the
    flat ``[T, ...]`` layout as well as batched leading dims (``[B, S, ...]``),
    which are flattened for the grid=(T,) launch.
    """
    post_b = post.squeeze(-1) if post.dim() == residual.dim() else post
    return hc_post_triton(
        post_b.to(device=x.device, dtype=x.dtype),
        comb.to(device=x.device, dtype=x.dtype),
        x,
        residual.to(device=x.device, dtype=x.dtype),
        out=out,
    )
