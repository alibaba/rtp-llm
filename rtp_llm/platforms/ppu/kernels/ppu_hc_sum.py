"""Ordered HC residual update, optionally fusing the BF16 MoE output sum."""

import torch
import triton
import triton.language as tl


@triton.jit
def _hc_post_sum(
    R,
    S,
    RES,
    POST,
    COMB,
    OUT,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    FUSE_SUM: tl.constexpr = True,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    output_lane = tl.arange(0, 4)
    routed = tl.load(R + row * D + col, col < D, 0).to(tl.float32)
    if FUSE_SUM:
        shared = tl.load(S + row * D + col, col < D, 0).to(tl.float32)
        # The original MoE epilogue stores BF16 before HC consumes it.
        summed = (routed + shared).to(tl.bfloat16).to(tl.float32)
    else:
        summed = routed
    post = tl.load(POST + row * 4 + output_lane)
    first = tl.load(RES + row * 4 * D + col, col < D, 0).to(tl.float32)
    mix = tl.load(COMB + row * 16 + output_lane)
    # Match TileLang's first contraction: post * summed + round(mix0 * res0).
    # Reversing the FMA operands changes BF16 bits near cancellation.
    acc = tl.fma(post[:, None], summed[None, :], mix[:, None] * first[None, :])
    for lane in tl.static_range(1, 4):
        residual = tl.load(RES + row * 4 * D + lane * D + col, col < D, 0).to(
            tl.float32
        )
        mix = tl.load(COMB + row * 16 + lane * 4 + output_lane)
        acc = tl.fma(mix[:, None], residual[None, :], acc)
    tl.store(
        OUT + row * 4 * D + output_lane[:, None] * D + col[None, :],
        acc,
        col[None, :] < D,
    )


def supports_hc_sum(x, residual, post, comb):
    if residual.ndim not in (3, 4) or residual.shape[-2:] != (4, 4096):
        return False
    leading = residual.shape[:-2]
    return (
        x.shape == (*leading, 4096)
        and post.shape == (*leading, 4, 1)
        and comb.shape == (*leading, 4, 4)
        and x.is_cuda
        and x.dtype == residual.dtype == torch.bfloat16
        and post.dtype == comb.dtype == torch.float32
        and all(
            t.device == x.device and t.is_contiguous()
            for t in (x, residual, post, comb)
        )
        and 0 < x.numel() // 4096 <= 512
    )


def supports_hc_post(x, residual, post, comb, out=None):
    # Limit selection to the measured decode/MTP row counts. Larger batches
    # retain TileLang, which has better Graph latency in that regime.
    return (
        supports_hc_sum(x, residual, post, comb)
        and x.numel() // 4096 <= 32
        and (
            out is None
            or (
                out.shape == residual.shape
                and out.dtype == residual.dtype
                and out.device == residual.device
                and out.is_contiguous()
            )
        )
    )


def hc_post(x, residual, post, comb, *, out=None):
    if not supports_hc_post(x, residual, post, comb, out):
        raise ValueError("PPU HC POST requires contiguous small-batch Flash inputs")
    rows = x.numel() // 4096
    block = 128 if rows <= 8 else 256
    out = torch.empty_like(residual) if out is None else out
    # A CTA reads all four lanes before writing its disjoint column tile;
    # out=residual therefore preserves HC.post's in-place contract.
    _hc_post_sum[(rows, triton.cdiv(4096, block))](
        x,
        x,
        residual,
        post,
        comb,
        out,
        4096,
        block,
        FUSE_SUM=False,
        num_warps=4,
        enable_fp_fusion=True,
    )
    return out


def hc_post_sum(routed, shared, residual, post, comb):
    if (
        not supports_hc_sum(routed, residual, post, comb)
        or shared.shape != routed.shape
        or shared.dtype != routed.dtype
        or shared.device != routed.device
        or not shared.is_contiguous()
    ):
        raise ValueError(
            "PPU HC sum requires contiguous BF16 Flash inputs and FP32 mixers"
        )
    rows = routed.numel() // 4096
    block = 128 if rows <= 8 else (256 if rows <= 32 else 512)
    # Each CTA reads all four residual lanes for its disjoint column tile
    # before storing. Preserve HC.post's existing in-place buffer contract.
    out = residual
    _hc_post_sum[(rows, triton.cdiv(4096, block))](
        routed,
        shared,
        residual,
        post,
        comb,
        out,
        4096,
        block,
        num_warps=4,
        enable_fp_fusion=True,
    )
    return out
