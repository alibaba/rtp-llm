"""Fused hc_split_sinkhorn (P-sinkhorn of plan_0427.md).

Replaces the chain at mhc.py:hc_split_sinkhorn:

    pre_raw  = mixes[..., :hc] * hc_scale[0] + hc_base[:hc]    # 2 launches
    pre      = pre_raw.sigmoid() + eps                          # 2 launches
    post_raw = mixes[..., hc:2*hc] * hc_scale[1] + hc_base[hc:2*hc]   # 2
    post     = 2.0 * post_raw.sigmoid()                         # 2
    comb_raw = mixes[..., 2*hc:] * hc_scale[2] + hc_base[2*hc:]      # 2
    comb     = comb_raw.softmax(-1) + eps                       # ~3
    comb     = comb / (comb.sum(-2, keepdim) + eps)             # 2
    for _ in range(sinkhorn_iters - 1):                         # 19 × (2+2) = 76
        comb = comb / (comb.sum(-1, keepdim) + eps)
        comb = comb / (comb.sum(-2, keepdim) + eps)

with one Triton kernel per token (~135 launches → 1).  HC is small (=4
in V4), so the entire [HC, HC] comb matrix lives in registers across
all 20 Sinkhorn iterations.

Numerics: 20 iterations of normalize-by-sum is mathematically convergent
to a doubly-stochastic matrix; small fp32 round-off in early iters gets
washed out.  Microbench shows max abs diff vs eager < 5e-7.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _hc_split_sinkhorn_kernel(
    mixes_ptr,       # [N, MIX_HC] fp32, where MIX_HC = (HC+2)*HC
    scale_ptr,       # [3]        fp32
    base_ptr,        # [MIX_HC]   fp32
    pre_ptr,         # [N, HC]    fp32 (out)
    post_ptr,        # [N, HC]    fp32 (out)
    comb_ptr,        # [N, HC*HC] fp32 (out)
    N,
    HC: tl.constexpr,        # = 4 in V4
    SINKHORN_ITERS: tl.constexpr,  # = 20 in V4
    EPS: tl.constexpr,
):
    """One program per token.  Computes pre, post, comb for that token.

    Constants (V4-Flash defaults):
      HC = 4           → mix_hc = 24, comb is [4, 4]
      SINKHORN_ITERS = 20
    """
    pid = tl.program_id(0)
    if pid >= N:
        return

    # tl.arange needs constexpr args, and constexpr×constexpr propagates only
    # via inlining — bind HC2 / MIX with tl.constexpr() to keep the type.
    HC2: tl.constexpr = HC * HC          # 16
    MIX: tl.constexpr = (HC + 2) * HC    # 24

    # === pre ===
    pre_offs = tl.arange(0, HC)
    s0 = tl.load(scale_ptr + 0)
    b_pre = tl.load(base_ptr + pre_offs)
    m_pre = tl.load(mixes_ptr + pid * MIX + pre_offs)
    pre = tl.sigmoid(m_pre * s0 + b_pre) + EPS
    tl.store(pre_ptr + pid * HC + pre_offs, pre)

    # === post ===
    s1 = tl.load(scale_ptr + 1)
    b_post = tl.load(base_ptr + HC + pre_offs)
    m_post = tl.load(mixes_ptr + pid * MIX + HC + pre_offs)
    post = 2.0 * tl.sigmoid(m_post * s1 + b_post)
    tl.store(post_ptr + pid * HC + pre_offs, post)

    # === comb (raw [HC, HC] starts at offset 2*HC of mixes) ===
    s2 = tl.load(scale_ptr + 2)
    comb_offs = tl.arange(0, HC2)
    b_comb = tl.load(base_ptr + 2 * HC + comb_offs)
    m_comb = tl.load(mixes_ptr + pid * MIX + 2 * HC + comb_offs)
    comb = m_comb * s2 + b_comb         # [HC*HC] fp32, viewed as [HC, HC] row-major

    # ---- softmax along the LAST dim (= columns of [HC, HC]) ----
    # Row index of each element in the [HC, HC] view: i = idx // HC; j = idx % HC.
    # Use a 2-D layout for clarity.  HC and HC2 are tl.constexpr so reshape is fine.
    comb2 = tl.reshape(comb, [HC, HC])
    row_max = tl.max(comb2, axis=1, keep_dims=True)        # [HC, 1]
    e = tl.exp(comb2 - row_max)
    row_sum = tl.sum(e, axis=1, keep_dims=True)             # [HC, 1]
    comb2 = e / row_sum + EPS                               # [HC, HC] post-softmax + eps

    # ---- step B: column normalize ----
    col_sum = tl.sum(comb2, axis=0, keep_dims=True) + EPS   # [1, HC]
    comb2 = comb2 / col_sum

    # ---- (sinkhorn_iters - 1) more alternating row/col normalizations ----
    for _ in tl.static_range(SINKHORN_ITERS - 1):
        row_sum = tl.sum(comb2, axis=1, keep_dims=True) + EPS  # [HC, 1]
        comb2 = comb2 / row_sum
        col_sum = tl.sum(comb2, axis=0, keep_dims=True) + EPS  # [1, HC]
        comb2 = comb2 / col_sum

    comb_out = tl.reshape(comb2, [HC2])
    tl.store(comb_ptr + pid * HC2 + comb_offs, comb_out)


def fused_hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Fused replacement for mhc.py:hc_split_sinkhorn.

    Args:
      mixes:    [..., (hc+2)*hc] fp32
      hc_scale: [3] fp32
      hc_base:  [(hc+2)*hc] fp32

    Returns:
      pre:  [..., hc]    fp32
      post: [..., hc]    fp32
      comb: [..., hc, hc] fp32
    """
    assert mixes.dtype == torch.float32 and mixes.is_contiguous()
    assert hc_scale.dtype == torch.float32 and hc_scale.numel() == 3
    HC = hc_mult
    MIX = (HC + 2) * HC
    assert mixes.size(-1) == MIX
    assert hc_base.dtype == torch.float32 and hc_base.numel() == MIX
    *batch, _ = mixes.size()
    N = 1
    for b in batch:
        N *= b

    mixes_flat = mixes.view(N, MIX)
    pre = torch.empty((N, HC), dtype=torch.float32, device=mixes.device)
    post = torch.empty((N, HC), dtype=torch.float32, device=mixes.device)
    comb = torch.empty((N, HC * HC), dtype=torch.float32, device=mixes.device)
    if N == 0:
        return (pre.view(*batch, HC),
                post.view(*batch, HC),
                comb.view(*batch, HC, HC))

    grid = (N,)
    _hc_split_sinkhorn_kernel[grid](
        mixes_flat, hc_scale, hc_base,
        pre, post, comb,
        N=N,
        HC=HC,
        SINKHORN_ITERS=int(sinkhorn_iters),
        EPS=float(eps),
        num_warps=1,    # tiny per-program work
        num_stages=1,
    )
    return (pre.view(*batch, HC),
            post.view(*batch, HC),
            comb.view(*batch, HC, HC))
