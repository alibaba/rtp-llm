"""Lightweight kernel to recompute only the final hidden state via recurrent
delta rule, without allocating the full output tensor.

This is a workaround for the chunk_gated_delta_rule Triton kernel producing
incorrect final_state on ROCm due to tl.dot precision issues. The recurrent
approach processes token-by-token in fp32 and has been verified correct on ROCm.

Memory usage: O(N * HV * K * V) for final_state only, vs O(B * T * HV * V)
for the full fused_recurrent output. For a 17684-token sequence this saves ~17 GiB.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.fla.op import exp


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def recurrent_final_state_kernel(
    k,
    v,
    g,
    beta,
    h0,
    ht,
    cu_seqlens,
    scale,
    N: tl.constexpr,
    T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vh: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Compute only the final hidden state h_T via the recurrent delta rule.

    Same math as fused_recurrent_gated_delta_rule_fwd_kernel but skips:
    - Loading q (not needed for state update)
    - Computing output o = sum(h * q)
    - Storing per-token output or intermediate states

    Only the final state after processing all T tokens is written to ht.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_n * T
        T = T

    if T <= 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_k = k + bos * stride_ks + i_h * stride_kh + o_k
    p_v = v + bos * stride_vs + i_hv * stride_vh + o_v
    p_beta = beta + bos * HV + i_hv
    p_g = g + bos * HV + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_n * HV * K * V + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)

        # h *= exp(g)
        b_h *= exp(b_g)
        # v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        # v *= beta
        b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        # h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        p_k += stride_ks
        p_v += stride_vs
        p_g += HV
        p_beta += HV

    # Store only the final state
    p_ht = ht + i_n * HV * K * V + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def recompute_final_state(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> torch.Tensor:
    """Recompute final hidden state using token-by-token recurrence in fp32.

    This is mathematically equivalent to running fused_recurrent_gated_delta_rule
    over the full sequence, but only returns the final state without allocating
    the full output tensor. Memory usage is O(N * HV * K * V) instead of
    O(B * T * HV * V).

    Args:
        k: Keys, shape [B, T, H, K].
        v: Values, shape [B, T, HV, V].
        g: Gating (log-space decay), shape [B, T, HV] or [B*T, HV].
        beta: Beta gating, shape [B, T, HV] or [B*T, HV].
        scale: Scale factor (unused for state computation, kept for API compat).
        initial_state: Initial hidden state [N, HV, K, V] or None.
        cu_seqlens: Cumulative sequence lengths [N+1] for varlen.
        use_qk_l2norm_in_kernel: Whether to L2-normalize k in kernel.

    Returns:
        final_state: [N, HV, K, V] in fp32.
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 8)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"

    if scale is None:
        scale = K ** -0.5

    final_state = k.new_empty(N, HV, K, V, dtype=torch.float32)

    # Reshape g and beta to [B*T, HV] if needed
    if g.ndim == 3 and g.shape[0] == B and g.shape[1] == T:
        g = g.reshape(B * T, HV)
    if beta.ndim == 3 and beta.shape[0] == B and beta.shape[1] == T:
        beta = beta.reshape(B * T, HV)

    stride_ks, stride_kh = k.stride(1), k.stride(2)
    stride_vs, stride_vh = v.stride(1), v.stride(2)

    grid = (NK, NV, N * HV)
    recurrent_final_state_kernel[grid](
        k=k,
        v=v,
        g=g,
        beta=beta,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_ks=stride_ks,
        stride_kh=stride_kh,
        stride_vs=stride_vs,
        stride_vh=stride_vh,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=1,
        num_stages=3,
    )
    return final_state
