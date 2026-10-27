# Decode-path fusion of the Gated DeltaNet in_proj_ba GEMV with fused_gdn_gating.
#
# in_proj_ba is a tiny bf16 GEMV (N = 2 * local_v_heads, K = hidden) that cuBLAS runs at
# ~6% of DRAM bandwidth, followed by a separate gating kernel. This kernel computes
#   b, a = (x @ W_ba^T) rounded to bf16 (same as the bf16 linear output)
#   g    = -exp(A_log) * softplus(a + dt_bias)   (fp32)
#   beta = sigmoid(b)                            (b dtype)
# in one launch. One program per (token, head); K is reduced in BLOCK_K chunks.

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_ba_gating_kernel(
    x_ptr,
    w_ptr,  # [2 * NUM_HEADS, K] row-major: rows [0, H) -> b, rows [H, 2H) -> a
    A_log_ptr,
    dt_bias_ptr,
    g_ptr,
    beta_ptr,
    K,
    stride_x,
    NUM_HEADS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    softplus_beta: tl.constexpr,
    threshold: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_h = tl.program_id(1)
    offs = tl.arange(0, BLOCK_K)
    acc_b = tl.zeros([BLOCK_K], dtype=tl.float32)
    acc_a = tl.zeros([BLOCK_K], dtype=tl.float32)
    x_row = x_ptr + i_t * stride_x
    wb_row = w_ptr + i_h * K
    wa_row = w_ptr + (NUM_HEADS + i_h) * K
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs
        m = k < K
        xv = tl.load(x_row + k, mask=m, other=0.0).to(tl.float32)
        acc_b += xv * tl.load(wb_row + k, mask=m, other=0.0).to(tl.float32)
        acc_a += xv * tl.load(wa_row + k, mask=m, other=0.0).to(tl.float32)
    # match the bf16 linear output rounding before gating
    b = tl.sum(acc_b, axis=0).to(tl.bfloat16).to(tl.float32)
    a = tl.sum(acc_a, axis=0).to(tl.bfloat16).to(tl.float32)
    xg = a + tl.load(dt_bias_ptr + i_h).to(tl.float32)
    softplus_x = tl.where(
        softplus_beta * xg <= threshold,
        (1 / softplus_beta) * tl.log(1 + tl.exp(softplus_beta * xg)),
        xg,
    )
    g = -tl.exp(tl.load(A_log_ptr + i_h).to(tl.float32)) * softplus_x
    tl.store(g_ptr + i_t * NUM_HEADS + i_h, g)
    tl.store(
        beta_ptr + i_t * NUM_HEADS + i_h, tl.sigmoid(b).to(beta_ptr.dtype.element_ty)
    )


def fused_ba_gdn_gating(
    hidden_states: torch.Tensor,
    w_ba_t: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_heads: int,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (g [1, T, H] fp32, beta [1, T, H] hidden dtype), equal to
    fused_gdn_gating(A_log, a, b, dt_bias) with b, a = split(hidden @ W_ba)."""
    assert hidden_states.dim() == 2 and hidden_states.stride(1) == 1
    T, K = hidden_states.shape
    assert w_ba_t.shape == (2 * num_heads, K) and w_ba_t.is_contiguous()
    g = torch.empty(1, T, num_heads, dtype=torch.float32, device=hidden_states.device)
    beta_out = torch.empty(
        1, T, num_heads, dtype=hidden_states.dtype, device=hidden_states.device
    )
    if T == 0:
        return g, beta_out
    _fused_ba_gating_kernel[(T, num_heads)](
        hidden_states,
        w_ba_t,
        A_log,
        dt_bias,
        g,
        beta_out,
        K,
        hidden_states.stride(0),
        NUM_HEADS=num_heads,
        BLOCK_K=1024,
        softplus_beta=beta,
        threshold=threshold,
        num_warps=4,
    )
    return g, beta_out
