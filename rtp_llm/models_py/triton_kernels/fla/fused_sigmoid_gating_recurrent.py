# Adapted from sglang: fused_sigmoid_gating_delta_rule_update_kernel
# This kernel fuses the sigmoid gating computation (previously in gdn_gating.py)
# with the recurrent delta rule update (previously in fused_recurrent.py) into
# a single Triton kernel, eliminating one kernel launch and intermediate tensors.
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.fla.op import exp
from rtp_llm.models_py.triton_kernels.fla.utils import input_guard


def _is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def _get_autotune_configs():
    if _is_cuda():
        return [
            triton.Config({"BV": 8}, num_stages=3, num_warps=1),
        ]
    else:
        # ROCm/HIP: larger BV and different launch params for better occupancy
        return [
            triton.Config({"BV": 64}, num_stages=1, num_warps=4),
        ]


@triton.jit
def cal_block_idx(x, seq_size_per_block):
    return (x - 1) // seq_size_per_block


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["block_map"] is not None,
    }
)
@triton.autotune(
    configs=_get_autotune_configs(),
    key=["K", "V"],
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0,
    ht,
    cu_seqlens,
    block_map,
    sequence_lengths,
    max_block_size: tl.int32,
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
    stride_qb: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    SEQ_SIZE_PER_BLOCK: tl.constexpr,
):
    """
    Fused kernel combining sigmoid gating computation with recurrent delta rule update.
    Replaces the two-step approach of fused_gdn_gating + fused_recurrent_gated_delta_rule.

    Gating computation (previously separate):
        g = -exp(A_log) * softplus(a + dt_bias)
        beta = sigmoid(b)

    Recurrent delta rule update:
        h *= exp(g)
        v -= sum(h * k, dim=0)
        v *= beta
        h += k[:, None] * v[None, :]
        o = sum(h * q, dim=0)
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if IS_CONTINUOUS_BATCHING:
        sequence_length = tl.load(sequence_lengths + i_n).to(tl.int64)
    else:
        sequence_length = 0

    if T <= 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_qs + i_h * stride_qh + o_k
    p_k = k + bos * stride_ks + i_h * stride_kh + o_k
    p_v = v + bos * stride_vs + i_hv * stride_vh + o_v
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating parameter pointers
    p_A_log = A_log + i_hv
    p_a = a + bos * HV + i_hv
    p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            load_block_offset = cal_block_idx(sequence_length - 1, SEQ_SIZE_PER_BLOCK)
            read_block_id = tl.load(
                block_map + i_n * max_block_size + load_block_offset
            ).to(tl.int64)
            if read_block_id <= 0:
                return
            p_h0 = h0 + read_block_id * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * K * V
        p_h0 = p_h0 + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Pre-load constant gating parameters
    b_A_log = tl.load(p_A_log).to(tl.float32)
    b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

    for i_t in range(0, T):
        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)
        b_a = tl.load(p_a).to(tl.float32)

        # === Fused gating computation (replaces fused_gdn_gating) ===
        # g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x

        # beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # === Recurrent delta rule update ===
        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        b_h *= exp(b_g)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Store intermediate states for continuous batching
        if INPLACE_FINAL_STATE and IS_CONTINUOUS_BATCHING:
            write_block_offset = (
                cal_block_idx(sequence_length, SEQ_SIZE_PER_BLOCK) + i_t
            )
            write_block_id = tl.load(
                block_map + i_n * max_block_size + write_block_offset
            ).to(tl.int64)
            p_ht = ht + write_block_id * stride_final_state_token
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
        p_ht = p_ht + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        # Advance pointers for next timestep
        p_q += stride_qs
        p_k += stride_ks
        p_o += HV * V
        p_v += stride_vs
        p_b += HV
        p_a += HV


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    inplace_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    block_map: Optional[torch.Tensor] = None,
    seq_size_per_block: int = 1,
    sequence_lengths: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused implementation combining sigmoid gating and recurrent delta rule update.

    This replaces the two-step approach:
        g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
        o, ht = fused_recurrent_gated_delta_rule(q, k, v, g, beta, ...)

    With a single fused kernel that computes gating inline during the recurrent loop,
    eliminating one kernel launch and the intermediate g/beta tensors.

    Args:
        A_log: Log of decay parameter, shape [HV].
        a: Gating input, shape [B*T, HV] or [B, T, HV].
        dt_bias: Bias for gating, shape [HV].
        q: Queries, shape [B, T, H, K].
        k: Keys, shape [B, T, H, K].
        v: Values, shape [B, T, HV, V].
        b: Beta input (for sigmoid), shape [B*T, HV] or [B, T, HV].
        scale: Scale factor. Defaults to 1/sqrt(K).
        initial_state: Initial hidden state, shape depends on block_map usage.
        inplace_final_state: Whether to store final state in-place.
        cu_seqlens: Cumulative sequence lengths for variable-length inputs.
        block_map: Block mapping for continuous batching.
        seq_size_per_block: Tokens per block for continuous batching.
        sequence_lengths: Sequence lengths for continuous batching.
        use_qk_l2norm_in_kernel: Whether to apply L2 norm to Q and K.
        softplus_beta: Beta parameter for softplus.
        softplus_threshold: Threshold for softplus numerical stability.

    Returns:
        Tuple of (output, final_state).
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    NK = triton.cdiv(K, BK)
    assert NK == 1, "NK > 1 is not supported yet"

    if scale is None:
        scale = k.shape[-1] ** -0.5

    # Reshape b and a to match expected layout: [B*T, HV]
    b_flat = b.reshape(B * T, HV)
    a_flat = a.reshape(B * T, HV)

    o = q.new_empty(NK, *v.shape)
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(T, HV, K, V, dtype=initial_state.dtype)

    stride_init_state_token = (
        initial_state.stride(0) if initial_state is not None else 0
    )
    stride_final_state_token = final_state.stride(0) if final_state is not None else 0

    stride_qb, stride_qs, stride_qh = q.stride(0), q.stride(1), q.stride(2)
    stride_kb, stride_ks, stride_kh = k.stride(0), k.stride(1), k.stride(2)
    stride_vb, stride_vs, stride_vh = v.stride(0), v.stride(1), v.stride(2)
    assert (
        q.stride(3) == 1 and k.stride(3) == 1 and v.stride(3) == 1
    ), "stride_qd, stride_kd, stride_vd must be 1"

    max_block_size = 0
    if block_map is not None:
        assert block_map.ndim == 2, "block_map must be a 2D tensor"
        max_block_size = block_map.shape[1]

    grid = lambda META: (NK, triton.cdiv(V, META["BV"]), N * HV)

    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a_flat,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b_flat,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        block_map=block_map,
        sequence_lengths=sequence_lengths,
        max_block_size=max_block_size,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        stride_qb=stride_qb,
        stride_qs=stride_qs,
        stride_qh=stride_qh,
        stride_kb=stride_kb,
        stride_ks=stride_ks,
        stride_kh=stride_kh,
        stride_vb=stride_vb,
        stride_vs=stride_vs,
        stride_vh=stride_vh,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=inplace_final_state,
        SEQ_SIZE_PER_BLOCK=seq_size_per_block,
    )
    o = o.squeeze(0)
    return o, final_state
