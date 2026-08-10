# Fused recurrent KDA kernel for rtp-llm decode.
# Combines KDA's gate logic (USE_GATE_IN_KERNEL, softplus, A_log, dt_bias)
# with rtp-llm's block_map-based continuous batching for ssm_state indexing.

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.fla.op import exp

logger = logging.getLogger(__name__)


@triton.jit
def softplus(x):
    return tl.where(x < 20.0, tl.math.log(1 + tl.math.exp(x)), x)


@triton.jit
def cal_block_idx(x, seq_size_per_block):
    return (x - 1) // seq_size_per_block


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["block_map"] is not None,
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_recurrent_kda_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    cu_seqlens,
    block_map,
    sequence_lengths,
    max_block_size: tl.int32,
    lower_bound,
    scale: tl.constexpr,
    N: tl.int64,
    T: tl.int64,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    SEQ_SIZE_PER_BLOCK: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
    ALLOW_NEG_EIGVAL: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    NK = tl.cdiv(K, BK)
    i_k = pid % NK
    pid_rest = pid // NK
    i_v = pid_rest % NV
    i_nh = pid_rest // NV
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    if IS_CONTINUOUS_BATCHING:
        sequence_length = tl.load(sequence_lengths + i_n).to(tl.int64)
    else:
        sequence_length = 0

    if T <= 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
    # KDA: g is per-dim [B, T, HV, K]
    p_g = g + (bos * HV + i_hv) * K + o_k
    p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    # Match FLA's transpose_state_layout=True arithmetic: keep the register
    # tile [V,K], while translating addresses to RTP's canonical [K,V] cache.
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            load_block_offset = cal_block_idx(sequence_length - 1, SEQ_SIZE_PER_BLOCK)
            read_block_id = tl.load(
                block_map + i_n * max_block_size + load_block_offset
            ).to(tl.int64)
            if read_block_id <= 0:
                return
            p_h0 = h0 + read_block_id * stride_init_state_token + i_hv * K * V
        else:
            p_h0 = h0 + (i_n * HV + i_hv) * K * V
        if STATE_V_FIRST:
            p_h0 = p_h0 + o_v[:, None] * K + o_k[None, :]
        else:
            p_h0 = p_h0 + o_k[None, :] * V + o_v[:, None]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in tl.range(0, T, num_stages=num_stages):
        b_q = tl.load(p_q, mask=mask_k, other=0, eviction_policy="evict_last").to(
            tl.float32
        )
        b_k = tl.load(p_k, mask=mask_k, other=0, eviction_policy="evict_last").to(
            tl.float32
        )
        b_v = tl.load(p_v, mask=mask_v, other=0, eviction_policy="evict_first").to(
            tl.float32
        )

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        # Load raw gate and apply gate activation if needed.
        b_g = tl.load(p_g, mask=mask_k, other=0, eviction_policy="evict_last").to(
            tl.float32
        )
        if USE_GATE_IN_KERNEL:
            b_A = tl.load(A_log + i_hv).to(tl.float32)
            if HAS_DT_BIAS:
                b_bias = tl.load(
                    dt_bias + i_hv * K + o_k,
                    mask=mask_k,
                    other=0,
                ).to(tl.float32)
                b_g = b_g + b_bias
            if USE_LOWER_BOUND:
                b_gk = lower_bound * tl.sigmoid(exp(b_A) * b_g)
            else:
                b_gk = -exp(b_A) * softplus(b_g)
        else:
            b_gk = b_g

        b_h *= exp(b_gk[None, :])

        b_v -= tl.sum(b_h * b_k[None, :], 1)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(
                p_beta,
                mask=mask_v,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            b_beta = tl.load(p_beta, eviction_policy="evict_last").to(tl.float32)
        if APPLY_BETA_SIGMOID:
            b_beta = tl.sigmoid(b_beta)
            if ALLOW_NEG_EIGVAL:
                b_beta = b_beta * 2
        b_v *= b_beta

        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(
            p_o,
            b_o.to(p_o.dtype.element_ty),
            mask=mask_v,
            eviction_policy="evict_first",
        )

        if IS_CONTINUOUS_BATCHING:
            if INPLACE_FINAL_STATE:
                write_block_offset = (
                    cal_block_idx(sequence_length, SEQ_SIZE_PER_BLOCK) + i_t
                )
                write_block_id = tl.load(
                    block_map + i_n * max_block_size + write_block_offset
                ).to(tl.int64)
                p_ht = ht + write_block_id * stride_final_state_token
            else:
                p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = p_ht + i_hv * K * V
            if STATE_V_FIRST:
                p_ht = p_ht + o_v[:, None] * K + o_k[None, :]
            else:
                p_ht = p_ht + o_k[None, :] * V + o_v[:, None]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_g += HV * K
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)

    if not IS_CONTINUOUS_BATCHING:
        p_ht = ht + (i_n * HV + i_hv) * K * V
        if STATE_V_FIRST:
            p_ht = p_ht + o_v[:, None] * K + o_k[None, :]
        else:
            p_ht = p_ht + o_k[None, :] * V + o_v[:, None]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    inplace_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    block_map: Optional[torch.Tensor] = None,
    seq_size_per_block: int = 1,
    sequence_lengths: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    lower_bound: Optional[float] = None,
    state_v_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), 32
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"

    o = torch.zeros_like(v)
    if inplace_final_state:
        assert initial_state is not None
        final_state = initial_state
    else:
        state_shape = (N, HV, V, K) if state_v_first else (N, HV, K, V)
        final_state = q.new_empty(*state_shape, dtype=torch.float32)

    stride_init_state_token = (
        initial_state.stride(0) if initial_state is not None else 1
    )
    stride_final_state_token = final_state.stride(0) if final_state is not None else 1

    max_block_size = 0
    if block_map is not None:
        assert block_map.ndim == 2, "block_map must be a 2D tensor"
        max_block_size = block_map.shape[1]

    grid = (NK * NV * N * HV,)
    fused_recurrent_kda_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        block_map=block_map,
        sequence_lengths=sequence_lengths,
        max_block_size=max_block_size,
        lower_bound=lower_bound,
        scale=scale,
        N=N,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=inplace_final_state,
        SEQ_SIZE_PER_BLOCK=seq_size_per_block,
        USE_GATE_IN_KERNEL=use_gate_in_kernel,
        APPLY_BETA_SIGMOID=use_beta_sigmoid_in_kernel,
        ALLOW_NEG_EIGVAL=allow_neg_eigval,
        STATE_V_FIRST=state_v_first,
        num_warps=4,
        num_stages=2,
    )
    return o, final_state


def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    inplace_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    block_map: Optional[torch.Tensor] = None,
    seq_size_per_block: int = 1,
    sequence_lengths: Optional[torch.Tensor] = None,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    lower_bound: Optional[float] = None,
    state_v_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    o, final_state = fused_recurrent_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=initial_state,
        A_log=A_log,
        dt_bias=dt_bias,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens,
        block_map=block_map,
        seq_size_per_block=seq_size_per_block,
        sequence_lengths=sequence_lengths,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        lower_bound=lower_bound,
        state_v_first=state_v_first,
    )
    return o, final_state
