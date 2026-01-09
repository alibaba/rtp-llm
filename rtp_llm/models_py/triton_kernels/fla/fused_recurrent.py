# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
from typing import Optional

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.fla.op import exp


# assume x always greater than 1
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
@triton.jit(do_not_specialize=["N", "T"])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    block_map,
    sequence_lengths,
    max_block_size: tl.int32,
    scale,
    N: tl.constexpr,  # num of sequences
    T: tl.constexpr,  # num of tokens
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_qb: tl.constexpr,  # q stride for batch/token dimension
    stride_qs: tl.constexpr,  # q stride for head dimension
    stride_qh: tl.constexpr,  # q stride for K dimension
    stride_kb: tl.constexpr,  # k stride for batch/token dimension
    stride_ks: tl.constexpr,  # k stride for head dimension
    stride_kh: tl.constexpr,  # k stride for K dimension
    stride_vb: tl.constexpr,  # v stride for batch/token dimension
    stride_vs: tl.constexpr,  # v stride for head dimension
    stride_vh: tl.constexpr,  # v stride for V dimension
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    SEQ_SIZE_PER_BLOCK: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if IS_CONTINUOUS_BATCHING:
        sequence_length = tl.load(sequence_lengths + i_n).to(tl.int64)
    else:
        # not used
        sequence_length = 0

    if T <= 0:
        # no tokens to process for this sequence
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_qs + i_h * stride_qh + o_k
    p_k = k + bos * stride_ks + i_h * stride_kh + o_k
    p_v = v + bos * stride_vs + i_hv * stride_vh + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
    p_g = g + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

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

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        # [BK, BV]
        b_h *= exp(b_g)
        # [BV]
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        # [BK, BV]
        b_h += b_k[:, None] * b_v[None, :]
        # [BV]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # keep the states for multi-query tokens
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

        p_q += stride_qs
        p_k += stride_ks
        p_o += HV * V
        p_v += stride_vs
        p_g += HV
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)


def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    block_map: Optional[torch.Tensor] = None,
    seq_size_per_block=1,
    sequence_lengths: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    o = q.new_empty(NK, *v.shape)
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(T, HV, K, V, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    # Get strides for q, k, v tensors to support non-contiguous tensors
    # Expected shape: [B, T, H, K/V]
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

    grid = (NK, NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
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
        BV=BV,
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
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=inplace_final_state,
        num_warps=num_warps,
        num_stages=num_stages,
        SEQ_SIZE_PER_BLOCK=seq_size_per_block,
    )
    o = o.squeeze(0)
    return o, final_state


class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        inplace_final_state: bool = True,
        cu_seqlens: Optional[torch.LongTensor] = None,
        block_map: Optional[torch.Tensor] = None,
        seq_size_per_block=1,
        sequence_lengths: Optional[torch.Tensor] = None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        o, final_state = fused_recurrent_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g.contiguous(),
            beta=beta.contiguous(),
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            block_map=block_map,
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=sequence_lengths,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

        return o, final_state


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    block_map: Optional[torch.Tensor] = None,
    seq_size_per_block=1,
    sequence_lengths: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA is applied if `HV > H`.
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV]`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        inplace_final_state: bool:
            Whether to store the final state in-place to save memory.
            Default: `True`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        ssm_state_indices (Optional[torch.Tensor]):
            Indices to map the input sequences to the initial/final states.
        num_accepted_tokens (Optional[torch.Tensor]):
            Number of accepted tokens for each sequence during decoding.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, K, V, device='cuda')
        >>> o, ht = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            cu_seqlens=cu_seqlens
        )
    """
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
            f"Please flatten variable-length inputs before processing."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    o, final_state = FusedRecurrentFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        inplace_final_state,
        cu_seqlens,
        block_map,
        seq_size_per_block,
        sequence_lengths,
        use_qk_l2norm_in_kernel,
    )
    return o, final_state
