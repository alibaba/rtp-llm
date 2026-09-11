# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk_fwd.py
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li


import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.fla.index import prepare_chunk_indices
from rtp_llm.models_py.triton_kernels.fla.op import exp2
from rtp_llm.models_py.triton_kernels.fla.wy_fast import recompute_w_u_fwd


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kkt_solve_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    k += (bos * Hg + i_h // (H // Hg)) * K
    A += (bos * H + i_h) * BT

    o_i = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    p_b0_i0 = tl.arange(0, BC).to(tl.int64) + (i_tc0)
    p_b0_m0 = (p_b0_i0 >= 0) & (p_b0_i0 < (T))
    p_b0 = (beta + bos * H + i_h) + p_b0_i0 * (H)
    p_b1_i0 = tl.arange(0, BC).to(tl.int64) + (i_tc1)
    p_b1_m0 = (p_b1_i0 >= 0) & (p_b1_i0 < (T))
    p_b1 = (beta + bos * H + i_h) + p_b1_i0 * (H)
    p_b2_i0 = tl.arange(0, BC).to(tl.int64) + (i_tc2)
    p_b2_m0 = (p_b2_i0 >= 0) & (p_b2_i0 < (T))
    p_b2 = (beta + bos * H + i_h) + p_b2_i0 * (H)
    p_b3_i0 = tl.arange(0, BC).to(tl.int64) + (i_tc3)
    p_b3_m0 = (p_b3_i0 >= 0) & (p_b3_i0 < (T))
    p_b3 = (beta + bos * H + i_h) + p_b3_i0 * (H)
    b_b0 = tl.load(p_b0, mask=p_b0_m0, other=0).to(tl.float32)
    b_b1 = tl.load(p_b1, mask=p_b1_m0, other=0).to(tl.float32)
    b_b2 = tl.load(p_b2, mask=p_b2_m0, other=0).to(tl.float32)
    b_b3 = tl.load(p_b3, mask=p_b3_m0, other=0).to(tl.float32)

    if USE_G:
        p_g0_i0 = tl.arange(0, BC).to(tl.int64) + (i_tc0)
        p_g0_m0 = (p_g0_i0 >= 0) & (p_g0_i0 < (T))
        p_g0 = (g + bos * H + i_h) + p_g0_i0 * (H)
        p_g1_i0 = tl.arange(0, BC).to(tl.int64) + (i_tc1)
        p_g1_m0 = (p_g1_i0 >= 0) & (p_g1_i0 < (T))
        p_g1 = (g + bos * H + i_h) + p_g1_i0 * (H)
        p_g2_i0 = tl.arange(0, BC).to(tl.int64) + (i_tc2)
        p_g2_m0 = (p_g2_i0 >= 0) & (p_g2_i0 < (T))
        p_g2 = (g + bos * H + i_h) + p_g2_i0 * (H)
        p_g3_i0 = tl.arange(0, BC).to(tl.int64) + (i_tc3)
        p_g3_m0 = (p_g3_i0 >= 0) & (p_g3_i0 < (T))
        p_g3 = (g + bos * H + i_h) + p_g3_i0 * (H)

        b_g0 = tl.load(p_g0, mask=p_g0_m0, other=0).to(tl.float32)
        b_g1 = tl.load(p_g1, mask=p_g1_m0, other=0).to(tl.float32)
        b_g2 = tl.load(p_g2, mask=p_g2_m0, other=0).to(tl.float32)
        b_g3 = tl.load(p_g3, mask=p_g3_m0, other=0).to(tl.float32)

    # Step 1: compute all 10 lower-triangular [BC, BC] blocks of K @ K^T
    b_A00 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A11 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A22 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A33 = tl.zeros([BC, BC], dtype=tl.float32)

    b_A10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k0_i0 = tl.arange(0, BC).to(tl.int64) + (i_tc0)
        p_k0_m0 = (p_k0_i0 >= 0) & (p_k0_i0 < (T))
        p_k0_i1 = tl.arange(0, BK).to(tl.int64) + (i_k * BK)
        p_k0_m1 = (p_k0_i1 >= 0) & (p_k0_i1 < (K))
        p_k0 = (k) + p_k0_i0[:, None] * (Hg * K) + p_k0_i1[None, :] * (1)
        b_k0 = tl.load(p_k0, mask=p_k0_m0[:, None] & p_k0_m1[None, :], other=0)
        b_A00 += tl.dot(b_k0, tl.trans(b_k0))

        if i_tc1 < T:
            p_k1_i0 = tl.arange(0, BC).to(tl.int64) + (i_tc1)
            p_k1_m0 = (p_k1_i0 >= 0) & (p_k1_i0 < (T))
            p_k1_i1 = tl.arange(0, BK).to(tl.int64) + (i_k * BK)
            p_k1_m1 = (p_k1_i1 >= 0) & (p_k1_i1 < (K))
            p_k1 = (k) + p_k1_i0[:, None] * (Hg * K) + p_k1_i1[None, :] * (1)
            b_k1 = tl.load(p_k1, mask=p_k1_m0[:, None] & p_k1_m1[None, :], other=0)
            b_A11 += tl.dot(b_k1, tl.trans(b_k1))
            b_A10 += tl.dot(b_k1, tl.trans(b_k0))

            if i_tc2 < T:
                p_k2_i0 = tl.arange(0, BC).to(tl.int64) + (i_tc2)
                p_k2_m0 = (p_k2_i0 >= 0) & (p_k2_i0 < (T))
                p_k2_i1 = tl.arange(0, BK).to(tl.int64) + (i_k * BK)
                p_k2_m1 = (p_k2_i1 >= 0) & (p_k2_i1 < (K))
                p_k2 = (k) + p_k2_i0[:, None] * (Hg * K) + p_k2_i1[None, :] * (1)
                b_k2 = tl.load(p_k2, mask=p_k2_m0[:, None] & p_k2_m1[None, :], other=0)
                b_A22 += tl.dot(b_k2, tl.trans(b_k2))
                b_A20 += tl.dot(b_k2, tl.trans(b_k0))
                b_A21 += tl.dot(b_k2, tl.trans(b_k1))

                if i_tc3 < T:
                    p_k3_i0 = tl.arange(0, BC).to(tl.int64) + (i_tc3)
                    p_k3_m0 = (p_k3_i0 >= 0) & (p_k3_i0 < (T))
                    p_k3_i1 = tl.arange(0, BK).to(tl.int64) + (i_k * BK)
                    p_k3_m1 = (p_k3_i1 >= 0) & (p_k3_i1 < (K))
                    p_k3 = (k) + p_k3_i0[:, None] * (Hg * K) + p_k3_i1[None, :] * (1)
                    b_k3 = tl.load(
                        p_k3, mask=p_k3_m0[:, None] & p_k3_m1[None, :], other=0
                    )
                    b_A33 += tl.dot(b_k3, tl.trans(b_k3))
                    b_A30 += tl.dot(b_k3, tl.trans(b_k0))
                    b_A31 += tl.dot(b_k3, tl.trans(b_k1))
                    b_A32 += tl.dot(b_k3, tl.trans(b_k2))

    # Step 2: apply gate and beta scaling
    m_d = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    if USE_G:
        b_A00 *= tl.where(
            m_d & m_tc0[:, None] & m_tc0[None, :],
            exp2(b_g0[:, None] - b_g0[None, :]),
            0.0,
        )
        b_A11 *= tl.where(
            m_d & m_tc1[:, None] & m_tc1[None, :],
            exp2(b_g1[:, None] - b_g1[None, :]),
            0.0,
        )
        b_A22 *= tl.where(
            m_d & m_tc2[:, None] & m_tc2[None, :],
            exp2(b_g2[:, None] - b_g2[None, :]),
            0.0,
        )
        b_A33 *= tl.where(
            m_d & m_tc3[:, None] & m_tc3[None, :],
            exp2(b_g3[:, None] - b_g3[None, :]),
            0.0,
        )

        b_A10 *= tl.where(
            m_tc1[:, None] & m_tc0[None, :],
            exp2(b_g1[:, None] - b_g0[None, :]),
            0.0,
        )
        b_A20 *= tl.where(
            m_tc2[:, None] & m_tc0[None, :],
            exp2(b_g2[:, None] - b_g0[None, :]),
            0.0,
        )
        b_A21 *= tl.where(
            m_tc2[:, None] & m_tc1[None, :],
            exp2(b_g2[:, None] - b_g1[None, :]),
            0.0,
        )
        b_A30 *= tl.where(
            m_tc3[:, None] & m_tc0[None, :],
            exp2(b_g3[:, None] - b_g0[None, :]),
            0.0,
        )
        b_A31 *= tl.where(
            m_tc3[:, None] & m_tc1[None, :],
            exp2(b_g3[:, None] - b_g1[None, :]),
            0.0,
        )
        b_A32 *= tl.where(
            m_tc3[:, None] & m_tc2[None, :],
            exp2(b_g3[:, None] - b_g2[None, :]),
            0.0,
        )
    else:
        b_A00 = tl.where(m_d, b_A00, 0.0)
        b_A11 = tl.where(m_d, b_A11, 0.0)
        b_A22 = tl.where(m_d, b_A22, 0.0)
        b_A33 = tl.where(m_d, b_A33, 0.0)

    b_A00 = b_A00 * b_b0[:, None]
    b_A11 = b_A11 * b_b1[:, None]
    b_A22 = b_A22 * b_b2[:, None]
    b_A33 = b_A33 * b_b3[:, None]

    b_A10 = b_A10 * b_b1[:, None]
    b_A20 = b_A20 * b_b2[:, None]
    b_A21 = b_A21 * b_b2[:, None]
    b_A30 = b_A30 * b_b3[:, None]
    b_A31 = b_A31 * b_b3[:, None]
    b_A32 = b_A32 * b_b3[:, None]

    # Step 3: forward substitution on diagonal blocks -> (I + A_diag)^{-1}
    b_Ai00 = -b_A00
    b_Ai11 = -b_A11
    b_Ai22 = -b_A22
    b_Ai33 = -b_A33

    # Loop to BC (compile-time constant); out-of-bounds rows are zeroed by m_tc* masks
    # applied in Step 2, so the forward-substitution is safe for T%BT != 0.
    for i in range(2, BC):
        b_a00 = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.0), 0)
        b_a00 = tl.where(o_i < i, b_a00, 0.0)
        b_a00 = b_a00 + tl.sum(b_a00[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
    for i in range(2, BC):
        b_a11 = tl.sum(tl.where((o_i == i)[:, None], -b_A11, 0.0), 0)
        b_a11 = tl.where(o_i < i, b_a11, 0.0)
        b_a11 = b_a11 + tl.sum(b_a11[:, None] * b_Ai11, 0)
        b_Ai11 = tl.where((o_i == i)[:, None], b_a11, b_Ai11)
    for i in range(2, BC):
        b_a22 = tl.sum(tl.where((o_i == i)[:, None], -b_A22, 0.0), 0)
        b_a22 = tl.where(o_i < i, b_a22, 0.0)
        b_a22 = b_a22 + tl.sum(b_a22[:, None] * b_Ai22, 0)
        b_Ai22 = tl.where((o_i == i)[:, None], b_a22, b_Ai22)
    for i in range(2, BC):
        b_a33 = tl.sum(tl.where((o_i == i)[:, None], -b_A33, 0.0), 0)
        b_a33 = tl.where(o_i < i, b_a33, 0.0)
        b_a33 = b_a33 + tl.sum(b_a33[:, None] * b_Ai33, 0)
        b_Ai33 = tl.where((o_i == i)[:, None], b_a33, b_Ai33)

    b_Ai00 += m_I
    b_Ai11 += m_I
    b_Ai22 += m_I
    b_Ai33 += m_I

    # Step 4: block merge -> full (I + A)^{-1}
    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_A10, input_precision="ieee"), b_Ai00, input_precision="ieee"
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_A21, input_precision="ieee"), b_Ai11, input_precision="ieee"
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_A32, input_precision="ieee"), b_Ai22, input_precision="ieee"
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_A20, b_Ai00, input_precision="ieee")
        + tl.dot(b_A21, b_Ai10, input_precision="ieee"),
        input_precision="ieee",
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_A31, b_Ai11, input_precision="ieee")
        + tl.dot(b_A32, b_Ai21, input_precision="ieee"),
        input_precision="ieee",
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_A30, b_Ai00, input_precision="ieee")
        + tl.dot(b_A31, b_Ai10, input_precision="ieee")
        + tl.dot(b_A32, b_Ai20, input_precision="ieee"),
        input_precision="ieee",
    )

    # Step 5: store full (I + A)^{-1} to output A
    p_A00_i0 = tl.arange(0, BC).to(tl.int64) + (i_tc0)
    p_A00_m0 = (p_A00_i0 >= 0) & (p_A00_i0 < (T))
    p_A00_i1 = tl.arange(0, BC).to(tl.int64) + (0)
    p_A00_m1 = (p_A00_i1 >= 0) & (p_A00_i1 < (BT))
    p_A00 = (A) + p_A00_i0[:, None] * (H * BT) + p_A00_i1[None, :] * (1)
    p_A10_i0 = tl.arange(0, BC).to(tl.int64) + (i_tc1)
    p_A10_m0 = (p_A10_i0 >= 0) & (p_A10_i0 < (T))
    p_A10_i1 = tl.arange(0, BC).to(tl.int64) + (0)
    p_A10_m1 = (p_A10_i1 >= 0) & (p_A10_i1 < (BT))
    p_A10 = (A) + p_A10_i0[:, None] * (H * BT) + p_A10_i1[None, :] * (1)
    p_A11_i0 = tl.arange(0, BC).to(tl.int64) + (i_tc1)
    p_A11_m0 = (p_A11_i0 >= 0) & (p_A11_i0 < (T))
    p_A11_i1 = tl.arange(0, BC).to(tl.int64) + (BC)
    p_A11_m1 = (p_A11_i1 >= 0) & (p_A11_i1 < (BT))
    p_A11 = (A) + p_A11_i0[:, None] * (H * BT) + p_A11_i1[None, :] * (1)
    p_A20_i0 = tl.arange(0, BC).to(tl.int64) + (i_tc2)
    p_A20_m0 = (p_A20_i0 >= 0) & (p_A20_i0 < (T))
    p_A20_i1 = tl.arange(0, BC).to(tl.int64) + (0)
    p_A20_m1 = (p_A20_i1 >= 0) & (p_A20_i1 < (BT))
    p_A20 = (A) + p_A20_i0[:, None] * (H * BT) + p_A20_i1[None, :] * (1)
    p_A21_i0 = tl.arange(0, BC).to(tl.int64) + (i_tc2)
    p_A21_m0 = (p_A21_i0 >= 0) & (p_A21_i0 < (T))
    p_A21_i1 = tl.arange(0, BC).to(tl.int64) + (BC)
    p_A21_m1 = (p_A21_i1 >= 0) & (p_A21_i1 < (BT))
    p_A21 = (A) + p_A21_i0[:, None] * (H * BT) + p_A21_i1[None, :] * (1)
    p_A22_i0 = tl.arange(0, BC).to(tl.int64) + (i_tc2)
    p_A22_m0 = (p_A22_i0 >= 0) & (p_A22_i0 < (T))
    p_A22_i1 = tl.arange(0, BC).to(tl.int64) + (2 * BC)
    p_A22_m1 = (p_A22_i1 >= 0) & (p_A22_i1 < (BT))
    p_A22 = (A) + p_A22_i0[:, None] * (H * BT) + p_A22_i1[None, :] * (1)
    p_A30_i0 = tl.arange(0, BC).to(tl.int64) + (i_tc3)
    p_A30_m0 = (p_A30_i0 >= 0) & (p_A30_i0 < (T))
    p_A30_i1 = tl.arange(0, BC).to(tl.int64) + (0)
    p_A30_m1 = (p_A30_i1 >= 0) & (p_A30_i1 < (BT))
    p_A30 = (A) + p_A30_i0[:, None] * (H * BT) + p_A30_i1[None, :] * (1)
    p_A31_i0 = tl.arange(0, BC).to(tl.int64) + (i_tc3)
    p_A31_m0 = (p_A31_i0 >= 0) & (p_A31_i0 < (T))
    p_A31_i1 = tl.arange(0, BC).to(tl.int64) + (BC)
    p_A31_m1 = (p_A31_i1 >= 0) & (p_A31_i1 < (BT))
    p_A31 = (A) + p_A31_i0[:, None] * (H * BT) + p_A31_i1[None, :] * (1)
    p_A32_i0 = tl.arange(0, BC).to(tl.int64) + (i_tc3)
    p_A32_m0 = (p_A32_i0 >= 0) & (p_A32_i0 < (T))
    p_A32_i1 = tl.arange(0, BC).to(tl.int64) + (2 * BC)
    p_A32_m1 = (p_A32_i1 >= 0) & (p_A32_i1 < (BT))
    p_A32 = (A) + p_A32_i0[:, None] * (H * BT) + p_A32_i1[None, :] * (1)
    p_A33_i0 = tl.arange(0, BC).to(tl.int64) + (i_tc3)
    p_A33_m0 = (p_A33_i0 >= 0) & (p_A33_i0 < (T))
    p_A33_i1 = tl.arange(0, BC).to(tl.int64) + (3 * BC)
    p_A33_m1 = (p_A33_i1 >= 0) & (p_A33_i1 < (BT))
    p_A33 = (A) + p_A33_i0[:, None] * (H * BT) + p_A33_i1[None, :] * (1)

    tl.store(
        p_A00, b_Ai00.to(A.dtype.element_ty), mask=p_A00_m0[:, None] & p_A00_m1[None, :]
    )
    tl.store(
        p_A10, b_Ai10.to(A.dtype.element_ty), mask=p_A10_m0[:, None] & p_A10_m1[None, :]
    )
    tl.store(
        p_A11, b_Ai11.to(A.dtype.element_ty), mask=p_A11_m0[:, None] & p_A11_m1[None, :]
    )
    tl.store(
        p_A20, b_Ai20.to(A.dtype.element_ty), mask=p_A20_m0[:, None] & p_A20_m1[None, :]
    )
    tl.store(
        p_A21, b_Ai21.to(A.dtype.element_ty), mask=p_A21_m0[:, None] & p_A21_m1[None, :]
    )
    tl.store(
        p_A22, b_Ai22.to(A.dtype.element_ty), mask=p_A22_m0[:, None] & p_A22_m1[None, :]
    )
    tl.store(
        p_A30, b_Ai30.to(A.dtype.element_ty), mask=p_A30_m0[:, None] & p_A30_m1[None, :]
    )
    tl.store(
        p_A31, b_Ai31.to(A.dtype.element_ty), mask=p_A31_m0[:, None] & p_A31_m1[None, :]
    )
    tl.store(
        p_A32, b_Ai32.to(A.dtype.element_ty), mask=p_A32_m0[:, None] & p_A32_m1[None, :]
    )
    tl.store(
        p_A33, b_Ai33.to(A.dtype.element_ty), mask=p_A33_m0[:, None] & p_A33_m1[None, :]
    )


def chunk_gated_delta_rule_fwd_intra_a_only(
    k: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, Hg, K = k.shape
    H = beta.shape[2]
    BT = chunk_size
    BC = 16

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    A = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    chunk_gated_delta_rule_fwd_kkt_solve_kernel[(NT, B * H)](
        k=k,
        g=g,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        BC=BC,
        BK=64,
        # num_warps=1 is optimal on AMD CDNA: this kernel is memory-bound with
        # 10 small BC×BC dot products per tile; extra warps increase VGPR pressure
        # without improving occupancy. Verified via rocprof PMC on MI355X.
        num_warps=1,
        num_stages=1,
    )
    return A


def chunk_gated_delta_rule_fwd_intra(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    A = chunk_gated_delta_rule_fwd_intra_a_only(
        k=k,
        beta=beta,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
    )
    return w, u, A
