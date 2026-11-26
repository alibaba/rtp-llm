# SPDX-License-Identifier: Apache-2.0
"""SM90 fused MoE router (BF16 Gate-GEMV/WGMMA + sqrtsoftplus + TopK + FP8 Pack).

Replaces the 3-kernel decode router sequence on Hopper (L20Z / SM90):
  1. cuBLASLt BF16 split-K GEMM  (``F.linear(x, gate._weight_bf16())``)
  2. cuBLASLt split-K reduction
  3. Triton ``_v4_gate_sqrtsoftplus_topk_kernel(HAS_PACK=True)``

with a self-contained header-free NVRTC SM90 CUDA suite specialized for the
DeepSeek-V4.1 decode geometries (``H = 5120``, ``(E, K) in {(384, 6), (128, 3)}``,
``1 <= M <= 128``):
  - Small decode (``M <= 4`` for ``E=384``, ``M <= 8`` for ``E=128``):
    single-launch zero-SMEM K-permuted direct-register ``mma.sync.aligned.m16n8k16``
    with concurrent FP8 K128 pack CTAs, lock-free ``atom.global.acq_rel.gpu``
    last-CTA arrival (zero spin-wait, deadlock-free with concurrent shared-expert
    stream, self-resetting flag for CUDA Graph replay), and ``redux.sync.max.u32``
    collaborative warp TopK.
  - Mid decode (``5 <= M <= 32`` for ``E=384``, ``9 <= M <= 32`` for ``E=128``):
    2-kernel zero-SMEM ``m16n8k16`` split-K GEMM followed by zero-SMEM
    collaborative ``redux.sync`` TopK + concurrent FP8 K128 pack.
  - Large decode (``33 <= M <= 128``):
    2-kernel Hopper ``wgmma.mma_async.sync.aligned.m64n64k16`` (1 warpgroup for
    ``M <= 64``, 2 warpgroups for ``65 <= M <= 128``) with 128B-swizzled
    ``cp.async`` pipelining, followed by zero-SMEM collaborative ``redux.sync``
    TopK + concurrent FP8 K128 pack.

Precision & determinism contract:
  - FP32 dot-product accumulation rounded to BF16 before ``sqrt(softplus(x))``,
    matching ``F.linear(x, weight_bf16)`` + ``fused_sqrtsoftplus_gate_pack``.
  - Fixed-order split-K reduction (100% deterministic).
  - Identical non-finite row fallback (``indices = [0..K-1]``,
    ``weights = route_scale / K``) and bit-for-bit identical FP8 E4M3FN K128
    activation staging (``amax = max(max|x|, 1e-4)``, ``scale = amax / 448.0``,
    ``q = (x * (1.0 / scale)).to(fp8_e4m3fn)``).
  - Zero per-call tensor allocations; 100% CUDA Graph capture & replay safe.

Set ``DSV4_SM90_GATE_GEMV_FUSED=0`` to disable and revert to the shipped
``F.linear`` + Triton gate-pack path.
"""

from __future__ import annotations

import ctypes
import os
import threading
from typing import Dict, Optional, Tuple

import torch

_SUPPORTED_HIDDEN = 5120
_SUPPORTED_SHAPES = frozenset({(384, 6), (128, 3)})
_MAX_TOKENS = 128
_PACK_GRAN_K = 128

# Workspace capacity per device: max(128 * 20 * 384, 128 * 40 * 128) = 983,040 floats (~3.93 MB)
_WORKSPACE_FLOATS = _MAX_TOKENS * 20 * 384

_CUDA_SRC = r"""
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;
typedef uint16_t bf16_t;

#define INF_F (__int_as_float(0x7f800000))

__device__ __forceinline__ float2 bf16x2_to_float2(uint32_t packed) {
    float lo, hi;
    asm("{\n\t"
        ".reg .b16 h0, h1;\n\t"
        "mov.b32 {h0, h1}, %2;\n\t"
        "cvt.f32.bf16 %0, h0;\n\t"
        "cvt.f32.bf16 %1, h1;\n\t"
        "}"
        : "=f"(lo), "=f"(hi) : "r"(packed));
    return make_float2(lo, hi);
}

// PTX: cvt.rn.satfinite.e4m3x2.f32 d, a, b -> bits[15:8]=e4m3(a), bits[7:0]=e4m3(b)
__device__ __forceinline__ uint16_t cvt_f32x2_to_e4m3x2(float lo, float hi) {
    uint16_t out;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(out) : "f"(hi), "f"(lo));
    return out;
}

__device__ __forceinline__ float bf16_round_f32(float x) {
    float out;
    asm("{\n\t"
        ".reg .b16 h;\n\t"
        "cvt.rn.bf16.f32 h, %1;\n\t"
        "cvt.f32.bf16 %0, h;\n\t"
        "}"
        : "=f"(out) : "f"(x));
    return out;
}

__device__ __forceinline__ uint4 ldg128_nc(const void* ptr) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(ptr));
    return v;
}

__device__ __forceinline__ uint4 ldg128(const void* ptr) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(ptr));
    return v;
}

__device__ __forceinline__ float4 ldg128_cg_f4(const void* ptr) {
    float4 v;
    asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "l"(ptr));
    return v;
}

__device__ __forceinline__ void stg_cg_f2(void* ptr, float2 val) {
    asm volatile("st.global.cg.v2.f32 [%0], {%1, %2};" :: "l"(ptr), "f"(val.x), "f"(val.y));
}

__device__ __forceinline__ int atomic_add_acq_rel_gpu(int* ptr, int val) {
    int old;
    asm volatile("atom.global.acq_rel.gpu.add.s32 %0, [%1], %2;"
                 : "=r"(old) : "l"(ptr), "r"(val));
    return old;
}

__device__ __forceinline__ void stg_cg_s32(int* ptr, int val) {
    asm volatile("st.global.cg.s32 [%0], %1;" :: "l"(ptr), "r"(val));
}

__device__ __forceinline__ uint32_t redux_max_u32(uint32_t val) {
    uint32_t res;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(res) : "r"(val));
    return res;
}

__device__ __forceinline__ uint32_t float_to_sortable_u32(float v) {
    // Canonicalize -0.0f -> +0.0f so signed zero ties break strictly by index.
    v = v + 0.0f;
    uint32_t b = __float_as_uint(v);
    uint32_t mask = (uint32_t)(-(int)(b >> 31)) | 0x80000000u;
    return b ^ mask;
}

__device__ __forceinline__ float fast_sqrtsoftplus(float sc) {
    float c = fminf(sc, 20.0f);
    float sp = (sc > 20.0f) ? sc : __logf(1.0f + __expf(c));
    return __fsqrt_rn(sp);
}

__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, const void* glob_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(glob_ptr));
}

__device__ __forceinline__ void cp_async_cg_16_pred(void* smem_ptr, const void* glob_ptr, bool pred) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    int src_in_bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_addr), "l"(glob_ptr), "r"(src_in_bytes));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

__device__ __forceinline__ uint64_t make_wgmma_desc_128b(const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0;
    desc |= ((uint64_t)((addr & 0x3FFFF) >> 4));
    desc |= ((uint64_t)64ULL << 32);
    desc |= ((uint64_t)1ULL << 62);
    return desc;
}

__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }

__device__ __forceinline__ void wgmma_m64n64k16_bf16_ss(
    float d[32], uint64_t desc_a, uint64_t desc_b, int scale_d
) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(desc_a), "l"(desc_b), "r"(scale_d)
    );
}

// ============================================================================
// FP8 E4M3FN K128 Activation Packing (bit-exact with Triton/DeepGEMM packer)
// ============================================================================
__device__ __forceinline__ void run_fp8_pack(
    const bf16_t* __restrict__ x, uint8_t* __restrict__ out_fp8, float* __restrict__ out_sf,
    int M, int H, int x_stride_m, int fp8_stride_m, int sf_stride_m,
    int pack_cta_id, int num_pack_ctas, int warp_id, int lane_id, int num_warps
) {
    int tiles_per_row = H >> 7;
    int total_tiles = M * tiles_per_row;
    int global_warp = pack_cta_id * num_warps + warp_id;
    int stride_warps = num_pack_ctas * num_warps;
    for (int t = global_warp; t < total_tiles; t += stride_warps) {
        int row = t / tiles_per_row;
        int tk = t - row * tiles_per_row;
        int col = (tk << 7) + (lane_id << 2);
        uint2 raw = *reinterpret_cast<const uint2*>(x + (int64_t)row * x_stride_m + col);
        float2 f0 = bf16x2_to_float2(raw.x);
        float2 f1 = bf16x2_to_float2(raw.y);
        float v0 = (fabsf(f0.x) < INF_F) ? f0.x : 0.0f;
        float v1 = (fabsf(f0.y) < INF_F) ? f0.y : 0.0f;
        float v2 = (fabsf(f1.x) < INF_F) ? f1.x : 0.0f;
        float v3 = (fabsf(f1.y) < INF_F) ? f1.y : 0.0f;
        float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
        }
        amax = fmaxf(amax, 1e-4f);
        // Match Triton/PyTorch constant-division lowering: amax * (1.0f / 448.0f).
        float scale = amax * (1.0f / 448.0f);
        float inv_scale = __fdiv_rn(1.0f, scale);
        uint16_t q01 = cvt_f32x2_to_e4m3x2(v0 * inv_scale, v1 * inv_scale);
        uint16_t q23 = cvt_f32x2_to_e4m3x2(v2 * inv_scale, v3 * inv_scale);
        *reinterpret_cast<uint32_t*>(out_fp8 + (int64_t)row * fp8_stride_m + col) =
            (uint32_t)q01 | ((uint32_t)q23 << 16);
        if (lane_id == 0) {
            out_sf[(int64_t)row * sf_stride_m + tk] = scale;
        }
    }
}

// ============================================================================
// Collaborative 3-Warp TopK for E=384, K=6 (partial layout [M, SPLIT_K, 384])
// ============================================================================
template <int SPLIT_K, bool USE_CG, bool MULTI_ROW = false>
__device__ __noinline__ void run_topk_e384_3warp(
    const float* __restrict__ partial, const float* __restrict__ bias,
    float* __restrict__ out_w, int64_t* __restrict__ out_idx,
    int row, int out_w_stride_m, int out_idx_stride_m,
    float route_scale, float norm_eps,
    int sub_w, int lane_id,
    uint32_t* s_keys, float* s_vals, int* s_eids, int* s_bad, int barrier_id,
    int total_rows = 0
) {
    int eb = sub_w * 128 + (lane_id << 2);
    const float* p_row = partial + (int64_t)row * (SPLIT_K * 384) + eb;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int sk = 0; sk < SPLIT_K; ++sk) {
        float4 p = USE_CG ? ldg128_cg_f4(p_row + sk * 384)
                          : *reinterpret_cast<const float4*>(p_row + sk * 384);
        acc.x += p.x; acc.y += p.y; acc.z += p.z; acc.w += p.w;
    }
    float4 bv = *reinterpret_cast<const float4*>(bias + eb);
    float av[4] = {acc.x, acc.y, acc.z, acc.w};
    float bvv[4] = {bv.x, bv.y, bv.z, bv.w};
    float s_act[4];
    uint32_t keys[4];
    int local_bad = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sc = bf16_round_f32(av[j]);
        float b = bvv[j];
        bool ok = (fabsf(sc) < INF_F) & (fabsf(b) < INF_F);
        local_bad |= (!ok);
        float sa = fast_sqrtsoftplus(ok ? sc : 0.0f);
        s_act[j] = sa;
        keys[j] = float_to_sortable_u32(sa + (ok ? b : 0.0f));
    }
    int wbad = __any_sync(0xffffffff, local_bad);
    if (lane_id == 0) s_bad[sub_w] = wbad;

    #pragma unroll
    for (int k = 0; k < 6; ++k) {
        uint32_t m01 = (keys[0] >= keys[1]) ? keys[0] : keys[1];
        uint32_t m23 = (keys[2] >= keys[3]) ? keys[2] : keys[3];
        uint32_t my_max = (m01 >= m23) ? m01 : m23;
        uint32_t w_max = redux_max_u32(my_max);
        int win_lane = __ffs(__ballot_sync(0xffffffff, my_max == w_max)) - 1;
        if (lane_id == win_lane) {
            float win_sa; int win_e;
            if (keys[0] == w_max)      { win_sa = s_act[0]; keys[0] = 0u; win_e = eb + 0; }
            else if (keys[1] == w_max) { win_sa = s_act[1]; keys[1] = 0u; win_e = eb + 1; }
            else if (keys[2] == w_max) { win_sa = s_act[2]; keys[2] = 0u; win_e = eb + 2; }
            else                       { win_sa = s_act[3]; keys[3] = 0u; win_e = eb + 3; }
            s_keys[sub_w * 6 + k] = w_max;
            s_vals[sub_w * 6 + k] = win_sa;
            s_eids[sub_w * 6 + k] = win_e;
        }
    }
    asm volatile("bar.sync %0, 96;" :: "r"(barrier_id));

    if (sub_w == 0) {
        bool row_ok = ((s_bad[0] | s_bad[1] | s_bad[2]) == 0);
        uint32_t key = (lane_id < 18) ? s_keys[lane_id] : 0u;
        float sa = (lane_id < 18) ? s_vals[lane_id] : 0.0f;
        int eid = (lane_id < 18) ? s_eids[lane_id] : 384;
        float w_sel[6];
        int idx_sel[6];
        #pragma unroll
        for (int k = 0; k < 6; ++k) {
            uint32_t w_max = redux_max_u32(key);
            int win_lane = __ffs(__ballot_sync(0xffffffff, key == w_max)) - 1;
            float wsa = __shfl_sync(0xffffffff, sa, win_lane);
            int we = __shfl_sync(0xffffffff, eid, win_lane);
            if (lane_id == win_lane) key = 0u;
            w_sel[k] = wsa;
            idx_sel[k] = row_ok ? we : k;
        }
        if (lane_id == 0) {
            float ws = w_sel[0] + w_sel[1] + w_sel[2] + w_sel[3] + w_sel[4] + w_sel[5] + norm_eps;
            float fb = route_scale / 6.0f;
            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                out_w[(int64_t)row * out_w_stride_m + k] = row_ok ? (w_sel[k] / ws * route_scale) : fb;
                out_idx[(int64_t)row * out_idx_stride_m + k] = (int64_t)idx_sel[k];
            }
        }
    }
    if constexpr (MULTI_ROW) {
        if (row + 4 < total_rows) {
            asm volatile("bar.sync %0, 96;" :: "r"(barrier_id));
        }
    }
}

// ============================================================================
// Single-Warp TopK for E=128, K=3 (partial layout [M, SPLIT_K, 128])
// ============================================================================
template <int SPLIT_K, bool USE_CG>
__device__ __noinline__ void run_topk_e128_1warp(
    const float* __restrict__ partial, const float* __restrict__ bias,
    float* __restrict__ out_w, int64_t* __restrict__ out_idx,
    int row, int out_w_stride_m, int out_idx_stride_m,
    float route_scale, float norm_eps, int lane_id
) {
    int eb = lane_id << 2;
    const float* p_row = partial + (int64_t)row * (SPLIT_K * 128) + eb;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int sk = 0; sk < SPLIT_K; ++sk) {
        float4 p = USE_CG ? ldg128_cg_f4(p_row + sk * 128)
                          : *reinterpret_cast<const float4*>(p_row + sk * 128);
        acc.x += p.x; acc.y += p.y; acc.z += p.z; acc.w += p.w;
    }
    float4 bv = *reinterpret_cast<const float4*>(bias + eb);
    float av[4] = {acc.x, acc.y, acc.z, acc.w};
    float bvv[4] = {bv.x, bv.y, bv.z, bv.w};
    float s_act[4];
    uint32_t keys[4];
    int local_bad = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sc = bf16_round_f32(av[j]);
        float b = bvv[j];
        bool ok = (fabsf(sc) < INF_F) & (fabsf(b) < INF_F);
        local_bad |= (!ok);
        float sa = fast_sqrtsoftplus(ok ? sc : 0.0f);
        s_act[j] = sa;
        keys[j] = float_to_sortable_u32(sa + (ok ? b : 0.0f));
    }
    bool row_ok = (__any_sync(0xffffffff, local_bad) == 0);
    float w_sel[3];
    int idx_sel[3];
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        uint32_t m01 = (keys[0] >= keys[1]) ? keys[0] : keys[1];
        uint32_t m23 = (keys[2] >= keys[3]) ? keys[2] : keys[3];
        uint32_t my_max = (m01 >= m23) ? m01 : m23;
        uint32_t w_max = redux_max_u32(my_max);
        int win_lane = __ffs(__ballot_sync(0xffffffff, my_max == w_max)) - 1;
        float win_sa = 0.0f; int win_e = 0;
        if (lane_id == win_lane) {
            if (keys[0] == w_max)      { win_sa = s_act[0]; keys[0] = 0u; win_e = eb + 0; }
            else if (keys[1] == w_max) { win_sa = s_act[1]; keys[1] = 0u; win_e = eb + 1; }
            else if (keys[2] == w_max) { win_sa = s_act[2]; keys[2] = 0u; win_e = eb + 2; }
            else                       { win_sa = s_act[3]; keys[3] = 0u; win_e = eb + 3; }
        }
        w_sel[k] = __shfl_sync(0xffffffff, win_sa, win_lane);
        idx_sel[k] = row_ok ? __shfl_sync(0xffffffff, win_e, win_lane) : k;
    }
    if (lane_id == 0) {
        float ws = w_sel[0] + w_sel[1] + w_sel[2] + norm_eps;
        float fb = route_scale / 3.0f;
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            out_w[(int64_t)row * out_w_stride_m + k] = row_ok ? (w_sel[k] / ws * route_scale) : fb;
            out_idx[(int64_t)row * out_idx_stride_m + k] = (int64_t)idx_sel[k];
        }
    }
}

// ============================================================================
// 1-Kernel Small-M Zero-SMEM MMA + Concurrent Pack + Lock-Free Last-CTA TopK
//   E=384, M<=4: SPLIT_K=5, GEMM_WARPS=8, TOTAL_WARPS=12 (4 parallel 3-warp rows)
//   E=128, M<=8: SPLIT_K=8, GEMM_WARPS=10, TOTAL_WARPS=10 (10 parallel 1-warp rows)
// ============================================================================
template <int E, int H, int SPLIT_K, int GEMM_WARPS, int TOTAL_WARPS>
__global__ void zsmem_small_m_1k(
    const bf16_t* __restrict__ x, const bf16_t* __restrict__ w, const float* __restrict__ bias,
    uint8_t* __restrict__ out_fp8, float* __restrict__ out_sf,
    float* __restrict__ out_w, int64_t* __restrict__ out_idx,
    float* __restrict__ partial, int* __restrict__ flag,
    int M, int x_stride_m, int fp8_stride_m, int sf_stride_m,
    int out_w_stride_m, int out_idx_stride_m,
    float route_scale, float norm_eps, int num_pack_ctas
) {
    constexpr int BN = 8;
    constexpr int NUM_N = E / BN;
    constexpr int NUM_GEMM_CTAS = NUM_N * SPLIT_K;

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    if (bid >= NUM_GEMM_CTAS) {
        run_fp8_pack(x, out_fp8, out_sf, M, H, x_stride_m, fp8_stride_m, sf_stride_m,
                     bid - NUM_GEMM_CTAS, num_pack_ctas, warp_id, lane_id, TOTAL_WARPS);
        return;
    }

    int n_tile = bid / SPLIT_K;
    int sk = bid - n_tile * SPLIT_K;
    int e_base = n_tile * BN;

    constexpr int K_PER_SPLIT = H / SPLIT_K;
    constexpr int K_STEP_CTA = GEMM_WARPS * 32;
    constexpr int ITERS = K_PER_SPLIT / K_STEP_CTA;

    __shared__ float sC[GEMM_WARPS * 128];

    if (warp_id < GEMM_WARPS) {
        int gid = lane_id >> 2;
        int tig = lane_id & 3;
        int k_start = sk * K_PER_SPLIT + warp_id * 32 + tig * 8;

        bool v0 = (gid < M);
        bool v1 = ((gid + 8) < M);
        const bf16_t* xp0 = x + (int64_t)(v0 ? gid : 0) * x_stride_m + k_start;
        const bf16_t* xp1 = x + (int64_t)(v1 ? (gid + 8) : 0) * x_stride_m + k_start;
        const bf16_t* wp  = w + (int64_t)(e_base + gid) * H + k_start;

        uint4 wv[ITERS];
        #pragma unroll
        for (int it = 0; it < ITERS; ++it) wv[it] = ldg128_nc(wp + it * K_STEP_CTA);

        float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
        #pragma unroll
        for (int it = 0; it < ITERS; ++it) {
            int koff = it * K_STEP_CTA;
            uint4 xv0 = v0 ? ldg128(xp0 + koff) : make_uint4(0, 0, 0, 0);
            uint4 xv1 = v1 ? ldg128(xp1 + koff) : make_uint4(0, 0, 0, 0);
            mma_m16n8k16_bf16(c0, c1, c2, c3, xv0.x, xv1.x, xv0.y, xv1.y, wv[it].x, wv[it].y);
            mma_m16n8k16_bf16(c0, c1, c2, c3, xv0.z, xv1.z, xv0.w, xv1.w, wv[it].z, wv[it].w);
        }

        int col = tig * 2;
        sC[warp_id * 128 + gid * 8 + col]           = c0;
        sC[warp_id * 128 + gid * 8 + col + 1]       = c1;
        sC[warp_id * 128 + (gid + 8) * 8 + col]     = c2;
        sC[warp_id * 128 + (gid + 8) * 8 + col + 1] = c3;
    }
    __syncthreads();

    if (tid < M * 4) {
        int row = tid >> 2;
        int col = (tid & 3) << 1;
        float s0 = 0.0f, s1 = 0.0f;
        #pragma unroll
        for (int wi = 0; wi < GEMM_WARPS; ++wi) {
            s0 += sC[wi * 128 + row * 8 + col];
            s1 += sC[wi * 128 + row * 8 + col + 1];
        }
        stg_cg_f2(partial + ((int64_t)row * SPLIT_K + sk) * E + (e_base + col), make_float2(s0, s1));
    }
    __syncthreads();

    __shared__ int is_last;
    if (tid == 0) {
        int prev = atomic_add_acq_rel_gpu(flag, 1);
        is_last = (prev == NUM_GEMM_CTAS - 1) ? 1 : 0;
        if (is_last) stg_cg_s32(flag, 0);
    }
    __syncthreads();
    if (!is_last) return;

    if constexpr (E == 128) {
        for (int row = warp_id; row < M; row += TOTAL_WARPS) {
            run_topk_e128_1warp<SPLIT_K, true>(
                partial, bias, out_w, out_idx, row,
                out_w_stride_m, out_idx_stride_m, route_scale, norm_eps, lane_id);
        }
    } else {
        constexpr int GROUPS = TOTAL_WARPS / 3;
        int grp = warp_id / 3;
        int sub_w = warp_id - grp * 3;
        if (grp < GROUPS) {
            __shared__ uint32_t s_keys[GROUPS][18];
            __shared__ float s_vals[GROUPS][18];
            __shared__ int s_eids[GROUPS][18];
            __shared__ int s_bad[GROUPS][3];
            for (int row = grp; row < M; row += GROUPS) {
                run_topk_e384_3warp<SPLIT_K, true, true>(
                    partial, bias, out_w, out_idx, row,
                    out_w_stride_m, out_idx_stride_m, route_scale, norm_eps,
                    sub_w, lane_id, s_keys[grp], s_vals[grp], s_eids[grp], s_bad[grp],
                    grp + 1, M);
            }
        }
    }
}

// ============================================================================
// Mid-M Zero-SMEM MMA K1 (for 5 <= M <= 32 on E=384, 9 <= M <= 32 on E=128)
// ============================================================================
template <int E, int H, int M_SUB, int N_SUB, int SPLIT_K, int WARPS>
__global__ void zsmem_mid_m_k1(
    const bf16_t* __restrict__ x, const bf16_t* __restrict__ w,
    float* __restrict__ partial, int M, int x_stride_m
) {
    constexpr int BM = M_SUB * 16;
    constexpr int BN = N_SUB * 8;
    constexpr int NUM_N = E / BN;
    constexpr int CTAS_PER_M = NUM_N * SPLIT_K;

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    int m_tile = bid / CTAS_PER_M;
    int rem = bid - m_tile * CTAS_PER_M;
    int n_tile = rem / SPLIT_K;
    int sk = rem - n_tile * SPLIT_K;
    int m_base = m_tile * BM;
    int e_base = n_tile * BN;

    constexpr int K_PER_SPLIT = H / SPLIT_K;
    constexpr int K_STEP = WARPS * 32;
    constexpr int ITERS = K_PER_SPLIT / K_STEP;

    int gid = lane_id >> 2;
    int tig = lane_id & 3;
    int k_start = sk * K_PER_SPLIT + warp_id * 32 + tig * 8;

    float c[M_SUB][N_SUB][4];
    #pragma unroll
    for (int mi = 0; mi < M_SUB; ++mi)
        #pragma unroll
        for (int ni = 0; ni < N_SUB; ++ni)
            c[mi][ni][0] = c[mi][ni][1] = c[mi][ni][2] = c[mi][ni][3] = 0.0f;

    bool v0[M_SUB], v1[M_SUB];
    const bf16_t* xp0[M_SUB];
    const bf16_t* xp1[M_SUB];
    #pragma unroll
    for (int mi = 0; mi < M_SUB; ++mi) {
        int r0 = m_base + mi * 16 + gid;
        int r1 = r0 + 8;
        v0[mi] = (r0 < M);
        v1[mi] = (r1 < M);
        xp0[mi] = x + (int64_t)(v0[mi] ? r0 : 0) * x_stride_m + k_start;
        xp1[mi] = x + (int64_t)(v1[mi] ? r1 : 0) * x_stride_m + k_start;
    }
    const bf16_t* wp[N_SUB];
    #pragma unroll
    for (int ni = 0; ni < N_SUB; ++ni) {
        wp[ni] = w + (int64_t)(e_base + ni * 8 + gid) * H + k_start;
    }

    if constexpr (ITERS <= 4 && N_SUB == 1) {
        uint4 wv[ITERS];
        #pragma unroll
        for (int it = 0; it < ITERS; ++it) wv[it] = ldg128_nc(wp[0] + it * K_STEP);
        #pragma unroll
        for (int it = 0; it < ITERS; ++it) {
            int koff = it * K_STEP;
            uint4 wb = wv[it];
            #pragma unroll
            for (int mi = 0; mi < M_SUB; ++mi) {
                uint4 xv0 = v0[mi] ? ldg128(xp0[mi] + koff) : make_uint4(0, 0, 0, 0);
                uint4 xv1 = v1[mi] ? ldg128(xp1[mi] + koff) : make_uint4(0, 0, 0, 0);
                mma_m16n8k16_bf16(c[mi][0][0], c[mi][0][1], c[mi][0][2], c[mi][0][3],
                                  xv0.x, xv1.x, xv0.y, xv1.y, wb.x, wb.y);
                mma_m16n8k16_bf16(c[mi][0][0], c[mi][0][1], c[mi][0][2], c[mi][0][3],
                                  xv0.z, xv1.z, xv0.w, xv1.w, wb.z, wb.w);
            }
        }
    } else {
        #pragma unroll
        for (int it = 0; it < ITERS; ++it) {
            int koff = it * K_STEP;
            uint4 wv[N_SUB];
            #pragma unroll
            for (int ni = 0; ni < N_SUB; ++ni) wv[ni] = ldg128_nc(wp[ni] + koff);
            #pragma unroll
            for (int mi = 0; mi < M_SUB; ++mi) {
                uint4 xv0 = v0[mi] ? ldg128(xp0[mi] + koff) : make_uint4(0, 0, 0, 0);
                uint4 xv1 = v1[mi] ? ldg128(xp1[mi] + koff) : make_uint4(0, 0, 0, 0);
                #pragma unroll
                for (int ni = 0; ni < N_SUB; ++ni) {
                    mma_m16n8k16_bf16(c[mi][ni][0], c[mi][ni][1], c[mi][ni][2], c[mi][ni][3],
                                      xv0.x, xv1.x, xv0.y, xv1.y, wv[ni].x, wv[ni].y);
                    mma_m16n8k16_bf16(c[mi][ni][0], c[mi][ni][1], c[mi][ni][2], c[mi][ni][3],
                                      xv0.z, xv1.z, xv0.w, xv1.w, wv[ni].z, wv[ni].w);
                }
            }
        }
    }

    constexpr int TILE_ELEMS = BM * BN;
    __shared__ float sC[WARPS * TILE_ELEMS];
    #pragma unroll
    for (int mi = 0; mi < M_SUB; ++mi) {
        int r0 = mi * 16 + gid;
        int r1 = r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < N_SUB; ++ni) {
            int col = ni * 8 + tig * 2;
            sC[warp_id * TILE_ELEMS + r0 * BN + col]     = c[mi][ni][0];
            sC[warp_id * TILE_ELEMS + r0 * BN + col + 1] = c[mi][ni][1];
            sC[warp_id * TILE_ELEMS + r1 * BN + col]     = c[mi][ni][2];
            sC[warp_id * TILE_ELEMS + r1 * BN + col + 1] = c[mi][ni][3];
        }
    }
    __syncthreads();

    constexpr int PAIRS = TILE_ELEMS / 2;
    #pragma unroll
    for (int idx = tid; idx < PAIRS; idx += WARPS * 32) {
        int ml = (idx * 2) / BN;
        int col = (idx * 2) - ml * BN;
        int row = m_base + ml;
        if (row < M) {
            float s0 = 0.0f, s1 = 0.0f;
            #pragma unroll
            for (int wi = 0; wi < WARPS; ++wi) {
                s0 += sC[wi * TILE_ELEMS + ml * BN + col];
                s1 += sC[wi * TILE_ELEMS + ml * BN + col + 1];
            }
            stg_cg_f2(partial + ((int64_t)row * SPLIT_K + sk) * E + (e_base + col), make_float2(s0, s1));
        }
    }
}

// ============================================================================
// Hopper SM90 WGMMA m64n64k16 K1 (for 33 <= M <= 128)
// ============================================================================
template <int E, int H, int NUM_WG, int SPLIT_K, int STAGES>
__global__ void wgmma_pure_k1(
    const bf16_t* __restrict__ x, const bf16_t* __restrict__ w,
    float* __restrict__ partial, int M, int x_stride_m
) {
    constexpr int BM = NUM_WG * 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int THREADS = NUM_WG * 128;

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int n_tile = bid / SPLIT_K;
    int sk = bid - n_tile * SPLIT_K;
    int e_base = n_tile * BN;

    constexpr int K_PER_SPLIT = H / SPLIT_K;
    constexpr int K_ITERS = K_PER_SPLIT / BK;
    int k_split_start = sk * K_PER_SPLIT;

    extern __shared__ __align__(1024) uint8_t smem_raw[];
    bf16_t* sA_all = reinterpret_cast<bf16_t*>(smem_raw);
    bf16_t* sB_all = sA_all + STAGES * BM * BK;

    constexpr int A_VECS = BM * 8;
    constexpr int B_VECS = BN * 8;

    auto load_stage = [&](int st, int k_off) {
        bf16_t* sA = sA_all + st * (BM * BK);
        bf16_t* sB = sB_all + st * (BN * BK);
        #pragma unroll
        for (int idx = tid; idx < B_VECS; idx += THREADS) {
            int r = idx >> 3;
            int c = idx & 7;
            int sc = c ^ (r & 7);
            cp_async_cg_16(&sB[r * BK + sc * 8], w + (int64_t)(e_base + r) * H + k_off + c * 8);
        }
        #pragma unroll
        for (int idx = tid; idx < A_VECS; idx += THREADS) {
            int r = idx >> 3;
            int c = idx & 7;
            int sc = c ^ (r & 7);
            cp_async_cg_16_pred(&sA[r * BK + sc * 8], x + (int64_t)r * x_stride_m + k_off + c * 8, r < M);
        }
        cp_async_commit();
    };

    #pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        if (st < K_ITERS) load_stage(st, k_split_start + st * BK);
        else cp_async_commit();
    }

    int wg_id = tid >> 7;
    int wg_tid = tid & 127;
    int wg_warp = wg_tid >> 5;
    int wg_lane = wg_tid & 31;

    float d[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) d[i] = 0.0f;

    int first = 1;
    for (int it = 0; it < K_ITERS; ++it) {
        int cur = it % STAGES;
        int next_it = it + (STAGES - 1);
        int nxt = next_it % STAGES;
        if (next_it < K_ITERS) {
            load_stage(nxt, k_split_start + next_it * BK);
            cp_async_wait<STAGES - 1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        const bf16_t* sA = sA_all + cur * (BM * BK) + wg_id * (64 * BK);
        const bf16_t* sB = sB_all + cur * (BN * BK);

        wgmma_fence();
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            uint64_t da = make_wgmma_desc_128b(sA + ks * 16);
            uint64_t db = make_wgmma_desc_128b(sB + ks * 16);
            wgmma_m64n64k16_bf16_ss(d, da, db, (first && ks == 0) ? 0 : 1);
        }
        first = 0;
        wgmma_commit();
        wgmma_wait<0>();
        __syncthreads();
    }

    int r0 = wg_id * 64 + wg_warp * 16 + (wg_lane >> 2);
    int r1 = r0 + 8;
    int tig = wg_lane & 3;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int col = e_base + j * 8 + tig * 2;
        if (r0 < M) stg_cg_f2(partial + ((int64_t)r0 * SPLIT_K + sk) * E + col, make_float2(d[j * 4 + 0], d[j * 4 + 1]));
        if (r1 < M) stg_cg_f2(partial + ((int64_t)r1 * SPLIT_K + sk) * E + col, make_float2(d[j * 4 + 2], d[j * 4 + 3]));
    }
}

// ============================================================================
// Zero-SMEM K2: TopK + Concurrent FP8 K128 Pack
// ============================================================================
template <int SPLIT_K>
__global__ void tk384_pack_k2(
    const float* __restrict__ partial, const float* __restrict__ bias,
    const bf16_t* __restrict__ x, uint8_t* __restrict__ out_fp8, float* __restrict__ out_sf,
    float* __restrict__ out_w, int64_t* __restrict__ out_idx,
    int M, int H, int x_stride_m, int fp8_stride_m, int sf_stride_m,
    int out_w_stride_m, int out_idx_stride_m,
    float route_scale, float norm_eps, int num_pack_ctas
) {
    int bid = blockIdx.x;
    int sub_w = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    if (bid >= M) {
        run_fp8_pack(x, out_fp8, out_sf, M, H, x_stride_m, fp8_stride_m, sf_stride_m,
                     bid - M, num_pack_ctas, sub_w, lane_id, 3);
        return;
    }
    __shared__ uint32_t s_keys[18];
    __shared__ float s_vals[18];
    __shared__ int s_eids[18];
    __shared__ int s_bad[3];
    run_topk_e384_3warp<SPLIT_K, false>(
        partial, bias, out_w, out_idx, bid,
        out_w_stride_m, out_idx_stride_m, route_scale, norm_eps,
        sub_w, lane_id, s_keys, s_vals, s_eids, s_bad, 1);
}

template <int SPLIT_K>
__global__ void tk128_pack_k2(
    const float* __restrict__ partial, const float* __restrict__ bias,
    const bf16_t* __restrict__ x, uint8_t* __restrict__ out_fp8, float* __restrict__ out_sf,
    float* __restrict__ out_w, int64_t* __restrict__ out_idx,
    int M, int H, int x_stride_m, int fp8_stride_m, int sf_stride_m,
    int out_w_stride_m, int out_idx_stride_m,
    float route_scale, float norm_eps, int num_pack_ctas
) {
    int bid = blockIdx.x;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int row_ctas = (M + 3) >> 2;
    if (bid >= row_ctas) {
        run_fp8_pack(x, out_fp8, out_sf, M, H, x_stride_m, fp8_stride_m, sf_stride_m,
                     bid - row_ctas, num_pack_ctas, warp_id, lane_id, 4);
        return;
    }
    int row = bid * 4 + warp_id;
    if (row >= M) return;
    run_topk_e128_1warp<SPLIT_K, false>(
        partial, bias, out_w, out_idx, row,
        out_w_stride_m, out_idx_stride_m, route_scale, norm_eps, lane_id);
}

extern "C" {
// --- E=384, K=6 ---
__global__ void s384_1k_sk5_w12(
    const bf16_t* x, const bf16_t* w, const float* b, uint8_t* f, float* sf,
    float* ow, int64_t* oi, float* p, int* fl,
    int M, int xs, int fs, int sfs, int ws, int is, float rs, float eps, int np
) {
    zsmem_small_m_1k<384, 5120, 5, 8, 12>(x, w, b, f, sf, ow, oi, p, fl, M, xs, fs, sfs, ws, is, rs, eps, np);
}
__global__ void z384_m1_n1_sk8_w10_k1(const bf16_t* x, const bf16_t* w, float* p, int M, int xs) {
    zsmem_mid_m_k1<384, 5120, 1, 1, 8, 10>(x, w, p, M, xs);
}
__global__ void z384_m2_n1_sk5_w8_k1(const bf16_t* x, const bf16_t* w, float* p, int M, int xs) {
    zsmem_mid_m_k1<384, 5120, 2, 1, 5, 8>(x, w, p, M, xs);
}
__global__ void wg384_64_sk20_s3(const bf16_t* x, const bf16_t* w, float* p, int M, int xs) {
    wgmma_pure_k1<384, 5120, 1, 20, 3>(x, w, p, M, xs);
}
__global__ void wg384_128_sk20_s3(const bf16_t* x, const bf16_t* w, float* p, int M, int xs) {
    wgmma_pure_k1<384, 5120, 2, 20, 3>(x, w, p, M, xs);
}
__global__ void tkp384_sk5(
    const float* p, const float* b, const bf16_t* x, uint8_t* f, float* sf,
    float* ow, int64_t* oi, int M, int H, int xs, int fs, int sfs, int ws, int is, float rs, float eps, int np
) {
    tk384_pack_k2<5>(p, b, x, f, sf, ow, oi, M, H, xs, fs, sfs, ws, is, rs, eps, np);
}
__global__ void tkp384_sk8(
    const float* p, const float* b, const bf16_t* x, uint8_t* f, float* sf,
    float* ow, int64_t* oi, int M, int H, int xs, int fs, int sfs, int ws, int is, float rs, float eps, int np
) {
    tk384_pack_k2<8>(p, b, x, f, sf, ow, oi, M, H, xs, fs, sfs, ws, is, rs, eps, np);
}
__global__ void tkp384_sk20(
    const float* p, const float* b, const bf16_t* x, uint8_t* f, float* sf,
    float* ow, int64_t* oi, int M, int H, int xs, int fs, int sfs, int ws, int is, float rs, float eps, int np
) {
    tk384_pack_k2<20>(p, b, x, f, sf, ow, oi, M, H, xs, fs, sfs, ws, is, rs, eps, np);
}

// --- E=128, K=3 ---
__global__ void s128_1k_sk8_w10(
    const bf16_t* x, const bf16_t* w, const float* b, uint8_t* f, float* sf,
    float* ow, int64_t* oi, float* p, int* fl,
    int M, int xs, int fs, int sfs, int ws, int is, float rs, float eps, int np
) {
    zsmem_small_m_1k<128, 5120, 8, 10, 10>(x, w, b, f, sf, ow, oi, p, fl, M, xs, fs, sfs, ws, is, rs, eps, np);
}
__global__ void z128_m1_n1_sk5_w8_k1(const bf16_t* x, const bf16_t* w, float* p, int M, int xs) {
    zsmem_mid_m_k1<128, 5120, 1, 1, 5, 8>(x, w, p, M, xs);
}
__global__ void z128_m1_n2_sk5_w8_k1(const bf16_t* x, const bf16_t* w, float* p, int M, int xs) {
    zsmem_mid_m_k1<128, 5120, 1, 2, 5, 8>(x, w, p, M, xs);
}
__global__ void wg128_64_sk20_s3(const bf16_t* x, const bf16_t* w, float* p, int M, int xs) {
    wgmma_pure_k1<128, 5120, 1, 20, 3>(x, w, p, M, xs);
}
__global__ void wg128_128_sk40_s2(const bf16_t* x, const bf16_t* w, float* p, int M, int xs) {
    wgmma_pure_k1<128, 5120, 2, 40, 2>(x, w, p, M, xs);
}
__global__ void tkp128_sk5(
    const float* p, const float* b, const bf16_t* x, uint8_t* f, float* sf,
    float* ow, int64_t* oi, int M, int H, int xs, int fs, int sfs, int ws, int is, float rs, float eps, int np
) {
    tk128_pack_k2<5>(p, b, x, f, sf, ow, oi, M, H, xs, fs, sfs, ws, is, rs, eps, np);
}
__global__ void tkp128_sk20(
    const float* p, const float* b, const bf16_t* x, uint8_t* f, float* sf,
    float* ow, int64_t* oi, int M, int H, int xs, int fs, int sfs, int ws, int is, float rs, float eps, int np
) {
    tk128_pack_k2<20>(p, b, x, f, sf, ow, oi, M, H, xs, fs, sfs, ws, is, rs, eps, np);
}
__global__ void tkp128_sk40(
    const float* p, const float* b, const bf16_t* x, uint8_t* f, float* sf,
    float* ow, int64_t* oi, int M, int H, int xs, int fs, int sfs, int ws, int is, float rs, float eps, int np
) {
    tk128_pack_k2<40>(p, b, x, f, sf, ow, oi, M, H, xs, fs, sfs, ws, is, rs, eps, np);
}
}
"""

_KERNEL_NAMES = (
    "s384_1k_sk5_w12",
    "z384_m1_n1_sk8_w10_k1",
    "z384_m2_n1_sk5_w8_k1",
    "wg384_64_sk20_s3",
    "wg384_128_sk20_s3",
    "tkp384_sk5",
    "tkp384_sk8",
    "tkp384_sk20",
    "s128_1k_sk8_w10",
    "z128_m1_n1_sk5_w8_k1",
    "z128_m1_n2_sk5_w8_k1",
    "wg128_64_sk20_s3",
    "wg128_128_sk40_s2",
    "tkp128_sk5",
    "tkp128_sk20",
    "tkp128_sk40",
)

_SMEM_WG64_S3 = 3 * (64 + 64) * 64 * 2    # 49,152 B
_SMEM_WG128_S3 = 3 * (128 + 64) * 64 * 2  # 73,728 B
_SMEM_WG128_S2 = 2 * (128 + 64) * 64 * 2  # 49,152 B

_HIGH_SMEM_KERNELS = {
    "wg384_64_sk20_s3": _SMEM_WG64_S3,
    "wg384_128_sk20_s3": _SMEM_WG128_S3,
    "wg128_64_sk20_s3": _SMEM_WG64_S3,
    "wg128_128_sk40_s2": _SMEM_WG128_S2,
}

_MAX_CACHED_STREAMS = 32
_LOCK = threading.Lock()
_BINDINGS_PROBED = False
_CUDA_DRV = None
_NVRTC = None
_CUBIN_BYTES: Optional[bytes] = None
_SM90_CAPABLE_DEVICES: Dict[int, bool] = {}
_DEVICE_MODULES: Dict[int, Tuple[object, Dict[str, object]]] = {}
_STREAM_BUFFERS: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
_CAPTURE_BUFFERS: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}


def _probe_bindings() -> bool:
    global _BINDINGS_PROBED, _CUDA_DRV, _NVRTC
    if _BINDINGS_PROBED:
        return _CUDA_DRV is not None and _NVRTC is not None
    with _LOCK:
        if _BINDINGS_PROBED:
            return _CUDA_DRV is not None and _NVRTC is not None
        try:
            import cuda.bindings.driver as cuda_drv
            import cuda.bindings.nvrtc as nvrtc

            _CUDA_DRV = cuda_drv
            _NVRTC = nvrtc
        except Exception:  # noqa: BLE001
            _CUDA_DRV = None
            _NVRTC = None
        _BINDINGS_PROBED = True
        return _CUDA_DRV is not None and _NVRTC is not None


def _check_nvrtc(res):
    if res[0] != _NVRTC.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"NVRTC error: {res}")
    return res[1:] if len(res) > 1 else None


def _check_drv(res):
    if res[0] != _CUDA_DRV.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error: {res}")
    return res[1:] if len(res) > 1 else None


def _compile_cubin() -> bytes:
    global _CUBIN_BYTES
    if _CUBIN_BYTES is not None:
        return _CUBIN_BYTES
    (prog,) = _check_nvrtc(
        _NVRTC.nvrtcCreateProgram(
            _CUDA_SRC.encode("utf-8"), b"sm90_gate_gemv_fused.cu", 0, [], []
        )
    )
    opts = [b"--gpu-architecture=sm_90a", b"--std=c++17"]
    res = _NVRTC.nvrtcCompileProgram(prog, len(opts), opts)
    if res[0] != _NVRTC.nvrtcResult.NVRTC_SUCCESS:
        (log_size,) = _check_nvrtc(_NVRTC.nvrtcGetProgramLogSize(prog))
        log = bytearray(log_size)
        _check_nvrtc(_NVRTC.nvrtcGetProgramLog(prog, log))
        _NVRTC.nvrtcDestroyProgram(prog)
        raise RuntimeError(
            "NVRTC compile failed for sm90_gate_gemv_fused:\n"
            + log.decode("utf-8", errors="replace")
        )
    (cubin_size,) = _check_nvrtc(_NVRTC.nvrtcGetCUBINSize(prog))
    cubin = bytearray(cubin_size)
    _check_nvrtc(_NVRTC.nvrtcGetCUBIN(prog, cubin))
    _check_nvrtc(_NVRTC.nvrtcDestroyProgram(prog))
    _CUBIN_BYTES = bytes(cubin)
    return _CUBIN_BYTES


def _is_sm90_device(dev_idx: int) -> bool:
    capable = _SM90_CAPABLE_DEVICES.get(dev_idx)
    if capable is not None:
        return capable
    capable = torch.cuda.is_available() and torch.cuda.get_device_capability(dev_idx) == (9, 0)
    _SM90_CAPABLE_DEVICES[dev_idx] = capable
    return capable


def _get_device_module(dev_idx: int) -> Tuple[object, Dict[str, object]]:
    mod_state = _DEVICE_MODULES.get(dev_idx)
    if mod_state is not None:
        return mod_state
    with _LOCK:
        mod_state = _DEVICE_MODULES.get(dev_idx)
        if mod_state is not None:
            return mod_state
        if not _probe_bindings():
            raise RuntimeError("cuda.bindings.driver/nvrtc unavailable")
        with torch.cuda.device(dev_idx):
            # Ensure CUDA context is active before cuModuleLoadData and preallocate capture buffer.
            if dev_idx not in _CAPTURE_BUFFERS:
                _CAPTURE_BUFFERS[dev_idx] = (
                    torch.empty((_WORKSPACE_FLOATS,), dtype=torch.float32, device=f"cuda:{dev_idx}"),
                    torch.zeros((1,), dtype=torch.int32, device=f"cuda:{dev_idx}"),
                )
            cubin = _compile_cubin()
            (mod,) = _check_drv(_CUDA_DRV.cuModuleLoadData(cubin))
            fns: Dict[str, object] = {}
            max_smem_attr = (
                _CUDA_DRV.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            )
            for name in _KERNEL_NAMES:
                (fn,) = _check_drv(
                    _CUDA_DRV.cuModuleGetFunction(mod, name.encode("utf-8"))
                )
                smem = _HIGH_SMEM_KERNELS.get(name, 0)
                if smem >= 48 * 1024:
                    _check_drv(_CUDA_DRV.cuFuncSetAttribute(fn, max_smem_attr, smem))
                fns[name] = fn
            mod_state = (mod, fns)
            _DEVICE_MODULES[dev_idx] = mod_state
            return mod_state


def _get_stream_buffers(
    dev_idx: int, stream_ptr: int, *, capturing: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (dev_idx, stream_ptr)
    bufs = _STREAM_BUFFERS.get(key)
    if bufs is not None:
        return bufs
    if capturing:
        cap_bufs = _CAPTURE_BUFFERS.get(dev_idx)
        if cap_bufs is None:
            raise RuntimeError(f"SM90 fused gate workspace not warmed before capture on cuda:{dev_idx}")
        return cap_bufs
    with _LOCK:
        bufs = _STREAM_BUFFERS.get(key)
        if bufs is not None:
            return bufs
        partial = torch.empty((_WORKSPACE_FLOATS,), dtype=torch.float32, device=f"cuda:{dev_idx}")
        flag = torch.zeros((1,), dtype=torch.int32, device=f"cuda:{dev_idx}")
        bufs = (partial, flag)
        if len(_STREAM_BUFFERS) < _MAX_CACHED_STREAMS:
            _STREAM_BUFFERS[key] = bufs
        if dev_idx not in _CAPTURE_BUFFERS:
            _CAPTURE_BUFFERS[dev_idx] = (
                torch.empty((_WORKSPACE_FLOATS,), dtype=torch.float32, device=f"cuda:{dev_idx}"),
                torch.zeros((1,), dtype=torch.int32, device=f"cuda:{dev_idx}"),
            )
        return bufs


def warmup_sm90_gate_gemv_pack(device: Optional[torch.device] = None) -> bool:
    """Compile and cache the SM90 fused router module + workspace on ``device``."""
    if os.environ.get("DSV4_SM90_GATE_GEMV_FUSED", "1") == "0":
        return False
    if not torch.cuda.is_available() or not _probe_bindings():
        return False
    dev = (
        torch.device("cuda", torch.cuda.current_device())
        if device is None
        else torch.device(device)
    )
    if dev.type != "cuda":
        return False
    dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
    if not _is_sm90_device(dev_idx):
        return False
    with torch.cuda.device(dev_idx):
        _get_device_module(dev_idx)
        stream_ptr = torch.cuda.current_stream(dev_idx).cuda_stream
        _get_stream_buffers(dev_idx, stream_ptr, capturing=False)
    return True


def can_use_sm90_gate_gemv_pack(
    x: torch.Tensor,
    weight_bf16: torch.Tensor,
    bias: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_weights: torch.Tensor,
    out_indices: torch.Tensor,
    topk: int,
) -> bool:
    """Check whether the SM90 fused Gate-GEMV/WGMMA + TopK + FP8 Pack path applies."""
    if os.environ.get("DSV4_SM90_GATE_GEMV_FUSED", "1") == "0":
        return False
    if not _probe_bindings():
        return False
    if (
        not isinstance(x, torch.Tensor)
        or not isinstance(weight_bf16, torch.Tensor)
        or not isinstance(bias, torch.Tensor)
    ):
        return False
    if (
        not x.is_cuda
        or x.ndim != 2
        or weight_bf16.ndim != 2
        or bias.ndim != 1
        or x.dtype != torch.bfloat16
        or weight_bf16.dtype != torch.bfloat16
        or bias.dtype != torch.float32
    ):
        return False
    m, h = x.shape
    e, w_h = weight_bf16.shape
    if (
        m <= 0
        or m > _MAX_TOKENS
        or h != _SUPPORTED_HIDDEN
        or w_h != h
        or (e, int(topk)) not in _SUPPORTED_SHAPES
        or bias.shape != (e,)
    ):
        return False
    if (
        out_fp8.dtype != torch.float8_e4m3fn
        or out_sf.dtype != torch.float32
        or out_weights.dtype != torch.float32
        or out_indices.dtype != torch.int64
        or out_fp8.shape != (m, h)
        or out_sf.shape != (m, h // _PACK_GRAN_K)
        or out_weights.shape != (m, int(topk))
        or out_indices.shape != (m, int(topk))
    ):
        return False
    dev = x.device
    if (
        weight_bf16.device != dev
        or bias.device != dev
        or out_fp8.device != dev
        or out_sf.device != dev
        or out_weights.device != dev
        or out_indices.device != dev
    ):
        return False
    dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
    if not _is_sm90_device(dev_idx):
        return False
    # Vectorized 128-bit load / 32-bit pack store alignment checks.
    if (
        x.stride(1) != 1
        or (x.stride(0) & 7) != 0
        or (x.data_ptr() & 15) != 0
        or not weight_bf16.is_contiguous()
        or (weight_bf16.data_ptr() & 15) != 0
        or not bias.is_contiguous()
        or (bias.data_ptr() & 15) != 0
        or out_fp8.stride(1) != 1
        or (out_fp8.stride(0) & 3) != 0
        or (out_fp8.data_ptr() & 3) != 0
        or out_sf.stride(1) != 1
        or out_weights.stride(1) != 1
        or out_indices.stride(1) != 1
    ):
        return False
    # Never trigger lazy NVRTC compilation / workspace alloc during stream capture.
    if torch.cuda.is_current_stream_capturing():
        if dev_idx not in _DEVICE_MODULES or (
            dev_idx not in _CAPTURE_BUFFERS
            and (dev_idx, torch.cuda.current_stream(dev_idx).cuda_stream) not in _STREAM_BUFFERS
        ):
            return False
    return True


def _launch_1k(
    fn,
    grid_x: int,
    block_x: int,
    stream: int,
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_w: torch.Tensor,
    out_idx: torch.Tensor,
    partial: torch.Tensor,
    flag: torch.Tensor,
    m: int,
    route_scale: float,
    norm_eps: float,
    num_pack_ctas: int,
) -> None:
    args = (
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(w.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out_fp8.data_ptr()),
        ctypes.c_void_p(out_sf.data_ptr()),
        ctypes.c_void_p(out_w.data_ptr()),
        ctypes.c_void_p(out_idx.data_ptr()),
        ctypes.c_void_p(partial.data_ptr()),
        ctypes.c_void_p(flag.data_ptr()),
        ctypes.c_int32(m),
        ctypes.c_int32(x.stride(0)),
        ctypes.c_int32(out_fp8.stride(0)),
        ctypes.c_int32(out_sf.stride(0)),
        ctypes.c_int32(out_w.stride(0)),
        ctypes.c_int32(out_idx.stride(0)),
        ctypes.c_float(route_scale),
        ctypes.c_float(norm_eps),
        ctypes.c_int32(num_pack_ctas),
    )
    arg_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
    _check_drv(
        _CUDA_DRV.cuLaunchKernel(
            fn, grid_x, 1, 1, block_x, 1, 1, 0, stream, ctypes.addressof(arg_ptrs), 0
        )
    )


def _launch_k1(
    fn,
    grid_x: int,
    block_x: int,
    smem_bytes: int,
    stream: int,
    x: torch.Tensor,
    w: torch.Tensor,
    partial: torch.Tensor,
    m: int,
) -> None:
    args = (
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(w.data_ptr()),
        ctypes.c_void_p(partial.data_ptr()),
        ctypes.c_int32(m),
        ctypes.c_int32(x.stride(0)),
    )
    arg_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
    _check_drv(
        _CUDA_DRV.cuLaunchKernel(
            fn,
            grid_x,
            1,
            1,
            block_x,
            1,
            1,
            smem_bytes,
            stream,
            ctypes.addressof(arg_ptrs),
            0,
        )
    )


def _launch_k2_pack(
    fn,
    grid_x: int,
    block_x: int,
    stream: int,
    partial: torch.Tensor,
    bias: torch.Tensor,
    x: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_w: torch.Tensor,
    out_idx: torch.Tensor,
    m: int,
    h: int,
    route_scale: float,
    norm_eps: float,
    num_pack_ctas: int,
) -> None:
    args = (
        ctypes.c_void_p(partial.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(out_fp8.data_ptr()),
        ctypes.c_void_p(out_sf.data_ptr()),
        ctypes.c_void_p(out_w.data_ptr()),
        ctypes.c_void_p(out_idx.data_ptr()),
        ctypes.c_int32(m),
        ctypes.c_int32(h),
        ctypes.c_int32(x.stride(0)),
        ctypes.c_int32(out_fp8.stride(0)),
        ctypes.c_int32(out_sf.stride(0)),
        ctypes.c_int32(out_w.stride(0)),
        ctypes.c_int32(out_idx.stride(0)),
        ctypes.c_float(route_scale),
        ctypes.c_float(norm_eps),
        ctypes.c_int32(num_pack_ctas),
    )
    arg_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
    _check_drv(
        _CUDA_DRV.cuLaunchKernel(
            fn, grid_x, 1, 1, block_x, 1, 1, 0, stream, ctypes.addressof(arg_ptrs), 0
        )
    )


def fused_sm90_gate_gemv_pack(
    x: torch.Tensor,
    weight_bf16: torch.Tensor,
    bias: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_weights: torch.Tensor,
    out_indices: torch.Tensor,
    topk: int,
    route_scale: float = 1.0,
    norm_eps: float = 1e-12,
) -> None:
    """Fused SM90 BF16 gate projection + sqrtsoftplus + TopK + FP8 K128 pack."""
    m, h = x.shape
    e = weight_bf16.shape[0]
    if m == 0:
        return
    if not can_use_sm90_gate_gemv_pack(
        x, weight_bf16, bias, out_fp8, out_sf, out_weights, out_indices, topk
    ):
        raise ValueError(
            "Unsupported inputs for fused_sm90_gate_gemv_pack: "
            f"x={tuple(x.shape)}({x.dtype}), w={tuple(weight_bf16.shape)}({weight_bf16.dtype}), "
            f"bias={tuple(bias.shape)}({bias.dtype}), topk={topk}"
        )

    dev_idx = x.device.index if x.device.index is not None else torch.cuda.current_device()
    _, fns = _get_device_module(dev_idx)
    stream = torch.cuda.current_stream(dev_idx).cuda_stream
    capturing = torch.cuda.is_current_stream_capturing()
    partial, flag = _get_stream_buffers(dev_idx, stream, capturing=capturing)
    rs = float(route_scale)
    eps = float(norm_eps)
    if torch.cuda.current_device() != dev_idx:
        with torch.cuda.device(dev_idx):
            _dispatch_sm90_gate_gemv_pack(
                fns, stream, e, m, h, x, weight_bf16, bias,
                out_fp8, out_sf, out_weights, out_indices, partial, flag, rs, eps,
            )
        return
    _dispatch_sm90_gate_gemv_pack(
        fns, stream, e, m, h, x, weight_bf16, bias,
        out_fp8, out_sf, out_weights, out_indices, partial, flag, rs, eps,
    )


def _dispatch_sm90_gate_gemv_pack(
    fns: Dict[str, object],
    stream: int,
    e: int,
    m: int,
    h: int,
    x: torch.Tensor,
    weight_bf16: torch.Tensor,
    bias: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_weights: torch.Tensor,
    out_indices: torch.Tensor,
    partial: torch.Tensor,
    flag: torch.Tensor,
    rs: float,
    eps: float,
) -> None:

    if e == 384:
        if m <= 4:
            # 1-kernel Zero-SMEM MMA + concurrent pack + 4-way 3-warp TopK
            num_pack = max(4, min(16, (m * 40 + 11) // 12))
            _launch_1k(
                fns["s384_1k_sk5_w12"],
                48 * 5 + num_pack,
                384,
                stream,
                x,
                weight_bf16,
                bias,
                out_fp8,
                out_sf,
                out_weights,
                out_indices,
                partial,
                flag,
                m,
                rs,
                eps,
                num_pack,
            )
            return
        num_pack = max(16, min(256, (m * 40 + 5) // 6))
        if m <= 16:
            _launch_k1(
                fns["z384_m1_n1_sk8_w10_k1"],
                48 * 8,
                320,
                0,
                stream,
                x,
                weight_bf16,
                partial,
                m,
            )
            _launch_k2_pack(
                fns["tkp384_sk8"],
                m + num_pack,
                96,
                stream,
                partial,
                bias,
                x,
                out_fp8,
                out_sf,
                out_weights,
                out_indices,
                m,
                h,
                rs,
                eps,
                num_pack,
            )
        elif m <= 32:
            _launch_k1(
                fns["z384_m2_n1_sk5_w8_k1"],
                48 * 5,
                256,
                0,
                stream,
                x,
                weight_bf16,
                partial,
                m,
            )
            _launch_k2_pack(
                fns["tkp384_sk5"],
                m + num_pack,
                96,
                stream,
                partial,
                bias,
                x,
                out_fp8,
                out_sf,
                out_weights,
                out_indices,
                m,
                h,
                rs,
                eps,
                num_pack,
            )
        elif m <= 64:
            _launch_k1(
                fns["wg384_64_sk20_s3"],
                6 * 20,
                128,
                _SMEM_WG64_S3,
                stream,
                x,
                weight_bf16,
                partial,
                m,
            )
            _launch_k2_pack(
                fns["tkp384_sk20"],
                m + num_pack,
                96,
                stream,
                partial,
                bias,
                x,
                out_fp8,
                out_sf,
                out_weights,
                out_indices,
                m,
                h,
                rs,
                eps,
                num_pack,
            )
        else:  # 65 <= m <= 128
            _launch_k1(
                fns["wg384_128_sk20_s3"],
                6 * 20,
                256,
                _SMEM_WG128_S3,
                stream,
                x,
                weight_bf16,
                partial,
                m,
            )
            _launch_k2_pack(
                fns["tkp384_sk20"],
                m + num_pack,
                96,
                stream,
                partial,
                bias,
                x,
                out_fp8,
                out_sf,
                out_weights,
                out_indices,
                m,
                h,
                rs,
                eps,
                num_pack,
            )
        return

    # e == 128, topk == 3
    if m <= 8:
        num_pack = max(4, min(16, (m * 40 + 9) // 10))
        _launch_1k(
            fns["s128_1k_sk8_w10"],
            16 * 8 + num_pack,
            320,
            stream,
            x,
            weight_bf16,
            bias,
            out_fp8,
            out_sf,
            out_weights,
            out_indices,
            partial,
            flag,
            m,
            rs,
            eps,
            num_pack,
        )
        return
    row_ctas = (m + 3) // 4
    num_pack = max(16, min(256, (m * 40 + 7) // 8))
    if m <= 16:
        _launch_k1(
            fns["z128_m1_n1_sk5_w8_k1"],
            16 * 5,
            256,
            0,
            stream,
            x,
            weight_bf16,
            partial,
            m,
        )
        _launch_k2_pack(
            fns["tkp128_sk5"],
            row_ctas + num_pack,
            128,
            stream,
            partial,
            bias,
            x,
            out_fp8,
            out_sf,
            out_weights,
            out_indices,
            m,
            h,
            rs,
            eps,
            num_pack,
        )
    elif m <= 32:
        m_tiles = (m + 15) // 16
        _launch_k1(
            fns["z128_m1_n2_sk5_w8_k1"],
            m_tiles * 8 * 5,
            256,
            0,
            stream,
            x,
            weight_bf16,
            partial,
            m,
        )
        _launch_k2_pack(
            fns["tkp128_sk5"],
            row_ctas + num_pack,
            128,
            stream,
            partial,
            bias,
            x,
            out_fp8,
            out_sf,
            out_weights,
            out_indices,
            m,
            h,
            rs,
            eps,
            num_pack,
        )
    elif m <= 64:
        _launch_k1(
            fns["wg128_64_sk20_s3"],
            2 * 20,
            128,
            _SMEM_WG64_S3,
            stream,
            x,
            weight_bf16,
            partial,
            m,
        )
        _launch_k2_pack(
            fns["tkp128_sk20"],
            row_ctas + num_pack,
            128,
            stream,
            partial,
            bias,
            x,
            out_fp8,
            out_sf,
            out_weights,
            out_indices,
            m,
            h,
            rs,
            eps,
            num_pack,
        )
    else:  # 65 <= m <= 128
        _launch_k1(
            fns["wg128_128_sk40_s2"],
            2 * 40,
            256,
            _SMEM_WG128_S2,
            stream,
            x,
            weight_bf16,
            partial,
            m,
        )
        _launch_k2_pack(
            fns["tkp128_sk40"],
            row_ctas + num_pack,
            128,
            stream,
            partial,
            bias,
            x,
            out_fp8,
            out_sf,
            out_weights,
            out_indices,
            m,
            h,
            rs,
            eps,
            num_pack,
        )
