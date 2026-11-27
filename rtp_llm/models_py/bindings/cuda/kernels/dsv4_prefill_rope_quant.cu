// Vectorized CUDA kernels for DeepSeek-V4 Flash Prefill SmallOps:
//   1. dsv4_fused_rmsnorm_rope
//   2. dsv4_fused_inv_rope_fp8_quant
//   3. dsv4_indexer_q_rope_fp8_fold

#include "rtp_llm/models_py/bindings/cuda/kernels/dsv4_prefill_rope_quant.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#ifndef USE_ROCM
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#endif

namespace torch_ext {

#ifndef USE_ROCM
namespace {

struct FastStrideDiv {
    int32_t  shift;
    bool     fit_u32;
    int64_t  stride;
    uint32_t d_u32;

    __device__ __forceinline__ int64_t div(int64_t x) const {
        if (shift >= 0) {
            return x >> shift;
        }
        if (fit_u32) {
            return static_cast<int64_t>(static_cast<uint32_t>(x) / d_u32);
        }
        return x / stride;
    }
};

inline FastStrideDiv make_fast_div(int64_t max_dividend, int64_t divisor) {
    FastStrideDiv fd{};
    fd.stride  = divisor;
    fd.d_u32   = static_cast<uint32_t>(divisor);
    fd.fit_u32 = (max_dividend >= 0) && (max_dividend <= 0xFFFFFFFFLL) && (divisor > 0) &&
                 (divisor <= 0xFFFFFFFFLL);
    if (divisor > 0 && (divisor & (divisor - 1)) == 0) {
        int s = 0;
        int64_t v = divisor;
        while ((v & 1LL) == 0) {
            v >>= 1;
            ++s;
        }
        fd.shift = s;
    } else {
        fd.shift = -1;
    }
    return fd;
}

__device__ __forceinline__ float2 bf16x2_raw_to_float2(uint32_t raw) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __nv_bfloat162 b2;
    memcpy(&b2, &raw, sizeof(b2));
    return __bfloat1622float2(b2);
#else
    __nv_bfloat16_raw lo, hi;
    lo.x = static_cast<unsigned short>(raw & 0xFFFFu);
    hi.x = static_cast<unsigned short>(raw >> 16);
    return make_float2(__bfloat162float(__nv_bfloat16(lo)), __bfloat162float(__nv_bfloat16(hi)));
#endif
}

__device__ __forceinline__ uint32_t float2_to_bf16x2_raw(float a, float b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __nv_bfloat162 b2 = __floats2bfloat162_rn(a, b);
    uint32_t       raw;
    memcpy(&raw, &b2, sizeof(raw));
    return raw;
#else
    __nv_bfloat16  ba = __float2bfloat16_rn(a);
    __nv_bfloat16  bb = __float2bfloat16_rn(b);
    unsigned short ua, ub;
    memcpy(&ua, &ba, sizeof(ua));
    memcpy(&ub, &bb, sizeof(ub));
    return static_cast<uint32_t>(ua) | (static_cast<uint32_t>(ub) << 16);
#endif
}

__device__ __forceinline__ float2 fp16x2_raw_to_float2(uint32_t raw) {
    __half2 h2;
    memcpy(&h2, &raw, sizeof(h2));
    return __half22float2(h2);
}

__device__ __forceinline__ uint32_t float2_to_fp16x2_raw(float a, float b) {
    __half2  h2 = __floats2half2_rn(a, b);
    uint32_t raw;
    memcpy(&raw, &h2, sizeof(raw));
    return raw;
}

template<typename T>
struct Vec8Traits;

template<>
struct Vec8Traits<__nv_bfloat16> {
    static __device__ __forceinline__ void unpack_u4(uint4 v, float out[8]) {
        const float2 p0 = bf16x2_raw_to_float2(v.x);
        const float2 p1 = bf16x2_raw_to_float2(v.y);
        const float2 p2 = bf16x2_raw_to_float2(v.z);
        const float2 p3 = bf16x2_raw_to_float2(v.w);
        out[0]          = p0.x;
        out[1]          = p0.y;
        out[2]          = p1.x;
        out[3]          = p1.y;
        out[4]          = p2.x;
        out[5]          = p2.y;
        out[6]          = p3.x;
        out[7]          = p3.y;
    }

    static __device__ __forceinline__ uint4 pack_u4(const float in[8]) {
        return make_uint4(float2_to_bf16x2_raw(in[0], in[1]),
                          float2_to_bf16x2_raw(in[2], in[3]),
                          float2_to_bf16x2_raw(in[4], in[5]),
                          float2_to_bf16x2_raw(in[6], in[7]));
    }
};

template<>
struct Vec8Traits<__half> {
    static __device__ __forceinline__ void unpack_u4(uint4 v, float out[8]) {
        const float2 p0 = fp16x2_raw_to_float2(v.x);
        const float2 p1 = fp16x2_raw_to_float2(v.y);
        const float2 p2 = fp16x2_raw_to_float2(v.z);
        const float2 p3 = fp16x2_raw_to_float2(v.w);
        out[0]          = p0.x;
        out[1]          = p0.y;
        out[2]          = p1.x;
        out[3]          = p1.y;
        out[4]          = p2.x;
        out[5]          = p2.y;
        out[6]          = p3.x;
        out[7]          = p3.y;
    }

    static __device__ __forceinline__ uint4 pack_u4(const float in[8]) {
        return make_uint4(float2_to_fp16x2_raw(in[0], in[1]),
                          float2_to_fp16x2_raw(in[2], in[3]),
                          float2_to_fp16x2_raw(in[4], in[5]),
                          float2_to_fp16x2_raw(in[6], in[7]));
    }
};

template<typename WeightT>
__device__ __forceinline__ void load_weight8(const WeightT* __restrict__ w_ptr, int col, float w_out[8]);

template<>
__device__ __forceinline__ void load_weight8<__nv_bfloat16>(const __nv_bfloat16* __restrict__ w_ptr,
                                                            int col,
                                                            float w_out[8]) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(w_ptr + col));
    Vec8Traits<__nv_bfloat16>::unpack_u4(u, w_out);
}

template<>
__device__ __forceinline__ void load_weight8<__half>(const __half* __restrict__ w_ptr, int col, float w_out[8]) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(w_ptr + col));
    Vec8Traits<__half>::unpack_u4(u, w_out);
}

template<>
__device__ __forceinline__ void load_weight8<float>(const float* __restrict__ w_ptr, int col, float w_out[8]) {
    const float4 a = __ldg(reinterpret_cast<const float4*>(w_ptr + col));
    const float4 b = __ldg(reinterpret_cast<const float4*>(w_ptr + col + 4));
    w_out[0]       = a.x;
    w_out[1]       = a.y;
    w_out[2]       = a.z;
    w_out[3]       = a.w;
    w_out[4]       = b.x;
    w_out[5]       = b.y;
    w_out[6]       = b.z;
    w_out[7]       = b.w;
}

template<typename PosT>
__device__ __forceinline__ int64_t resolve_freq_row(const PosT* __restrict__ pos_ptr, int64_t freq_idx, int64_t max_pos) {
    if constexpr (std::is_same_v<PosT, void>) {
        (void)pos_ptr;
        (void)max_pos;
        return freq_idx;
    } else {
        int64_t p = static_cast<int64_t>(__ldg(pos_ptr + freq_idx));
        p         = p < 0 ? 0 : p;
        p         = p >= max_pos ? (max_pos - 1) : p;
        return p;
    }
}

// ---------------------------------------------------------------------------
// 1. dsv4_fused_rmsnorm_rope
//
// Name contains "fused_rmsnorm_rope" to match existing profiler regexes.
// ---------------------------------------------------------------------------
template<typename ScalarT, typename WeightT, typename PosT, bool HAS_WEIGHT, bool INVERSE>
__global__ void dsv4_fused_rmsnorm_rope_d512_kernel(const ScalarT* __restrict__ x_ptr,
                                                    const WeightT* __restrict__ w_ptr,
                                                    const float2* __restrict__ freqs_ptr,
                                                    const PosT* __restrict__ pos_ptr,
                                                    ScalarT* __restrict__ out_ptr,
                                                    int64_t       N,
                                                    FastStrideDiv fdiv,
                                                    int64_t       max_pos,
                                                    int           rd,
                                                    float         eps) {
    const int     lane     = threadIdx.x & 31;
    const int64_t warp_gid = static_cast<int64_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (warp_gid >= N) {
        if constexpr (!HAS_WEIGHT) {
            (void)w_ptr;
        }
        return;
    }

    const int col0        = lane * 8;
    const int col1        = 256 + lane * 8;
    const int nope_offset = 512 - rd;
    const int rd_half     = rd >> 1;

    const ScalarT* x_row = x_ptr + warp_gid * 512;
    const uint4    u0    = *reinterpret_cast<const uint4*>(x_row + col0);
    const uint4    u1    = *reinterpret_cast<const uint4*>(x_row + col1);

    float f0[8], f1[8];
    Vec8Traits<ScalarT>::unpack_u4(u0, f0);
    Vec8Traits<ScalarT>::unpack_u4(u1, f1);

    float sum_sq = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sum_sq += f0[i] * f0[i] + f1[i] * f1[i];
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, mask);
    }
    const float inv = rsqrtf(sum_sq * (1.0f / 512.0f) + eps);

    if constexpr (HAS_WEIGHT) {
        float w0[8], w1[8];
        load_weight8(w_ptr, col0, w0);
        load_weight8(w_ptr, col1, w1);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            f0[i] = f0[i] * inv * w0[i];
            f1[i] = f1[i] * inv * w1[i];
        }
    } else {
        (void)w_ptr;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            f0[i] *= inv;
            f1[i] *= inv;
        }
    }

    if (rd > 0) {
        const int64_t freq_idx = fdiv.div(warp_gid);
        const int64_t freq_row = resolve_freq_row(pos_ptr, freq_idx, max_pos);
        const float2* f_row    = freqs_ptr + freq_row * rd_half;

        if (nope_offset == 448) {
            if (lane >= 24) {
                const int    pair_base = (lane - 24) * 4;
                const float4 fq0       = __ldg(reinterpret_cast<const float4*>(f_row + pair_base));
                const float4 fq1       = __ldg(reinterpret_cast<const float4*>(f_row + pair_base + 2));
                const float2 cs[4]     = {{fq0.x, fq0.y}, {fq0.z, fq0.w}, {fq1.x, fq1.y}, {fq1.z, fq1.w}};
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const float c = cs[p].x;
                    const float s = INVERSE ? -cs[p].y : cs[p].y;
                    const float e = f1[2 * p];
                    const float o = f1[2 * p + 1];
                    f1[2 * p]     = e * c - o * s;
                    f1[2 * p + 1] = e * s + o * c;
                }
            }
        } else {
            if (col0 + 7 >= nope_offset) {
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const int c_idx = col0 + 2 * p;
                    if (c_idx >= nope_offset) {
                        const float2 cs = __ldg(f_row + ((c_idx - nope_offset) >> 1));
                        const float  s  = INVERSE ? -cs.y : cs.y;
                        const float  e  = f0[2 * p];
                        const float  o  = f0[2 * p + 1];
                        f0[2 * p]       = e * cs.x - o * s;
                        f0[2 * p + 1]   = e * s + o * cs.x;
                    }
                }
            }
            if (col1 + 7 >= nope_offset) {
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const int c_idx = col1 + 2 * p;
                    if (c_idx >= nope_offset) {
                        const float2 cs = __ldg(f_row + ((c_idx - nope_offset) >> 1));
                        const float  s  = INVERSE ? -cs.y : cs.y;
                        const float  e  = f1[2 * p];
                        const float  o  = f1[2 * p + 1];
                        f1[2 * p]       = e * cs.x - o * s;
                        f1[2 * p + 1]   = e * s + o * cs.x;
                    }
                }
            }
        }
    }

    ScalarT* out_row                          = out_ptr + warp_gid * 512;
    *reinterpret_cast<uint4*>(out_row + col0) = Vec8Traits<ScalarT>::pack_u4(f0);
    *reinterpret_cast<uint4*>(out_row + col1) = Vec8Traits<ScalarT>::pack_u4(f1);
}

template<typename ScalarT, typename WeightT, typename PosT, bool HAS_WEIGHT, bool INVERSE>
__global__ void dsv4_fused_rmsnorm_rope_d128_kernel(const ScalarT* __restrict__ x_ptr,
                                                    const WeightT* __restrict__ w_ptr,
                                                    const float2* __restrict__ freqs_ptr,
                                                    const PosT* __restrict__ pos_ptr,
                                                    ScalarT* __restrict__ out_ptr,
                                                    int64_t       N,
                                                    FastStrideDiv fdiv,
                                                    int64_t       max_pos,
                                                    int           rd,
                                                    float         eps) {
    const int     sub_lane    = threadIdx.x & 15;
    const int     half_in_blk = threadIdx.x >> 4;
    const int64_t row         = static_cast<int64_t>(blockIdx.x) * (blockDim.x >> 4) + half_in_blk;
    if (row >= N) {
        if constexpr (!HAS_WEIGHT) {
            (void)w_ptr;
        }
        return;
    }

    const int col         = sub_lane * 8;
    const int nope_offset = 128 - rd;
    const int rd_half     = rd >> 1;

    const uint4 u = *reinterpret_cast<const uint4*>(x_ptr + row * 128 + col);
    float       f[8];
    Vec8Traits<ScalarT>::unpack_u4(u, f);

    float sum_sq = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sum_sq += f[i] * f[i];
    }
    sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, 8);
    sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, 4);
    sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, 2);
    sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, 1);
    const float inv = rsqrtf(sum_sq * (1.0f / 128.0f) + eps);

    if constexpr (HAS_WEIGHT) {
        float w[8];
        load_weight8(w_ptr, col, w);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            f[i] = f[i] * inv * w[i];
        }
    } else {
        (void)w_ptr;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            f[i] *= inv;
        }
    }

    if (rd > 0 && col + 7 >= nope_offset) {
        const int64_t freq_idx = fdiv.div(row);
        const int64_t freq_row = resolve_freq_row(pos_ptr, freq_idx, max_pos);
        const float2* f_row    = freqs_ptr + freq_row * rd_half;
        if (nope_offset == 64) {
            const int    pair_base = (sub_lane - 8) * 4;
            const float4 fq0       = __ldg(reinterpret_cast<const float4*>(f_row + pair_base));
            const float4 fq1       = __ldg(reinterpret_cast<const float4*>(f_row + pair_base + 2));
            const float2 cs[4]     = {{fq0.x, fq0.y}, {fq0.z, fq0.w}, {fq1.x, fq1.y}, {fq1.z, fq1.w}};
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float c = cs[p].x;
                const float s = INVERSE ? -cs[p].y : cs[p].y;
                const float e = f[2 * p];
                const float o = f[2 * p + 1];
                f[2 * p]      = e * c - o * s;
                f[2 * p + 1]  = e * s + o * c;
            }
        } else {
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int c_idx = col + 2 * p;
                if (c_idx >= nope_offset) {
                    const float2 cs = __ldg(f_row + ((c_idx - nope_offset) >> 1));
                    const float  s  = INVERSE ? -cs.y : cs.y;
                    const float  e  = f[2 * p];
                    const float  o  = f[2 * p + 1];
                    f[2 * p]        = e * cs.x - o * s;
                    f[2 * p + 1]    = e * s + o * cs.x;
                }
            }
        }
    }

    *reinterpret_cast<uint4*>(out_ptr + row * 128 + col) = Vec8Traits<ScalarT>::pack_u4(f);
}

template<typename ScalarT, typename WeightT, typename PosT, bool HAS_WEIGHT, bool INVERSE>
__global__ void dsv4_fused_rmsnorm_rope_vec_kernel(const ScalarT* __restrict__ x_ptr,
                                                   const WeightT* __restrict__ w_ptr,
                                                   const float2* __restrict__ freqs_ptr,
                                                   const PosT* __restrict__ pos_ptr,
                                                   ScalarT* __restrict__ out_ptr,
                                                   int64_t       N,
                                                   int           D,
                                                   FastStrideDiv fdiv,
                                                   int64_t       max_pos,
                                                   int           rd,
                                                   float         eps) {
    constexpr int MAX_PASSES    = 4;
    const int     lane          = threadIdx.x & 31;
    const int64_t warps_per_blk = blockDim.x >> 5;
    const int64_t total_warps   = static_cast<int64_t>(gridDim.x) * warps_per_blk;
    const int64_t warp_gid      = static_cast<int64_t>(blockIdx.x) * warps_per_blk + (threadIdx.x >> 5);
    const int     passes        = (D + 255) >> 8;
    const int     nope_offset   = D - rd;
    const int     rd_half       = rd >> 1;
    const float   inv_D         = 1.0f / static_cast<float>(D);

    for (int64_t row = warp_gid; row < N; row += total_warps) {
        const ScalarT* x_row = x_ptr + row * D;
        float          vals[MAX_PASSES][8];
        float          sum_sq = 0.0f;

        for (int p = 0; p < passes; ++p) {
            const int col = p * 256 + lane * 8;
            if (col < D) {
                const uint4 u = *reinterpret_cast<const uint4*>(x_row + col);
                Vec8Traits<ScalarT>::unpack_u4(u, vals[p]);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    sum_sq += vals[p][i] * vals[p][i];
                }
            }
        }
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, mask);
        }
        const float inv = rsqrtf(sum_sq * inv_D + eps);

        const float2* f_row = nullptr;
        if (rd > 0) {
            const int64_t freq_idx = fdiv.div(row);
            const int64_t freq_row = resolve_freq_row(pos_ptr, freq_idx, max_pos);
            f_row                  = freqs_ptr + freq_row * rd_half;
        }

        ScalarT* out_row = out_ptr + row * D;
        for (int p = 0; p < passes; ++p) {
            const int col = p * 256 + lane * 8;
            if (col < D) {
                if constexpr (HAS_WEIGHT) {
                    float w[8];
                    load_weight8(w_ptr, col, w);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        vals[p][i] = vals[p][i] * inv * w[i];
                    }
                } else {
                    (void)w_ptr;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        vals[p][i] *= inv;
                    }
                }
                if (rd > 0 && col + 7 >= nope_offset) {
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int c_idx = col + 2 * k;
                        if (c_idx >= nope_offset) {
                            const float2 cs = __ldg(f_row + ((c_idx - nope_offset) >> 1));
                            const float  s  = INVERSE ? -cs.y : cs.y;
                            const float  e  = vals[p][2 * k];
                            const float  o  = vals[p][2 * k + 1];
                            vals[p][2 * k]     = e * cs.x - o * s;
                            vals[p][2 * k + 1] = e * s + o * cs.x;
                        }
                    }
                }
                *reinterpret_cast<uint4*>(out_row + col) = Vec8Traits<ScalarT>::pack_u4(vals[p]);
            }
        }
    }
}

template<typename ScalarT, typename WeightT, typename PosT, bool HAS_WEIGHT, bool INVERSE>
void launch_rmsnorm_rope_typed(const ScalarT* x_ptr,
                               const WeightT* w_ptr,
                               const float2*  freqs_ptr,
                               const PosT*    pos_ptr,
                               ScalarT*       out_ptr,
                               int64_t        N,
                               int            D,
                               int64_t        freq_stride_n,
                               int64_t        max_pos,
                               int            rd,
                               float          eps,
                               int            sm_count,
                               cudaStream_t   stream) {
    const FastStrideDiv fdiv = make_fast_div(N, freq_stride_n);
    if (D == 512) {
        int block_dim = 128;
        int grid_dim  = 0;
        if (N <= 256) {
            block_dim = 32;
            grid_dim  = static_cast<int>(N);
        } else if (N <= 4096) {
            block_dim = 64;
            grid_dim  = static_cast<int>((N + 1) >> 1);
        } else {
            block_dim = 128;
            grid_dim  = static_cast<int>((N + 3) >> 2);
        }
        dsv4_fused_rmsnorm_rope_d512_kernel<ScalarT, WeightT, PosT, HAS_WEIGHT, INVERSE>
            <<<grid_dim, block_dim, 0, stream>>>(
                x_ptr, w_ptr, freqs_ptr, pos_ptr, out_ptr, N, fdiv, max_pos, rd, eps);
    } else if (D == 128) {
        int block_dim = 128;
        int grid_dim  = 0;
        if (N <= 256) {
            block_dim = 32;
            grid_dim  = static_cast<int>((N + 1) >> 1);
        } else if (N <= 4096) {
            block_dim = 64;
            grid_dim  = static_cast<int>((N + 3) >> 2);
        } else {
            block_dim = 128;
            grid_dim  = static_cast<int>((N + 7) >> 3);
        }
        dsv4_fused_rmsnorm_rope_d128_kernel<ScalarT, WeightT, PosT, HAS_WEIGHT, INVERSE>
            <<<grid_dim, block_dim, 0, stream>>>(
                x_ptr, w_ptr, freqs_ptr, pos_ptr, out_ptr, N, fdiv, max_pos, rd, eps);
    } else {
        const int     block_dim = (N <= 256) ? 32 : 128;
        const int64_t warps_blk = block_dim >> 5;
        const int     max_blks  = std::max(1, sm_count) * 16;
        const int64_t need      = (N + warps_blk - 1) / warps_blk;
        const int     grid_dim  = static_cast<int>(std::min<int64_t>(need, max_blks));
        dsv4_fused_rmsnorm_rope_vec_kernel<ScalarT, WeightT, PosT, HAS_WEIGHT, INVERSE>
            <<<grid_dim, block_dim, 0, stream>>>(
                x_ptr, w_ptr, freqs_ptr, pos_ptr, out_ptr, N, D, fdiv, max_pos, rd, eps);
    }
}

template<typename ScalarT, typename WeightT, bool HAS_WEIGHT, bool INVERSE>
void dispatch_rmsnorm_rope_pos(const ScalarT*                      x_ptr,
                               const WeightT*                      w_ptr,
                               const float2*                       freqs_ptr,
                               const std::optional<torch::Tensor>& position_ids,
                               ScalarT*                            out_ptr,
                               int64_t                             N,
                               int                                 D,
                               int64_t                             freq_stride_n,
                               int64_t                             max_pos,
                               int                                 rd,
                               float                               eps,
                               int                                 sm_count,
                               cudaStream_t                        stream) {
    if (!position_ids.has_value()) {
        launch_rmsnorm_rope_typed<ScalarT, WeightT, void, HAS_WEIGHT, INVERSE>(
            x_ptr, w_ptr, freqs_ptr, static_cast<const void*>(nullptr), out_ptr, N, D, freq_stride_n, max_pos, rd, eps, sm_count, stream);
    } else if (position_ids->dtype() == torch::kInt32) {
        launch_rmsnorm_rope_typed<ScalarT, WeightT, int32_t, HAS_WEIGHT, INVERSE>(
            x_ptr, w_ptr, freqs_ptr, position_ids->data_ptr<int32_t>(), out_ptr, N, D, freq_stride_n, max_pos, rd, eps, sm_count, stream);
    } else if (position_ids->dtype() == torch::kInt64) {
        launch_rmsnorm_rope_typed<ScalarT, WeightT, int64_t, HAS_WEIGHT, INVERSE>(
            x_ptr, w_ptr, freqs_ptr, position_ids->data_ptr<int64_t>(), out_ptr, N, D, freq_stride_n, max_pos, rd, eps, sm_count, stream);
    } else {
        TORCH_CHECK(false, "position_ids must be int32 or int64");
    }
}

template<typename ScalarT, bool INVERSE>
void dispatch_rmsnorm_rope_weight(const ScalarT*                      x_ptr,
                                  const std::optional<torch::Tensor>& weight,
                                  const float2*                       freqs_ptr,
                                  const std::optional<torch::Tensor>& position_ids,
                                  ScalarT*                            out_ptr,
                                  int64_t                             N,
                                  int                                 D,
                                  int64_t                             freq_stride_n,
                                  int64_t                             max_pos,
                                  int                                 rd,
                                  float                               eps,
                                  int                                 sm_count,
                                  cudaStream_t                        stream) {
    if (!weight.has_value()) {
        dispatch_rmsnorm_rope_pos<ScalarT, float, false, INVERSE>(
            x_ptr, static_cast<const float*>(nullptr), freqs_ptr, position_ids, out_ptr, N, D, freq_stride_n, max_pos, rd, eps, sm_count, stream);
    } else if (weight->dtype() == torch::kBFloat16) {
        dispatch_rmsnorm_rope_pos<ScalarT, __nv_bfloat16, true, INVERSE>(
            x_ptr,
            reinterpret_cast<const __nv_bfloat16*>(weight->data_ptr<at::BFloat16>()),
            freqs_ptr,
            position_ids,
            out_ptr,
            N,
            D,
            freq_stride_n,
            max_pos,
            rd,
            eps,
            sm_count,
            stream);
    } else if (weight->dtype() == torch::kHalf) {
        dispatch_rmsnorm_rope_pos<ScalarT, __half, true, INVERSE>(
            x_ptr,
            reinterpret_cast<const __half*>(weight->data_ptr<at::Half>()),
            freqs_ptr,
            position_ids,
            out_ptr,
            N,
            D,
            freq_stride_n,
            max_pos,
            rd,
            eps,
            sm_count,
            stream);
    } else if (weight->dtype() == torch::kFloat32) {
        dispatch_rmsnorm_rope_pos<ScalarT, float, true, INVERSE>(
            x_ptr, weight->data_ptr<float>(), freqs_ptr, position_ids, out_ptr, N, D, freq_stride_n, max_pos, rd, eps, sm_count, stream);
    } else {
        TORCH_CHECK(false, "weight dtype must be bfloat16, float16, or float32");
    }
}

void dsv4_fused_rmsnorm_rope_impl(const torch::Tensor&                x,
                                  const std::optional<torch::Tensor>& weight,
                                  const torch::Tensor&                freqs_ri,
                                  const std::optional<torch::Tensor>& position_ids,
                                  torch::Tensor&                      out,
                                  int64_t                             freq_stride_n,
                                  int64_t                             rope_head_dim,
                                  double                              eps,
                                  bool                                inverse) {
    TORCH_CHECK(x.is_cuda() && out.is_cuda(), "x and out must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "x and out must be contiguous");
    TORCH_CHECK(x.dim() == 2 && out.sizes() == x.sizes(), "x and out must have matching [N, D] shapes");
    TORCH_CHECK(x.dtype() == out.dtype(), "x and out dtypes must match");
    TORCH_CHECK(freqs_ri.is_cuda() && freqs_ri.is_contiguous() && freqs_ri.dtype() == torch::kFloat32,
                "freqs_ri must be contiguous CUDA float32");

    const int64_t N = x.size(0);
    const int64_t D = x.size(1);
    if (N == 0) {
        return;
    }

    TORCH_CHECK(D > 0 && D <= 1024 && (D % 8 == 0), "D must be multiple of 8 and <= 1024");
    TORCH_CHECK(rope_head_dim >= 0 && rope_head_dim <= D && (rope_head_dim % 2 == 0),
                "invalid rope_head_dim");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0 &&
                    (reinterpret_cast<uintptr_t>(out.data_ptr()) & 15) == 0,
                "x and out pointers must be 16-byte aligned");
    if (weight.has_value()) {
        TORCH_CHECK(weight->is_cuda() && weight->is_contiguous() && weight->dim() == 1 && weight->size(0) == D,
                    "weight must be contiguous 1D [D] CUDA tensor");
        TORCH_CHECK((reinterpret_cast<uintptr_t>(weight->data_ptr()) & 15) == 0,
                    "weight pointer must be 16-byte aligned");
    }

    const float2* freqs_ptr = reinterpret_cast<const float2*>(freqs_ri.data_ptr<float>());
    TORCH_CHECK((reinterpret_cast<uintptr_t>(freqs_ptr) & 15) == 0 || rope_head_dim != 64,
                "freqs_ri pointer must be 16-byte aligned");
    const int64_t max_pos  = freqs_ri.size(0);
    const int     rd       = static_cast<int>(rope_head_dim);
    const float   eps_f    = static_cast<float>(eps);
    const int     sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    cudaStream_t  stream   = at::cuda::getCurrentCUDAStream();

    if (x.dtype() == torch::kBFloat16) {
        const __nv_bfloat16* x_p = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
        __nv_bfloat16*       o_p = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
        if (inverse) {
            dispatch_rmsnorm_rope_weight<__nv_bfloat16, true>(
                x_p, weight, freqs_ptr, position_ids, o_p, N, static_cast<int>(D), freq_stride_n, max_pos, rd, eps_f, sm_count, stream);
        } else {
            dispatch_rmsnorm_rope_weight<__nv_bfloat16, false>(
                x_p, weight, freqs_ptr, position_ids, o_p, N, static_cast<int>(D), freq_stride_n, max_pos, rd, eps_f, sm_count, stream);
        }
    } else if (x.dtype() == torch::kHalf) {
        const __half* x_p = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
        __half*       o_p = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
        if (inverse) {
            dispatch_rmsnorm_rope_weight<__half, true>(
                x_p, weight, freqs_ptr, position_ids, o_p, N, static_cast<int>(D), freq_stride_n, max_pos, rd, eps_f, sm_count, stream);
        } else {
            dispatch_rmsnorm_rope_weight<__half, false>(
                x_p, weight, freqs_ptr, position_ids, o_p, N, static_cast<int>(D), freq_stride_n, max_pos, rd, eps_f, sm_count, stream);
        }
    } else {
        TORCH_CHECK(false, "dsv4_fused_rmsnorm_rope CUDA only supports bfloat16 and float16");
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dsv4_fused_rmsnorm_rope launch failed: ", cudaGetErrorString(err));
}

// ---------------------------------------------------------------------------
// 2. dsv4_fused_inv_rope_fp8_quant (D == 512, RD == 64, quant_group_size == 128)
//
// Name contains "fused_inv_rope_fp8_quant" to match existing profiler regexes.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t compute_ue8m0_quant_u2(const float f[8],
                                                           float       block_absmax,
                                                           float       fp8_max_f,
                                                           float       eps_f,
                                                           float       inv_fp8_max_f,
                                                           uint2&      out_u2) {
    const float    bmax      = fmaxf(block_absmax, eps_f);
    const float    s_raw     = bmax * inv_fp8_max_f;
    const uint32_t raw_bits  = __float_as_uint(s_raw);
    uint32_t       exp       = ((raw_bits >> 23) & 0xFFu) + (((raw_bits & 0x7FFFFFu) != 0u) ? 1u : 0u);
    exp                      = min(max(exp, 1u), 254u);
    const uint32_t inv_bits  = (exp == 254u) ? (1u << 22) : ((254u - exp) << 23);
    const float    inv_scale = __uint_as_float(inv_bits);

    __nv_fp8x2_storage_t q[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float v0 = fminf(fmaxf(f[2 * i] * inv_scale, -fp8_max_f), fp8_max_f);
        const float v1 = fminf(fmaxf(f[2 * i + 1] * inv_scale, -fp8_max_f), fp8_max_f);
        q[i]           = __nv_cvt_float2_to_fp8x2(make_float2(v0, v1), __NV_SATFINITE, __NV_E4M3);
    }
    out_u2.x = static_cast<uint32_t>(q[0]) | (static_cast<uint32_t>(q[1]) << 16);
    out_u2.y = static_cast<uint32_t>(q[2]) | (static_cast<uint32_t>(q[3]) << 16);
    return exp;
}

__device__ __forceinline__ float halfwarp_absmax_reduce(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return v;
}

__device__ __forceinline__ uint32_t process_head_d512_rd64(const __nv_bfloat16* __restrict__ o_head,
                                                           uint8_t* __restrict__ fp8_head,
                                                           const float2* __restrict__ freqs_row,
                                                           float fp8_max_f,
                                                           float eps_f,
                                                           float inv_fp8_max_f) {
    const int lane = threadIdx.x & 31;
    const int col0 = lane * 8;
    const int col1 = 256 + lane * 8;

    // Dual-issue both 512B loads together, unpacking u1 only after Pass 0 to
    // keep live FP32 registers compact.
    const uint4 u0 = __ldg(reinterpret_cast<const uint4*>(o_head + col0));
    const uint4 u1 = __ldg(reinterpret_cast<const uint4*>(o_head + col1));

    float f0[8];
    Vec8Traits<__nv_bfloat16>::unpack_u4(u0, f0);

    // Block 0 (lanes 0..15) and Block 1 (lanes 16..31).
    float max0 = fabsf(f0[0]);
#pragma unroll
    for (int i = 1; i < 8; ++i) {
        max0 = fmaxf(max0, fabsf(f0[i]));
    }
    max0 = halfwarp_absmax_reduce(max0);

    uint2          out0_u2;
    const uint32_t exp0 = compute_ue8m0_quant_u2(f0, max0, fp8_max_f, eps_f, inv_fp8_max_f, out0_u2);
    *reinterpret_cast<uint2*>(fp8_head + col0) = out0_u2;

    float f1[8];
    Vec8Traits<__nv_bfloat16>::unpack_u4(u1, f1);

    // Inverse RoPE in-register ONLY on lanes 24..31 ([448..511]).
    if (lane >= 24) {
        const int    pair_base = (lane - 24) * 4;
        const float4 fq0       = *reinterpret_cast<const float4*>(freqs_row + pair_base);
        const float4 fq1       = *reinterpret_cast<const float4*>(freqs_row + pair_base + 2);
        const float2 cs[4]     = {{fq0.x, fq0.y}, {fq0.z, fq0.w}, {fq1.x, fq1.y}, {fq1.z, fq1.w}};
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float c     = cs[p].x;
            const float s     = cs[p].y;
            const float e     = f1[2 * p];
            const float o_val = f1[2 * p + 1];
            f1[2 * p]         = e * c + o_val * s;
            f1[2 * p + 1]     = o_val * c - e * s;
        }
    }

    // Block 2 (lanes 0..15) and Block 3 (lanes 16..31).
    float max1 = fabsf(f1[0]);
#pragma unroll
    for (int i = 1; i < 8; ++i) {
        max1 = fmaxf(max1, fabsf(f1[i]));
    }
    max1 = halfwarp_absmax_reduce(max1);

    uint2          out1_u2;
    const uint32_t exp1 = compute_ue8m0_quant_u2(f1, max1, fp8_max_f, eps_f, inv_fp8_max_f, out1_u2);
    *reinterpret_cast<uint2*>(fp8_head + col1) = out1_u2;

    const uint32_t e0 = __shfl_sync(0xffffffffu, exp0, 0);
    const uint32_t e1 = __shfl_sync(0xffffffffu, exp0, 16);
    const uint32_t e2 = __shfl_sync(0xffffffffu, exp1, 0);
    const uint32_t e3 = __shfl_sync(0xffffffffu, exp1, 16);
    return e0 | (e1 << 8) | (e2 << 16) | (e3 << 24);
}

// Small-M warp-per-head kernel (saturates SMs when M is small).
template<typename PosT>
__global__ void dsv4_fused_inv_rope_fp8_quant_warp_kernel(const __nv_bfloat16* __restrict__ o_ptr,
                                                          const float2* __restrict__ freqs_ptr,
                                                          const PosT* __restrict__ pos_ptr,
                                                          uint8_t* __restrict__ fp8_ptr,
                                                          int32_t* __restrict__ scale_ptr,
                                                          int64_t       M,
                                                          int64_t       tma_M,
                                                          int64_t       scale_zero_rows,
                                                          int64_t       max_pos,
                                                          FastStrideDiv gh_div,
                                                          FastStrideDiv hpg_div,
                                                          FastStrideDiv q_len_div,
                                                          int           heads_per_group,
                                                          int64_t       o_stride_token,
                                                          int64_t       o_stride_head,
                                                          int64_t       fp8_stride_group,
                                                          int64_t       fp8_stride_token,
                                                          int64_t       scale_stride_group,
                                                          int64_t       scale_stride_k,
                                                          float         fp8_max_f,
                                                          float         eps_f,
                                                          float         inv_fp8_max_f,
                                                          int64_t       total_items) {
    const int     lane     = threadIdx.x & 31;
    const int64_t warp_gid = static_cast<int64_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (warp_gid >= total_items) {
        return;
    }

    const int64_t m       = gh_div.div(warp_gid);
    const int64_t gh_idx  = warp_gid - m * gh_div.stride;
    const int64_t g64     = hpg_div.div(gh_idx);
    const int     h_in_g  = static_cast<int>(gh_idx - g64 * heads_per_group);
    int32_t*      sf_base = scale_ptr + g64 * scale_stride_group +
                       static_cast<int64_t>(h_in_g) * scale_stride_k;

    if (m >= M) {
        if (lane == 0 && m < scale_zero_rows) {
            sf_base[m] = 0;
        }
        return;
    }

    const int64_t b_idx    = q_len_div.div(m);
    const int64_t freq_row = resolve_freq_row(pos_ptr, b_idx, max_pos);
    const float2* f_row    = freqs_ptr + freq_row * 32;

    const __nv_bfloat16* o_head   = o_ptr + m * o_stride_token + gh_idx * o_stride_head;
    uint8_t*             fp8_head = fp8_ptr + g64 * fp8_stride_group + m * fp8_stride_token +
                        static_cast<int64_t>(h_in_g) * 512;

    const uint32_t packed = process_head_d512_rd64(o_head, fp8_head, f_row, fp8_max_f, eps_f, inv_fp8_max_f);
    if (lane == 0) {
        sf_base[m] = static_cast<int32_t>(packed);
        if (m == M - 1) {
            for (int64_t zm = tma_M; zm < scale_zero_rows; ++zm) {
                sf_base[zm] = 0;
            }
        }
    }
}

// Prefill 4-token-tile CTA kernel:
// - blockIdx.x = g, blockIdx.y = tok_tile (when num_tok_tiles <= 65535), so
//   all groups of a token launch in the same wave for sequential DRAM/L2 hits.
// - Pre-issues token-0 u0/u1 loads before __syncthreads() so HBM read latency
//   overlaps with populating shared s_freqs[4][32].
template<typename PosT, bool GROUP_ON_X>
__global__ void dsv4_fused_inv_rope_fp8_quant_4tok_kernel(const __nv_bfloat16* __restrict__ o_ptr,
                                                          const float2* __restrict__ freqs_ptr,
                                                          const PosT* __restrict__ pos_ptr,
                                                          uint8_t* __restrict__ fp8_ptr,
                                                          int32_t* __restrict__ scale_ptr,
                                                          int64_t       M,
                                                          int64_t       scale_zero_rows,
                                                          int64_t       max_pos,
                                                          FastStrideDiv q_len_div,
                                                          int           heads_per_group,
                                                          int64_t       o_stride_token,
                                                          int64_t       o_stride_head,
                                                          int64_t       fp8_stride_group,
                                                          int64_t       fp8_stride_token,
                                                          int64_t       scale_stride_group,
                                                          int64_t       scale_stride_k,
                                                          float         fp8_max_f,
                                                          float         eps_f,
                                                          float         inv_fp8_max_f) {
    __shared__ alignas(16) float2 s_freqs[4][32];

    const int64_t num_tok_tiles = (M + 3) >> 2;
    const int     g             = GROUP_ON_X ? static_cast<int>(blockIdx.x) : static_cast<int>(blockIdx.y);
    const int64_t tok_tile      = GROUP_ON_X ? static_cast<int64_t>(blockIdx.y) : static_cast<int64_t>(blockIdx.x);
    const int64_t m_base        = tok_tile * 4;

    if (threadIdx.x < 128) {
        const int     tk = threadIdx.x >> 5;
        const int     p  = threadIdx.x & 31;
        const int64_t m  = m_base + tk;
        if (m < M) {
            const int64_t b_idx    = q_len_div.div(m);
            const int64_t freq_row = resolve_freq_row(pos_ptr, b_idx, max_pos);
            s_freqs[tk][p]         = __ldg(freqs_ptr + freq_row * 32 + p);
        }
    }

    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    __syncthreads();

    const __nv_bfloat16* o_g_base   = o_ptr + static_cast<int64_t>(g) * heads_per_group * o_stride_head;
    uint8_t*             fp8_g_base = fp8_ptr + static_cast<int64_t>(g) * fp8_stride_group;
    int32_t*             sf_g_base  = scale_ptr + static_cast<int64_t>(g) * scale_stride_group;

    for (int h = warp_id; h < heads_per_group; h += 8) {
        uint32_t packed[4] = {0u, 0u, 0u, 0u};
#pragma unroll
        for (int tk = 0; tk < 4; ++tk) {
            const int64_t m = m_base + tk;
            if (m < M) {
                const __nv_bfloat16* o_head =
                    o_g_base + m * o_stride_token + static_cast<int64_t>(h) * o_stride_head;
                const uint4 u0 = __ldg(reinterpret_cast<const uint4*>(o_head + lane * 8));
                const uint4 u1 = __ldg(reinterpret_cast<const uint4*>(o_head + 256 + lane * 8));
                uint8_t* fp8_head = fp8_g_base + m * fp8_stride_token + static_cast<int64_t>(h) * 512;

                float f0[8];
                Vec8Traits<__nv_bfloat16>::unpack_u4(u0, f0);
                float max0 = fabsf(f0[0]);
#pragma unroll
                for (int i = 1; i < 8; ++i) {
                    max0 = fmaxf(max0, fabsf(f0[i]));
                }
                max0 = halfwarp_absmax_reduce(max0);
                uint2          out0_u2;
                const uint32_t exp0 = compute_ue8m0_quant_u2(f0, max0, fp8_max_f, eps_f, inv_fp8_max_f, out0_u2);
                *reinterpret_cast<uint2*>(fp8_head + lane * 8) = out0_u2;

                float f1[8];
                Vec8Traits<__nv_bfloat16>::unpack_u4(u1, f1);
                if (lane >= 24) {
                    const int    pair_base = (lane - 24) * 4;
                    const float4 fq0       = *reinterpret_cast<const float4*>(s_freqs[tk] + pair_base);
                    const float4 fq1       = *reinterpret_cast<const float4*>(s_freqs[tk] + pair_base + 2);
                    const float2 cs[4]     = {{fq0.x, fq0.y}, {fq0.z, fq0.w}, {fq1.x, fq1.y}, {fq1.z, fq1.w}};
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const float c     = cs[p].x;
                        const float s     = cs[p].y;
                        const float e     = f1[2 * p];
                        const float o_val = f1[2 * p + 1];
                        f1[2 * p]         = e * c + o_val * s;
                        f1[2 * p + 1]     = o_val * c - e * s;
                    }
                }
                float max1 = fabsf(f1[0]);
#pragma unroll
                for (int i = 1; i < 8; ++i) {
                    max1 = fmaxf(max1, fabsf(f1[i]));
                }
                max1 = halfwarp_absmax_reduce(max1);
                uint2          out1_u2;
                const uint32_t exp1 = compute_ue8m0_quant_u2(f1, max1, fp8_max_f, eps_f, inv_fp8_max_f, out1_u2);
                *reinterpret_cast<uint2*>(fp8_head + 256 + lane * 8) = out1_u2;

                const uint32_t e0 = __shfl_sync(0xffffffffu, exp0, 0);
                const uint32_t e1 = __shfl_sync(0xffffffffu, exp0, 16);
                const uint32_t e2 = __shfl_sync(0xffffffffu, exp1, 0);
                const uint32_t e3 = __shfl_sync(0xffffffffu, exp1, 16);
                packed[tk]        = e0 | (e1 << 8) | (e2 << 16) | (e3 << 24);
            }
        }
        if (lane == 0) {
            int32_t* sf_col = sf_g_base + static_cast<int64_t>(h) * scale_stride_k + m_base;
            if (m_base + 3 < scale_zero_rows && ((reinterpret_cast<uintptr_t>(sf_col) & 15) == 0)) {
                *reinterpret_cast<uint4*>(sf_col) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
            } else {
#pragma unroll
                for (int tk = 0; tk < 4; ++tk) {
                    if (m_base + tk < scale_zero_rows) {
                        sf_col[tk] = static_cast<int32_t>(packed[tk]);
                    }
                }
            }
            if (tok_tile == num_tok_tiles - 1) {
                const int64_t tma_M = num_tok_tiles * 4;
                int32_t*      sf_h  = sf_g_base + static_cast<int64_t>(h) * scale_stride_k;
                for (int64_t zm = tma_M; zm < scale_zero_rows; ++zm) {
                    sf_h[zm] = 0;
                }
            }
        }
    }
}

template<typename PosT>
void launch_inv_rope_fp8_quant_typed(const __nv_bfloat16* o_ptr,
                                     const float2*        freqs_ptr,
                                     const PosT*          pos_ptr,
                                     uint8_t*             fp8_ptr,
                                     int32_t*             scale_ptr,
                                     int64_t              M,
                                     int64_t              scale_zero_rows,
                                     int64_t              max_pos,
                                     int64_t              q_len_per_b,
                                     int                  n_groups,
                                     int                  heads_per_group,
                                     int64_t              o_stride_token,
                                     int64_t              o_stride_head,
                                     int64_t              fp8_stride_group,
                                     int64_t              fp8_stride_token,
                                     int64_t              scale_stride_group,
                                     int64_t              scale_stride_k,
                                     float                fp8_max_f,
                                     float                eps_f,
                                     cudaStream_t         stream) {
    const float         inv_fp8_max_f = 1.0f / fp8_max_f;
    const FastStrideDiv q_len_div     = make_fast_div(M, q_len_per_b);
    if (M < 128) {
        const int64_t       tma_M       = ((M + 3) >> 2) << 2;
        const int64_t       gh          = static_cast<int64_t>(n_groups) * heads_per_group;
        const int64_t       total_items = tma_M * gh;
        const FastStrideDiv gh_div      = make_fast_div(total_items, gh);
        const FastStrideDiv hpg_div     = make_fast_div(gh, heads_per_group);
        const int           block_dim   = (total_items <= 512) ? 32 : 128;
        const int64_t       warps_blk   = block_dim >> 5;
        const int           grid_dim    = static_cast<int>((total_items + warps_blk - 1) / warps_blk);
        dsv4_fused_inv_rope_fp8_quant_warp_kernel<PosT><<<grid_dim, block_dim, 0, stream>>>(
            o_ptr,
            freqs_ptr,
            pos_ptr,
            fp8_ptr,
            scale_ptr,
            M,
            tma_M,
            scale_zero_rows,
            max_pos,
            gh_div,
            hpg_div,
            q_len_div,
            heads_per_group,
            o_stride_token,
            o_stride_head,
            fp8_stride_group,
            fp8_stride_token,
            scale_stride_group,
            scale_stride_k,
            fp8_max_f,
            eps_f,
            inv_fp8_max_f,
            total_items);
    } else {
        const int64_t num_tok_tiles = (M + 3) >> 2;
        constexpr int block_dim     = 256;
        if (num_tok_tiles <= 65535 && n_groups > 0) {
            const dim3 grid(static_cast<unsigned int>(n_groups), static_cast<unsigned int>(num_tok_tiles), 1);
            dsv4_fused_inv_rope_fp8_quant_4tok_kernel<PosT, true><<<grid, block_dim, 0, stream>>>(
                o_ptr,
                freqs_ptr,
                pos_ptr,
                fp8_ptr,
                scale_ptr,
                M,
                scale_zero_rows,
                max_pos,
                q_len_div,
                heads_per_group,
                o_stride_token,
                o_stride_head,
                fp8_stride_group,
                fp8_stride_token,
                scale_stride_group,
                scale_stride_k,
                fp8_max_f,
                eps_f,
                inv_fp8_max_f);
        } else {
            const dim3 grid(static_cast<unsigned int>(num_tok_tiles), static_cast<unsigned int>(n_groups), 1);
            dsv4_fused_inv_rope_fp8_quant_4tok_kernel<PosT, false><<<grid, block_dim, 0, stream>>>(
                o_ptr,
                freqs_ptr,
                pos_ptr,
                fp8_ptr,
                scale_ptr,
                M,
                scale_zero_rows,
                max_pos,
                q_len_div,
                heads_per_group,
                o_stride_token,
                o_stride_head,
                fp8_stride_group,
                fp8_stride_token,
                scale_stride_group,
                scale_stride_k,
                fp8_max_f,
                eps_f,
                inv_fp8_max_f);
        }
    }
}

void dsv4_fused_inv_rope_fp8_quant_impl(const torch::Tensor&                o,
                                        const torch::Tensor&                freqs_ri,
                                        const std::optional<torch::Tensor>& position_ids,
                                        torch::Tensor&                      fp8_work,
                                        torch::Tensor&                      scale_work,
                                        int64_t                             q_len_per_b,
                                        int64_t                             n_groups,
                                        int64_t                             heads_per_group,
                                        int64_t                             nope_dim,
                                        int64_t                             rope_head_dim,
                                        int64_t                             quant_group_size,
                                        double                              fp8_max,
                                        double                              eps,
                                        int64_t                             scale_zero_rows) {
    TORCH_CHECK(o.is_cuda() && freqs_ri.is_cuda() && fp8_work.is_cuda() && scale_work.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(o.dtype() == torch::kBFloat16, "o must be bfloat16");
    TORCH_CHECK(freqs_ri.dtype() == torch::kFloat32 && freqs_ri.is_contiguous(),
                "freqs_ri must be contiguous float32");
    TORCH_CHECK(fp8_work.dtype() == torch::kFloat8_e4m3fn, "fp8_work must be float8_e4m3fn");
    TORCH_CHECK(scale_work.dtype() == torch::kInt32, "scale_work must be int32");
    TORCH_CHECK(o.dim() == 3 && o.stride(2) == 1, "o must be [M, H, D] with unit last stride");
    TORCH_CHECK(fp8_work.dim() == 3 && fp8_work.stride(2) == 1,
                "fp8_work must be [G, M, d] with unit last stride");
    TORCH_CHECK(scale_work.dim() == 3 && scale_work.stride(1) == 1,
                "scale_work must be [G, M, hpg] with stride(1) == 1");
    TORCH_CHECK(o.size(2) == 512 && nope_dim == 448 && rope_head_dim == 64 && quant_group_size == 128,
                "CUDA kernel currently requires D=512, nope_dim=448, rope_head_dim=64, quant_group_size=128");

    const int64_t M = o.size(0);
    if (M == 0) {
        return;
    }
    TORCH_CHECK((reinterpret_cast<uintptr_t>(o.data_ptr()) & 15) == 0, "o ptr must be 16-byte aligned");
    TORCH_CHECK((o.stride(0) & 7) == 0 && (o.stride(1) & 7) == 0, "o strides must be multiple of 8");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(fp8_work.data_ptr()) & 7) == 0, "fp8_work ptr must be 8-byte aligned");
    TORCH_CHECK((fp8_work.stride(0) & 7) == 0 && (fp8_work.stride(1) & 7) == 0,
                "fp8_work strides must be multiple of 8");

    const __nv_bfloat16* o_ptr     = reinterpret_cast<const __nv_bfloat16*>(o.data_ptr<at::BFloat16>());
    const float2*        freqs_ptr = reinterpret_cast<const float2*>(freqs_ri.data_ptr<float>());
    TORCH_CHECK((reinterpret_cast<uintptr_t>(freqs_ptr) & 15) == 0,
                "freqs_ri pointer must be 16-byte aligned");
    uint8_t*      fp8_ptr   = reinterpret_cast<uint8_t*>(fp8_work.data_ptr());
    int32_t*      scale_ptr = scale_work.data_ptr<int32_t>();
    const int64_t max_pos   = freqs_ri.size(0);
    cudaStream_t  stream    = at::cuda::getCurrentCUDAStream();

    if (!position_ids.has_value()) {
        launch_inv_rope_fp8_quant_typed<void>(o_ptr,
                                              freqs_ptr,
                                              static_cast<const void*>(nullptr),
                                              fp8_ptr,
                                              scale_ptr,
                                              M,
                                              scale_zero_rows,
                                              max_pos,
                                              q_len_per_b,
                                              static_cast<int>(n_groups),
                                              static_cast<int>(heads_per_group),
                                              o.stride(0),
                                              o.stride(1),
                                              fp8_work.stride(0),
                                              fp8_work.stride(1),
                                              scale_work.stride(0),
                                              scale_work.stride(2),
                                              static_cast<float>(fp8_max),
                                              static_cast<float>(eps),
                                              stream);
    } else if (position_ids->dtype() == torch::kInt32) {
        launch_inv_rope_fp8_quant_typed<int32_t>(o_ptr,
                                                 freqs_ptr,
                                                 position_ids->data_ptr<int32_t>(),
                                                 fp8_ptr,
                                                 scale_ptr,
                                                 M,
                                                 scale_zero_rows,
                                                 max_pos,
                                                 q_len_per_b,
                                                 static_cast<int>(n_groups),
                                                 static_cast<int>(heads_per_group),
                                                 o.stride(0),
                                                 o.stride(1),
                                                 fp8_work.stride(0),
                                                 fp8_work.stride(1),
                                                 scale_work.stride(0),
                                                 scale_work.stride(2),
                                                 static_cast<float>(fp8_max),
                                                 static_cast<float>(eps),
                                                 stream);
    } else if (position_ids->dtype() == torch::kInt64) {
        launch_inv_rope_fp8_quant_typed<int64_t>(o_ptr,
                                                 freqs_ptr,
                                                 position_ids->data_ptr<int64_t>(),
                                                 fp8_ptr,
                                                 scale_ptr,
                                                 M,
                                                 scale_zero_rows,
                                                 max_pos,
                                                 q_len_per_b,
                                                 static_cast<int>(n_groups),
                                                 static_cast<int>(heads_per_group),
                                                 o.stride(0),
                                                 o.stride(1),
                                                 fp8_work.stride(0),
                                                 fp8_work.stride(1),
                                                 scale_work.stride(0),
                                                 scale_work.stride(2),
                                                 static_cast<float>(fp8_max),
                                                 static_cast<float>(eps),
                                                 stream);
    } else {
        TORCH_CHECK(false, "position_ids must be int32 or int64");
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dsv4_fused_inv_rope_fp8_quant launch failed: ", cudaGetErrorString(err));
}

// ---------------------------------------------------------------------------
// 3. dsv4_indexer_q_rope_fp8_fold (D == 128, split NoPE/RoPE, low-reg full grid)
// ---------------------------------------------------------------------------
template<typename WeightT>
__device__ __forceinline__ float load_scalar_weight(const WeightT* __restrict__ w_ptr, int64_t row);

template<>
__device__ __forceinline__ float load_scalar_weight<__nv_bfloat16>(const __nv_bfloat16* __restrict__ w_ptr,
                                                                   int64_t row) {
    return __bfloat162float(w_ptr[row]);
}

template<>
__device__ __forceinline__ float load_scalar_weight<float>(const float* __restrict__ w_ptr, int64_t row) {
    return w_ptr[row];
}

// Pipelined 2D-grid kernel for freq_stride_n % 32 == 0 (D=128, RD=64):
// - 32 heads per CTA (blockDim = 128, 4 iters of 8 heads), zero div.s64
// - Preloads iteration-0 q_ptr, s_freqs[32], and s_w[32] before single __syncthreads(),
//   and prefetches u_next across all 4 iterations with full 128-bit/64-bit coalescing.
template<typename WeightT, bool HEAD_ON_X>
__global__ void dsv4_indexer_q_rope_fp8_fold_grouped32_d64_kernel(const __nv_bfloat16* __restrict__ q_ptr,
                                                                  const WeightT* __restrict__ w_ptr,
                                                                  const float2* __restrict__ freqs_ptr,
                                                                  uint8_t* __restrict__ q_fp8_ptr,
                                                                  float* __restrict__ w_fold_ptr,
                                                                  int64_t freq_stride_n,
                                                                  float   fp8_max_f) {
    __shared__ alignas(16) float2 s_freqs[32];
    __shared__ float              s_w[32];

    const int64_t freq_idx    = HEAD_ON_X ? static_cast<int64_t>(blockIdx.y) : static_cast<int64_t>(blockIdx.x);
    const int64_t head_tile   = HEAD_ON_X ? static_cast<int64_t>(blockIdx.x) : static_cast<int64_t>(blockIdx.y);
    const int64_t tile_base   = freq_idx * freq_stride_n + head_tile * 32;
    const int     sub_lane    = threadIdx.x & 15;
    const int     half_in_blk = threadIdx.x >> 4;
    const int     col         = sub_lane * 8;

    uint4 u = __ldg(reinterpret_cast<const uint4*>(q_ptr + (tile_base + half_in_blk) * 128 + col));
    if (threadIdx.x < 32) {
        s_freqs[threadIdx.x] = __ldg(freqs_ptr + freq_idx * 32 + threadIdx.x);
        s_w[threadIdx.x]     = load_scalar_weight(w_ptr, tile_base + threadIdx.x);
    }
    __syncthreads();

#pragma unroll
    for (int it = 0; it < 4; ++it) {
        const int     local_row = it * 8 + half_in_blk;
        const int64_t row       = tile_base + local_row;
        uint4         u_next    = make_uint4(0u, 0u, 0u, 0u);
        if (it < 3) {
            u_next = __ldg(reinterpret_cast<const uint4*>(q_ptr + (row + 8) * 128 + col));
        }

        float f[8];
        Vec8Traits<__nv_bfloat16>::unpack_u4(u, f);
        u = u_next;

        if (sub_lane >= 8) {
            const int pair_base = (sub_lane - 8) * 4;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float2 cs = s_freqs[pair_base + p];
                const float  e  = f[2 * p];
                const float  o  = f[2 * p + 1];
                const float  re = fmaf(e, cs.x, -(o * cs.y));
                const float  ro = fmaf(o, cs.x, e * cs.y);
                const float2 rb = bf16x2_raw_to_float2(float2_to_bf16x2_raw(re, ro));
                f[2 * p]        = rb.x;
                f[2 * p + 1]    = rb.y;
            }
        }

        float m = fabsf(f[0]);
#pragma unroll
        for (int i = 1; i < 8; ++i) {
            m = fmaxf(m, fabsf(f[i]));
        }
        m = halfwarp_absmax_reduce(m);

        const float scale = fmaxf(__fdividef(m, fp8_max_f), 1e-12f);

        __nv_fp8x2_storage_t q[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            q[i] = __nv_cvt_float2_to_fp8x2(
                make_float2(__fdividef(f[2 * i], scale), __fdividef(f[2 * i + 1], scale)),
                __NV_SATFINITE,
                __NV_E4M3);
        }
        uint2 out_u2;
        out_u2.x = static_cast<uint32_t>(q[0]) | (static_cast<uint32_t>(q[1]) << 16);
        out_u2.y = static_cast<uint32_t>(q[2]) | (static_cast<uint32_t>(q[3]) << 16);
        *reinterpret_cast<uint2*>(q_fp8_ptr + row * 128 + col) = out_u2;

        if (sub_lane == 0) {
            w_fold_ptr[row] = s_w[local_row] * scale;
        }
    }
}

template<typename WeightT, bool RD_IS_64>
__global__ void dsv4_indexer_q_rope_fp8_fold_kernel(const __nv_bfloat16* __restrict__ q_ptr,
                                                    const WeightT* __restrict__ w_ptr,
                                                    const float2* __restrict__ freqs_ptr,
                                                    uint8_t* __restrict__ q_fp8_ptr,
                                                    float* __restrict__ w_fold_ptr,
                                                    int64_t       BSH,
                                                    FastStrideDiv fdiv,
                                                    int           rd,
                                                    float         fp8_max_f) {
    const int     sub_lane    = threadIdx.x & 15;
    const int     half_in_blk = threadIdx.x >> 4;
    const int64_t row         = static_cast<int64_t>(blockIdx.x) * (blockDim.x >> 4) + half_in_blk;
    if (row >= BSH) {
        return;
    }

    const int col         = sub_lane * 8;
    const int nope_offset = RD_IS_64 ? 64 : (128 - rd);
    const int rd_half     = RD_IS_64 ? 32 : (rd >> 1);

    const uint4 u = *reinterpret_cast<const uint4*>(q_ptr + row * 128 + col);
    float       f[8];
    Vec8Traits<__nv_bfloat16>::unpack_u4(u, f);

    if constexpr (RD_IS_64) {
        (void)rd;
        (void)nope_offset;
        (void)rd_half;
        if (sub_lane >= 8) {
            const int64_t freq_idx  = fdiv.div(row);
            const float2* f_row     = freqs_ptr + freq_idx * 32;
            const int     pair_base = (sub_lane - 8) * 4;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float2 cs = __ldg(f_row + pair_base + p);
                const float  e  = f[2 * p];
                const float  o  = f[2 * p + 1];
                const float  re = e * cs.x - o * cs.y;
                const float  ro = o * cs.x + e * cs.y;
                const float2 rb = bf16x2_raw_to_float2(float2_to_bf16x2_raw(re, ro));
                f[2 * p]        = rb.x;
                f[2 * p + 1]    = rb.y;
            }
        }
    } else {
        if (col + 7 >= nope_offset) {
            const int64_t freq_idx = fdiv.div(row);
            const float2* f_row    = freqs_ptr + freq_idx * rd_half;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int c_idx = col + 2 * p;
                if (c_idx >= nope_offset) {
                    const float2 cs = __ldg(f_row + ((c_idx - nope_offset) >> 1));
                    const float  e  = f[2 * p];
                    const float  o  = f[2 * p + 1];
                    const float  re = fmaf(e, cs.x, -(o * cs.y));
                    const float  ro = fmaf(o, cs.x, e * cs.y);
                    const float2 rb = bf16x2_raw_to_float2(float2_to_bf16x2_raw(re, ro));
                    f[2 * p]        = rb.x;
                    f[2 * p + 1]    = rb.y;
                }
            }
        }
    }

    float m = fabsf(f[0]);
#pragma unroll
    for (int i = 1; i < 8; ++i) {
        m = fmaxf(m, fabsf(f[i]));
    }
    m = halfwarp_absmax_reduce(m);

    const float scale = fmaxf(__fdividef(m, fp8_max_f), 1e-12f);

    __nv_fp8x2_storage_t q[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        q[i] = __nv_cvt_float2_to_fp8x2(
            make_float2(__fdividef(f[2 * i], scale), __fdividef(f[2 * i + 1], scale)),
            __NV_SATFINITE,
            __NV_E4M3);
    }
    uint2 out_u2;
    out_u2.x = static_cast<uint32_t>(q[0]) | (static_cast<uint32_t>(q[1]) << 16);
    out_u2.y = static_cast<uint32_t>(q[2]) | (static_cast<uint32_t>(q[3]) << 16);
    *reinterpret_cast<uint2*>(q_fp8_ptr + row * 128 + col) = out_u2;

    if (sub_lane == 0) {
        w_fold_ptr[row] = load_scalar_weight(w_ptr, row) * scale;
    }
}

template<typename WeightT>
void launch_indexer_q_rope_typed(const __nv_bfloat16* q_ptr,
                                 const WeightT*       w_ptr,
                                 const float2*        freqs_ptr,
                                 uint8_t*             q_fp8_ptr,
                                 float*               w_fold_ptr,
                                 int64_t              BSH,
                                 int64_t              freq_stride_n,
                                 int                  rd,
                                 float                fp8_max_f,
                                 cudaStream_t         stream) {
    const int64_t n_freq = BSH / freq_stride_n;
    if (rd == 64 && freq_stride_n >= 32 && (freq_stride_n % 32 == 0) && BSH >= 4096) {
        const int64_t head_tiles = freq_stride_n / 32;
        if (head_tiles <= 65535 && n_freq <= 65535) {
            const dim3 grid(static_cast<unsigned int>(head_tiles), static_cast<unsigned int>(n_freq), 1);
            dsv4_indexer_q_rope_fp8_fold_grouped32_d64_kernel<WeightT, true><<<grid, 128, 0, stream>>>(
                q_ptr, w_ptr, freqs_ptr, q_fp8_ptr, w_fold_ptr, freq_stride_n, fp8_max_f);
            return;
        }
        if (head_tiles <= 65535) {
            const dim3 grid(static_cast<unsigned int>(n_freq), static_cast<unsigned int>(head_tiles), 1);
            dsv4_indexer_q_rope_fp8_fold_grouped32_d64_kernel<WeightT, false><<<grid, 128, 0, stream>>>(
                q_ptr, w_ptr, freqs_ptr, q_fp8_ptr, w_fold_ptr, freq_stride_n, fp8_max_f);
            return;
        }
    }

    const FastStrideDiv fdiv = make_fast_div(BSH, freq_stride_n);
    int block_dim = 128;
    int grid_dim  = 0;
    if (BSH <= 256) {
        block_dim = 32;
        grid_dim  = static_cast<int>((BSH + 1) >> 1);
    } else if (BSH <= 4096) {
        block_dim = 64;
        grid_dim  = static_cast<int>((BSH + 3) >> 2);
    } else {
        block_dim = 128;
        grid_dim  = static_cast<int>((BSH + 7) >> 3);
    }

    if (rd == 64) {
        dsv4_indexer_q_rope_fp8_fold_kernel<WeightT, true><<<grid_dim, block_dim, 0, stream>>>(
            q_ptr, w_ptr, freqs_ptr, q_fp8_ptr, w_fold_ptr, BSH, fdiv, rd, fp8_max_f);
    } else {
        dsv4_indexer_q_rope_fp8_fold_kernel<WeightT, false><<<grid_dim, block_dim, 0, stream>>>(
            q_ptr, w_ptr, freqs_ptr, q_fp8_ptr, w_fold_ptr, BSH, fdiv, rd, fp8_max_f);
    }
}

void dsv4_indexer_q_rope_fp8_fold_impl(const torch::Tensor& q_bf16,
                                       const torch::Tensor& weights,
                                       const torch::Tensor& freqs_ri,
                                       torch::Tensor&       q_fp8,
                                       torch::Tensor&       w_fold,
                                       int64_t              freq_stride_n,
                                       int64_t              rope_head_dim,
                                       double               fp8_max) {
    TORCH_CHECK(q_bf16.is_cuda() && weights.is_cuda() && freqs_ri.is_cuda() && q_fp8.is_cuda() && w_fold.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(q_bf16.is_contiguous() && weights.is_contiguous() && freqs_ri.is_contiguous() &&
                    q_fp8.is_contiguous() && w_fold.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(q_bf16.dim() == 2 && q_bf16.size(1) == 128 && q_bf16.dtype() == torch::kBFloat16,
                "q_bf16 must be [BSH, 128] bfloat16");
    TORCH_CHECK(q_fp8.sizes() == q_bf16.sizes() && q_fp8.dtype() == torch::kFloat8_e4m3fn,
                "q_fp8 must match q_bf16 shape with float8_e4m3fn");
    TORCH_CHECK(weights.dim() == 1 && weights.size(0) == q_bf16.size(0), "weights must be [BSH]");
    TORCH_CHECK(w_fold.dim() == 1 && w_fold.size(0) == q_bf16.size(0) && w_fold.dtype() == torch::kFloat32,
                "w_fold must be [BSH] float32");
    TORCH_CHECK(freqs_ri.dtype() == torch::kFloat32, "freqs_ri must be float32");
    TORCH_CHECK(rope_head_dim > 0 && rope_head_dim <= 128 && (rope_head_dim % 2 == 0),
                "invalid rope_head_dim");

    const int64_t BSH = q_bf16.size(0);
    if (BSH == 0) {
        return;
    }
    TORCH_CHECK((reinterpret_cast<uintptr_t>(q_bf16.data_ptr()) & 15) == 0 &&
                    (reinterpret_cast<uintptr_t>(q_fp8.data_ptr()) & 7) == 0,
                "q pointers must be vector aligned");

    const __nv_bfloat16* q_ptr     = reinterpret_cast<const __nv_bfloat16*>(q_bf16.data_ptr<at::BFloat16>());
    const float2*        freqs_ptr = reinterpret_cast<const float2*>(freqs_ri.data_ptr<float>());
    uint8_t*             q_fp8_ptr = reinterpret_cast<uint8_t*>(q_fp8.data_ptr());
    float*               w_fold_p  = w_fold.data_ptr<float>();
    cudaStream_t         stream    = at::cuda::getCurrentCUDAStream();

    if (weights.dtype() == torch::kBFloat16) {
        launch_indexer_q_rope_typed<__nv_bfloat16>(
            q_ptr,
            reinterpret_cast<const __nv_bfloat16*>(weights.data_ptr<at::BFloat16>()),
            freqs_ptr,
            q_fp8_ptr,
            w_fold_p,
            BSH,
            freq_stride_n,
            static_cast<int>(rope_head_dim),
            static_cast<float>(fp8_max),
            stream);
    } else if (weights.dtype() == torch::kFloat32) {
        launch_indexer_q_rope_typed<float>(
            q_ptr,
            weights.data_ptr<float>(),
            freqs_ptr,
            q_fp8_ptr,
            w_fold_p,
            BSH,
            freq_stride_n,
            static_cast<int>(rope_head_dim),
            static_cast<float>(fp8_max),
            stream);
    } else {
        TORCH_CHECK(false, "weights dtype must be bfloat16 or float32");
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dsv4_indexer_q_rope_fp8_fold launch failed: ", cudaGetErrorString(err));
}

}  // namespace
#endif  // USE_ROCM

void dsv4_fused_rmsnorm_rope(const torch::Tensor&                x,
                             const std::optional<torch::Tensor>& weight,
                             const torch::Tensor&                freqs_ri,
                             const std::optional<torch::Tensor>& position_ids,
                             torch::Tensor&                      out,
                             int64_t                             freq_stride_n,
                             int64_t                             rope_head_dim,
                             double                              eps,
                             bool                                inverse) {
#ifndef USE_ROCM
    dsv4_fused_rmsnorm_rope_impl(x, weight, freqs_ri, position_ids, out, freq_stride_n, rope_head_dim, eps, inverse);
#else
    (void)x;
    (void)weight;
    (void)freqs_ri;
    (void)position_ids;
    (void)out;
    (void)freq_stride_n;
    (void)rope_head_dim;
    (void)eps;
    (void)inverse;
    TORCH_CHECK(false, "dsv4_fused_rmsnorm_rope CUDA op is not supported on ROCm");
#endif
}

void dsv4_fused_inv_rope_fp8_quant(const torch::Tensor&                o,
                                   const torch::Tensor&                freqs_ri,
                                   const std::optional<torch::Tensor>& position_ids,
                                   torch::Tensor&                      fp8_work,
                                   torch::Tensor&                      scale_work,
                                   int64_t                             q_len_per_b,
                                   int64_t                             n_groups,
                                   int64_t                             heads_per_group,
                                   int64_t                             nope_dim,
                                   int64_t                             rope_head_dim,
                                   int64_t                             quant_group_size,
                                   double                              fp8_max,
                                   double                              eps,
                                   int64_t                             scale_zero_rows) {
#ifndef USE_ROCM
    dsv4_fused_inv_rope_fp8_quant_impl(o,
                                       freqs_ri,
                                       position_ids,
                                       fp8_work,
                                       scale_work,
                                       q_len_per_b,
                                       n_groups,
                                       heads_per_group,
                                       nope_dim,
                                       rope_head_dim,
                                       quant_group_size,
                                       fp8_max,
                                       eps,
                                       scale_zero_rows);
#else
    (void)o;
    (void)freqs_ri;
    (void)position_ids;
    (void)fp8_work;
    (void)scale_work;
    (void)q_len_per_b;
    (void)n_groups;
    (void)heads_per_group;
    (void)nope_dim;
    (void)rope_head_dim;
    (void)quant_group_size;
    (void)fp8_max;
    (void)eps;
    (void)scale_zero_rows;
    TORCH_CHECK(false, "dsv4_fused_inv_rope_fp8_quant CUDA op is not supported on ROCm");
#endif
}

void dsv4_indexer_q_rope_fp8_fold(const torch::Tensor& q_bf16,
                                  const torch::Tensor& weights,
                                  const torch::Tensor& freqs_ri,
                                  torch::Tensor&       q_fp8,
                                  torch::Tensor&       w_fold,
                                  int64_t              freq_stride_n,
                                  int64_t              rope_head_dim,
                                  double               fp8_max) {
#ifndef USE_ROCM
    dsv4_indexer_q_rope_fp8_fold_impl(
        q_bf16, weights, freqs_ri, q_fp8, w_fold, freq_stride_n, rope_head_dim, fp8_max);
#else
    (void)q_bf16;
    (void)weights;
    (void)freqs_ri;
    (void)q_fp8;
    (void)w_fold;
    (void)freq_stride_n;
    (void)rope_head_dim;
    (void)fp8_max;
    TORCH_CHECK(false, "dsv4_indexer_q_rope_fp8_fold CUDA op is not supported on ROCm");
#endif
}

}  // namespace torch_ext
