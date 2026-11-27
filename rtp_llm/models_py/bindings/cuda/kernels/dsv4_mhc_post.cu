// Vectorized CUDA kernel for DeepSeek-V4 mHC post-residual merge.
//
// Computes, for token t in [0, M), output lane o in [0, H), column d in [0, D):
//   out[t, o, d] = bf16(post[t, o] * x[t, d] + sum_{i=0}^{H-1} comb[t, i, o] * residual[t, i, d])
//
// Each thread reads x[t, d..d+7] and all H lanes of residual[t, :, d..d+7] before
// storing any out[t, :, d..d+7], so in-place aliasing (out.data_ptr() == residual.data_ptr())
// is exact and barrier-free. For large M with H == 4 and D % 1024 == 0, uses a 2-stage
// cp.async.cg double-buffered pipeline (1 CTA per token, 1024 cols/stage) with direct
// global uint4 stores (no output SMEM bounce).

#include "rtp_llm/models_py/bindings/cuda/kernels/dsv4_mhc_post.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include <cstdint>
#include <cstring>

#ifndef USE_ROCM
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#endif

namespace torch_ext {

#ifndef USE_ROCM
namespace {

constexpr int kMaxVecH    = 8;
constexpr int kMaxScalarH = 32;

struct FastDiv32 {
    int32_t  shift;
    uint32_t divisor;

    __device__ __forceinline__ void divmod(int64_t val, int64_t& q, int64_t& r) const {
        if (shift >= 0) {
            q = val >> shift;
            r = val & static_cast<int64_t>(divisor - 1u);
        } else {
            q = val / static_cast<int64_t>(divisor);
            r = val - q * static_cast<int64_t>(divisor);
        }
    }
};

inline FastDiv32 make_fast_div32(int64_t divisor) {
    FastDiv32 fd{};
    fd.divisor = static_cast<uint32_t>(divisor);
    if (divisor > 0 && (divisor & (divisor - 1)) == 0) {
        int     s = 0;
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

__device__ __forceinline__ void unpack_u4_bf16(uint4 v, float out[8]) {
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

__device__ __forceinline__ uint4 pack_u4_bf16(const float in[8]) {
    return make_uint4(float2_to_bf16x2_raw(in[0], in[1]),
                      float2_to_bf16x2_raw(in[2], in[3]),
                      float2_to_bf16x2_raw(in[4], in[5]),
                      float2_to_bf16x2_raw(in[6], in[7]));
}

__device__ __forceinline__ float load_coeff_f32(const void* __restrict__ ptr, int64_t idx, bool is_bf16) {
    if (is_bf16) {
        const uint16_t raw = __ldg(reinterpret_cast<const uint16_t*>(ptr) + idx);
        __nv_bfloat16_raw b_raw;
        b_raw.x = raw;
        return __bfloat162float(__nv_bfloat16(b_raw));
    }
    const float* p = reinterpret_cast<const float*>(ptr);
    return __ldg(p + idx);
}

__device__ __forceinline__ void cp_async_ca_16(void* smem_dst, const void* gmem_src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" : : "r"(dst), "l"(gmem_src) : "memory");
#else
    (void)smem_dst;
    (void)gmem_src;
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N) : "memory");
#endif
}

// 2-stage cp.async.ca 1-CTA-per-token kernel for compile-time D (4096 / 7168),
// H == 4, FP32 16B-aligned post/comb, M >= 1024.
// Only thread `tid` accesses smem[stage][...][tid], so ZERO __syncthreads() barriers exist.
template<int64_t D_CONST>
__global__ void __launch_bounds__(128, 1) dsv4_mhc_post_h4_cpasync_kernel(const __nv_bfloat16* __restrict__ x_ptr,
                                                                           const __nv_bfloat16* __restrict__ res_ptr,
                                                                           const float* __restrict__ post_ptr,
                                                                           const float* __restrict__ comb_ptr,
                                                                           __nv_bfloat16* __restrict__ out_ptr,
                                                                           int64_t M) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int N_TILES = static_cast<int>(D_CONST >> 10);
    const int64_t t       = static_cast<int64_t>(blockIdx.x);
    if (t >= M) {
        return;
    }

    __shared__ __align__(16) uint4 sm_x[2][128];
    __shared__ __align__(16) uint4 sm_r[2][4][128];

    const int            tid   = static_cast<int>(threadIdx.x);
    const int64_t        col0  = static_cast<int64_t>(tid) * 8;
    const __nv_bfloat16* x_row = x_ptr + t * D_CONST + col0;
    const __nv_bfloat16* r_row = res_ptr + t * (4 * D_CONST) + col0;
    __nv_bfloat16*       o_row = out_ptr + t * (4 * D_CONST) + col0;

    // Issue stage 0 & stage 1 prefetches before loading post/comb coefficients
    cp_async_ca_16(&sm_x[0][tid], x_row);
    cp_async_ca_16(&sm_r[0][0][tid], r_row);
    cp_async_ca_16(&sm_r[0][1][tid], r_row + D_CONST);
    cp_async_ca_16(&sm_r[0][2][tid], r_row + 2 * D_CONST);
    cp_async_ca_16(&sm_r[0][3][tid], r_row + 3 * D_CONST);
    cp_async_commit();

    if constexpr (N_TILES >= 2) {
        cp_async_ca_16(&sm_x[1][tid], x_row + 1024);
        cp_async_ca_16(&sm_r[1][0][tid], r_row + 1024);
        cp_async_ca_16(&sm_r[1][1][tid], r_row + D_CONST + 1024);
        cp_async_ca_16(&sm_r[1][2][tid], r_row + 2 * D_CONST + 1024);
        cp_async_ca_16(&sm_r[1][3][tid], r_row + 3 * D_CONST + 1024);
        cp_async_commit();
    }

    const float4 p4   = __ldg(reinterpret_cast<const float4*>(post_ptr + t * 4));
    const float4 c4_0 = __ldg(reinterpret_cast<const float4*>(comb_ptr + t * 16));
    const float4 c4_1 = __ldg(reinterpret_cast<const float4*>(comb_ptr + t * 16 + 4));
    const float4 c4_2 = __ldg(reinterpret_cast<const float4*>(comb_ptr + t * 16 + 8));
    const float4 c4_3 = __ldg(reinterpret_cast<const float4*>(comb_ptr + t * 16 + 12));

    const float p[4]  = {p4.x, p4.y, p4.z, p4.w};
    const float c[16] = {
        c4_0.x, c4_0.y, c4_0.z, c4_0.w,
        c4_1.x, c4_1.y, c4_1.z, c4_1.w,
        c4_2.x, c4_2.y, c4_2.z, c4_2.w,
        c4_3.x, c4_3.y, c4_3.z, c4_3.w,
    };

#pragma unroll 1
    for (int tile = 0; tile < N_TILES; ++tile) {
        const int cur = tile & 1;
        if (tile + 1 < N_TILES) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }

        const uint4 xu  = sm_x[cur][tid];
        const uint4 ru0 = sm_r[cur][0][tid];
        const uint4 ru1 = sm_r[cur][1][tid];
        const uint4 ru2 = sm_r[cur][2][tid];
        const uint4 ru3 = sm_r[cur][3][tid];

        // Prefetch tile + 2 immediately into the same cur SMEM slot once read into regs!
        if (tile + 2 < N_TILES) {
            const int64_t off_n = static_cast<int64_t>(tile + 2) * 1024;
            cp_async_ca_16(&sm_x[cur][tid], x_row + off_n);
            cp_async_ca_16(&sm_r[cur][0][tid], r_row + off_n);
            cp_async_ca_16(&sm_r[cur][1][tid], r_row + D_CONST + off_n);
            cp_async_ca_16(&sm_r[cur][2][tid], r_row + 2 * D_CONST + off_n);
            cp_async_ca_16(&sm_r[cur][3][tid], r_row + 3 * D_CONST + off_n);
            cp_async_commit();
        }

        float xf[8];
        unpack_u4_bf16(xu, xf);
        uint4 out_u[4];
#pragma unroll
        for (int o = 0; o < 4; ++o) {
            const float po = p[o];
            const float c0 = c[o];
            const float c1 = c[4 + o];
            const float c2 = c[8 + o];
            const float c3 = c[12 + o];

            const float2 fr0_0 = bf16x2_raw_to_float2(ru0.x);
            const float2 fr1_0 = bf16x2_raw_to_float2(ru1.x);
            const float2 fr2_0 = bf16x2_raw_to_float2(ru2.x);
            const float2 fr3_0 = bf16x2_raw_to_float2(ru3.x);
            const float  a0    = po * xf[0] + c0 * fr0_0.x + c1 * fr1_0.x + c2 * fr2_0.x + c3 * fr3_0.x;
            const float  a1    = po * xf[1] + c0 * fr0_0.y + c1 * fr1_0.y + c2 * fr2_0.y + c3 * fr3_0.y;

            const float2 fr0_1 = bf16x2_raw_to_float2(ru0.y);
            const float2 fr1_1 = bf16x2_raw_to_float2(ru1.y);
            const float2 fr2_1 = bf16x2_raw_to_float2(ru2.y);
            const float2 fr3_1 = bf16x2_raw_to_float2(ru3.y);
            const float  a2    = po * xf[2] + c0 * fr0_1.x + c1 * fr1_1.x + c2 * fr2_1.x + c3 * fr3_1.x;
            const float  a3    = po * xf[3] + c0 * fr0_1.y + c1 * fr1_1.y + c2 * fr2_1.y + c3 * fr3_1.y;

            const float2 fr0_2 = bf16x2_raw_to_float2(ru0.z);
            const float2 fr1_2 = bf16x2_raw_to_float2(ru1.z);
            const float2 fr2_2 = bf16x2_raw_to_float2(ru2.z);
            const float2 fr3_2 = bf16x2_raw_to_float2(ru3.z);
            const float  a4    = po * xf[4] + c0 * fr0_2.x + c1 * fr1_2.x + c2 * fr2_2.x + c3 * fr3_2.x;
            const float  a5    = po * xf[5] + c0 * fr0_2.y + c1 * fr1_2.y + c2 * fr2_2.y + c3 * fr3_2.y;

            const float2 fr0_3 = bf16x2_raw_to_float2(ru0.w);
            const float2 fr1_3 = bf16x2_raw_to_float2(ru1.w);
            const float2 fr2_3 = bf16x2_raw_to_float2(ru2.w);
            const float2 fr3_3 = bf16x2_raw_to_float2(ru3.w);
            const float  a6    = po * xf[6] + c0 * fr0_3.x + c1 * fr1_3.x + c2 * fr2_3.x + c3 * fr3_3.x;
            const float  a7    = po * xf[7] + c0 * fr0_3.y + c1 * fr1_3.y + c2 * fr2_3.y + c3 * fr3_3.y;

            out_u[o] = make_uint4(float2_to_bf16x2_raw(a0, a1),
                                  float2_to_bf16x2_raw(a2, a3),
                                  float2_to_bf16x2_raw(a4, a5),
                                  float2_to_bf16x2_raw(a6, a7));
        }

        const int64_t off_c                                    = static_cast<int64_t>(tile) * 1024;
        *reinterpret_cast<uint4*>(o_row + off_c)               = out_u[0];
        *reinterpret_cast<uint4*>(o_row + D_CONST + off_c)     = out_u[1];
        *reinterpret_cast<uint4*>(o_row + 2 * D_CONST + off_c) = out_u[2];
        *reinterpret_cast<uint4*>(o_row + 3 * D_CONST + off_c) = out_u[3];
    }
#else
    (void)x_ptr;
    (void)res_ptr;
    (void)post_ptr;
    (void)comb_ptr;
    (void)out_ptr;
    (void)M;
#endif
}

// Fast 128-bit direct kernel for H == 4, 16-byte aligned pointers, D % 8 == 0.
__global__ void dsv4_mhc_post_h4_vec8_kernel(const __nv_bfloat16* __restrict__ x_ptr,
                                             const __nv_bfloat16* __restrict__ res_ptr,
                                             const void* __restrict__ post_ptr,
                                             const void* __restrict__ comb_ptr,
                                             __nv_bfloat16* __restrict__ out_ptr,
                                             int64_t   M,
                                             int64_t   D,
                                             int64_t   D_vec,
                                             FastDiv32 fdiv_tiles,
                                             bool      post_is_bf16,
                                             bool      comb_is_bf16) {
    __shared__ float sm_post[4];
    __shared__ float sm_comb[16];

    const int64_t total_work = M * static_cast<int64_t>(fdiv_tiles.divisor);
    for (int64_t work = static_cast<int64_t>(blockIdx.x); work < total_work; work += static_cast<int64_t>(gridDim.x)) {
        if (work != static_cast<int64_t>(blockIdx.x)) {
            __syncthreads();
        }
        int64_t t        = 0;
        int64_t tile_idx = 0;
        fdiv_tiles.divmod(work, t, tile_idx);

        if (threadIdx.x < 4) {
            sm_post[threadIdx.x] = load_coeff_f32(post_ptr, t * 4 + threadIdx.x, post_is_bf16);
        }
        if (threadIdx.x < 16) {
            sm_comb[threadIdx.x] = load_coeff_f32(comb_ptr, t * 16 + threadIdx.x, comb_is_bf16);
        }
        __syncthreads();

        const int64_t v = tile_idx * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
        if (v >= D_vec) {
            continue;
        }
        const int64_t col = v * 8;

        const __nv_bfloat16* x_row = x_ptr + t * D;
        const __nv_bfloat16* r_row = res_ptr + t * 4 * D;
        __nv_bfloat16*       o_row = out_ptr + t * 4 * D;

        const uint4 xu  = __ldg(reinterpret_cast<const uint4*>(x_row + col));
        const uint4 ru0 = *reinterpret_cast<const uint4*>(r_row + col);
        const uint4 ru1 = *reinterpret_cast<const uint4*>(r_row + D + col);
        const uint4 ru2 = *reinterpret_cast<const uint4*>(r_row + 2 * D + col);
        const uint4 ru3 = *reinterpret_cast<const uint4*>(r_row + 3 * D + col);

        float xf[8];
        unpack_u4_bf16(xu, xf);

#pragma unroll
        for (int o = 0; o < 4; ++o) {
            const float po = sm_post[o];
            const float c0 = sm_comb[o];
            const float c1 = sm_comb[4 + o];
            const float c2 = sm_comb[8 + o];
            const float c3 = sm_comb[12 + o];

            const float2 fr0_0 = bf16x2_raw_to_float2(ru0.x);
            const float2 fr1_0 = bf16x2_raw_to_float2(ru1.x);
            const float2 fr2_0 = bf16x2_raw_to_float2(ru2.x);
            const float2 fr3_0 = bf16x2_raw_to_float2(ru3.x);
            const float  a0    = po * xf[0] + c0 * fr0_0.x + c1 * fr1_0.x + c2 * fr2_0.x + c3 * fr3_0.x;
            const float  a1    = po * xf[1] + c0 * fr0_0.y + c1 * fr1_0.y + c2 * fr2_0.y + c3 * fr3_0.y;

            const float2 fr0_1 = bf16x2_raw_to_float2(ru0.y);
            const float2 fr1_1 = bf16x2_raw_to_float2(ru1.y);
            const float2 fr2_1 = bf16x2_raw_to_float2(ru2.y);
            const float2 fr3_1 = bf16x2_raw_to_float2(ru3.y);
            const float  a2    = po * xf[2] + c0 * fr0_1.x + c1 * fr1_1.x + c2 * fr2_1.x + c3 * fr3_1.x;
            const float  a3    = po * xf[3] + c0 * fr0_1.y + c1 * fr1_1.y + c2 * fr2_1.y + c3 * fr3_1.y;

            const float2 fr0_2 = bf16x2_raw_to_float2(ru0.z);
            const float2 fr1_2 = bf16x2_raw_to_float2(ru1.z);
            const float2 fr2_2 = bf16x2_raw_to_float2(ru2.z);
            const float2 fr3_2 = bf16x2_raw_to_float2(ru3.z);
            const float  a4    = po * xf[4] + c0 * fr0_2.x + c1 * fr1_2.x + c2 * fr2_2.x + c3 * fr3_2.x;
            const float  a5    = po * xf[5] + c0 * fr0_2.y + c1 * fr1_2.y + c2 * fr2_2.y + c3 * fr3_2.y;

            const float2 fr0_3 = bf16x2_raw_to_float2(ru0.w);
            const float2 fr1_3 = bf16x2_raw_to_float2(ru1.w);
            const float2 fr2_3 = bf16x2_raw_to_float2(ru2.w);
            const float2 fr3_3 = bf16x2_raw_to_float2(ru3.w);
            const float  a6    = po * xf[6] + c0 * fr0_3.x + c1 * fr1_3.x + c2 * fr2_3.x + c3 * fr3_3.x;
            const float  a7    = po * xf[7] + c0 * fr0_3.y + c1 * fr1_3.y + c2 * fr2_3.y + c3 * fr3_3.y;

            *reinterpret_cast<uint4*>(o_row + o * D + col) = make_uint4(float2_to_bf16x2_raw(a0, a1),
                                                                        float2_to_bf16x2_raw(a2, a3),
                                                                        float2_to_bf16x2_raw(a4, a5),
                                                                        float2_to_bf16x2_raw(a6, a7));
        }
    }
}

// Generic vectorized kernel for H <= kMaxVecH, 16-byte aligned pointers, D % 8 == 0.
__global__ void dsv4_mhc_post_generic_vec8_kernel(const __nv_bfloat16* __restrict__ x_ptr,
                                                  const __nv_bfloat16* __restrict__ res_ptr,
                                                  const void* __restrict__ post_ptr,
                                                  const void* __restrict__ comb_ptr,
                                                  __nv_bfloat16* __restrict__ out_ptr,
                                                  int64_t   M,
                                                  int       H,
                                                  int64_t   D,
                                                  int64_t   D_vec,
                                                  FastDiv32 fdiv_tiles,
                                                  bool      post_is_bf16,
                                                  bool      comb_is_bf16) {
    const int64_t total_work = M * static_cast<int64_t>(fdiv_tiles.divisor);
    for (int64_t work = static_cast<int64_t>(blockIdx.x); work < total_work; work += static_cast<int64_t>(gridDim.x)) {
        int64_t t        = 0;
        int64_t tile_idx = 0;
        fdiv_tiles.divmod(work, t, tile_idx);

        const int64_t v = tile_idx * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
        if (v >= D_vec) {
            continue;
        }
        const int64_t col = v * 8;

        const __nv_bfloat16* x_row = x_ptr + t * D;
        const __nv_bfloat16* r_row = res_ptr + t * static_cast<int64_t>(H) * D;
        __nv_bfloat16*       o_row = out_ptr + t * static_cast<int64_t>(H) * D;

        const uint4 xu = __ldg(reinterpret_cast<const uint4*>(x_row + col));
        uint4       ru[kMaxVecH];
        for (int i = 0; i < H; ++i) {
            ru[i] = *reinterpret_cast<const uint4*>(r_row + static_cast<int64_t>(i) * D + col);
        }

        float xf[8];
        unpack_u4_bf16(xu, xf);

        for (int o = 0; o < H; ++o) {
            const float po = load_coeff_f32(post_ptr, t * H + o, post_is_bf16);
            float       acc[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                acc[k] = po * xf[k];
            }
            for (int i = 0; i < H; ++i) {
                const float ci = load_coeff_f32(comb_ptr, t * H * H + i * H + o, comb_is_bf16);
                float       rf[8];
                unpack_u4_bf16(ru[i], rf);
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    acc[k] += ci * rf[k];
                }
            }
            *reinterpret_cast<uint4*>(o_row + static_cast<int64_t>(o) * D + col) = pack_u4_bf16(acc);
        }
    }
}

// Scalar BF16 fallback kernel for arbitrary D (including D % 8 != 0 or unaligned views).
__global__ void dsv4_mhc_post_scalar_kernel(const __nv_bfloat16* __restrict__ x_ptr,
                                            const __nv_bfloat16* __restrict__ res_ptr,
                                            const void* __restrict__ post_ptr,
                                            const void* __restrict__ comb_ptr,
                                            __nv_bfloat16* __restrict__ out_ptr,
                                            int64_t   M,
                                            int       H,
                                            int64_t   D,
                                            FastDiv32 fdiv_tiles,
                                            bool      post_is_bf16,
                                            bool      comb_is_bf16) {
    const int64_t total_work = M * static_cast<int64_t>(fdiv_tiles.divisor);
    for (int64_t work = static_cast<int64_t>(blockIdx.x); work < total_work; work += static_cast<int64_t>(gridDim.x)) {
        int64_t t        = 0;
        int64_t tile_idx = 0;
        fdiv_tiles.divmod(work, t, tile_idx);

        const int64_t d = tile_idx * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
        if (d >= D) {
            continue;
        }

        const uint16_t    x_raw = __ldg(reinterpret_cast<const uint16_t*>(x_ptr) + t * D + d);
        __nv_bfloat16_raw x_b;
        x_b.x               = x_raw;
        const float   xv    = __bfloat162float(__nv_bfloat16(x_b));
        const int64_t r_off = t * static_cast<int64_t>(H) * D + d;

        float r_cache[kMaxScalarH];
        if (H <= kMaxScalarH) {
            for (int i = 0; i < H; ++i) {
                r_cache[i] = __bfloat162float(res_ptr[r_off + static_cast<int64_t>(i) * D]);
            }
            for (int o = 0; o < H; ++o) {
                float acc = load_coeff_f32(post_ptr, t * H + o, post_is_bf16) * xv;
                for (int i = 0; i < H; ++i) {
                    acc += load_coeff_f32(comb_ptr, t * H * H + i * H + o, comb_is_bf16) * r_cache[i];
                }
                out_ptr[r_off + static_cast<int64_t>(o) * D] = __float2bfloat16_rn(acc);
            }
        } else {
            // Only safe when H <= kMaxScalarH if out aliases residual (checked on host).
            for (int o = 0; o < H; ++o) {
                float acc = load_coeff_f32(post_ptr, t * H + o, post_is_bf16) * xv;
                for (int i = 0; i < H; ++i) {
                    acc += load_coeff_f32(comb_ptr, t * H * H + i * H + o, comb_is_bf16)
                           * __bfloat162float(res_ptr[r_off + static_cast<int64_t>(i) * D]);
                }
                out_ptr[r_off + static_cast<int64_t>(o) * D] = __float2bfloat16_rn(acc);
            }
        }
    }
}

bool ranges_overlap(uintptr_t a_start, size_t a_bytes, uintptr_t b_start, size_t b_bytes) {
    if (a_bytes == 0 || b_bytes == 0) {
        return false;
    }
    const uintptr_t a_end = a_start + a_bytes;
    const uintptr_t b_end = b_start + b_bytes;
    return a_start < b_end && b_start < a_end;
}

bool is_aligned(const void* ptr, uintptr_t align) {
    return (reinterpret_cast<uintptr_t>(ptr) & (align - 1ULL)) == 0ULL;
}

}  // namespace
#endif  // USE_ROCM

void dsv4_mhc_post_fwd(const torch::Tensor& x,
                       const torch::Tensor& residual,
                       const torch::Tensor& post,
                       const torch::Tensor& comb,
                       torch::Tensor&       out) {
#ifndef USE_ROCM
    TORCH_CHECK(x.is_cuda() && residual.is_cuda() && post.is_cuda() && comb.is_cuda() && out.is_cuda(),
                "dsv4_mhc_post_fwd requires CUDA tensors");
    const auto dev = x.device();
    TORCH_CHECK(residual.device() == dev && post.device() == dev && comb.device() == dev && out.device() == dev,
                "dsv4_mhc_post_fwd requires all tensors on the same CUDA device");
    TORCH_CHECK(x.is_contiguous() && residual.is_contiguous() && post.is_contiguous() && comb.is_contiguous()
                    && out.is_contiguous(),
                "dsv4_mhc_post_fwd requires contiguous tensors");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && residual.scalar_type() == torch::kBFloat16
                    && out.scalar_type() == torch::kBFloat16,
                "dsv4_mhc_post_fwd requires bfloat16 x, residual, and out");
    TORCH_CHECK(post.scalar_type() == torch::kFloat32 || post.scalar_type() == torch::kBFloat16,
                "dsv4_mhc_post_fwd requires float32 or bfloat16 post");
    TORCH_CHECK(comb.scalar_type() == torch::kFloat32 || comb.scalar_type() == torch::kBFloat16,
                "dsv4_mhc_post_fwd requires float32 or bfloat16 comb");

    TORCH_CHECK(x.dim() >= 2, "dsv4_mhc_post_fwd requires x.dim() >= 2");
    TORCH_CHECK(residual.dim() == x.dim() + 1, "dsv4_mhc_post_fwd requires residual.dim() == x.dim() + 1");
    TORCH_CHECK(out.sizes() == residual.sizes(), "dsv4_mhc_post_fwd requires out.shape == residual.shape");

    const int64_t prefix_dims = x.dim() - 1;
    int64_t       M           = 1;
    for (int64_t i = 0; i < prefix_dims; ++i) {
        TORCH_CHECK(residual.size(i) == x.size(i), "dsv4_mhc_post_fwd x/residual leading dim mismatch");
        M *= x.size(i);
    }
    const int64_t H_i64 = residual.size(-2);
    const int64_t D     = x.size(-1);
    TORCH_CHECK(residual.size(-1) == D, "dsv4_mhc_post_fwd x/residual dim mismatch");
    TORCH_CHECK(H_i64 >= 1 && D >= 0, "dsv4_mhc_post_fwd requires H >= 1 and D >= 0");
    const int H = static_cast<int>(H_i64);

    // Validate post ([..., H] or [..., H, 1])
    TORCH_CHECK(post.dim() == x.dim() || post.dim() == x.dim() + 1,
                "dsv4_mhc_post_fwd requires post shape [..., H] or [..., H, 1]");
    for (int64_t i = 0; i < prefix_dims; ++i) {
        TORCH_CHECK(post.size(i) == x.size(i), "dsv4_mhc_post_fwd post leading dim mismatch");
    }
    TORCH_CHECK(post.size(prefix_dims) == H_i64, "dsv4_mhc_post_fwd post hc_mult mismatch");
    if (post.dim() == x.dim() + 1) {
        TORCH_CHECK(post.size(prefix_dims + 1) == 1, "dsv4_mhc_post_fwd requires post trailing dim == 1");
    }

    // Validate comb ([..., H, H])
    TORCH_CHECK(comb.dim() == x.dim() + 1, "dsv4_mhc_post_fwd requires comb shape [..., H, H]");
    for (int64_t i = 0; i < prefix_dims; ++i) {
        TORCH_CHECK(comb.size(i) == x.size(i), "dsv4_mhc_post_fwd comb leading dim mismatch");
    }
    TORCH_CHECK(comb.size(prefix_dims) == H_i64 && comb.size(prefix_dims + 1) == H_i64,
                "dsv4_mhc_post_fwd comb hc_mult mismatch");

    if (M == 0 || D == 0) {
        return;
    }

    // Exact in-place out == residual (or out == x when H == 1) is safe because
    // each thread loads all inputs for (t, col..col+k) before writing any output.
    const auto   x_addr   = reinterpret_cast<uintptr_t>(x.data_ptr());
    const auto   r_addr   = reinterpret_cast<uintptr_t>(residual.data_ptr());
    const auto   p_addr   = reinterpret_cast<uintptr_t>(post.data_ptr());
    const auto   c_addr   = reinterpret_cast<uintptr_t>(comb.data_ptr());
    const auto   o_addr   = reinterpret_cast<uintptr_t>(out.data_ptr());
    const size_t x_bytes  = static_cast<size_t>(x.numel()) * x.element_size();
    const size_t r_bytes  = static_cast<size_t>(residual.numel()) * residual.element_size();
    const size_t p_bytes  = static_cast<size_t>(post.numel()) * post.element_size();
    const size_t c_bytes  = static_cast<size_t>(comb.numel()) * comb.element_size();
    const size_t o_bytes  = static_cast<size_t>(out.numel()) * out.element_size();
    const bool   alias_r  = (o_addr == r_addr);
    const bool   alias_x1 = (H == 1 && o_addr == x_addr);
    if (!alias_r) {
        TORCH_CHECK(!ranges_overlap(o_addr, o_bytes, r_addr, r_bytes),
                    "dsv4_mhc_post_fwd does not support shifted out/residual memory overlap");
    }
    if (!alias_x1) {
        TORCH_CHECK(!ranges_overlap(o_addr, o_bytes, x_addr, x_bytes),
                    "dsv4_mhc_post_fwd does not support out/x memory overlap");
    }
    TORCH_CHECK(!ranges_overlap(o_addr, o_bytes, p_addr, p_bytes),
                "dsv4_mhc_post_fwd does not support out/post memory overlap");
    TORCH_CHECK(!ranges_overlap(o_addr, o_bytes, c_addr, c_bytes),
                "dsv4_mhc_post_fwd does not support out/comb memory overlap");

    const auto* x_ptr        = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    const auto* r_ptr        = reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr());
    auto*       o_ptr        = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const void* p_ptr        = post.data_ptr();
    const void* c_ptr        = comb.data_ptr();
    const bool  post_is_bf16 = (post.scalar_type() == torch::kBFloat16);
    const bool  comb_is_bf16 = (comb.scalar_type() == torch::kBFloat16);

    const bool can_vec8 = (H <= kMaxVecH) && ((D & 7LL) == 0) && is_aligned(x_ptr, 16) && is_aligned(r_ptr, 16)
                          && is_aligned(o_ptr, 16);
    const bool can_regpipe = can_vec8 && (H == 4) && (M >= 1024) && !post_is_bf16 && !comb_is_bf16
                             && is_aligned(p_ptr, 16) && is_aligned(c_ptr, 16) && (D == 4096 || D == 7168);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (can_regpipe) {
        const int grid_x = static_cast<int>(M);
        if (D == 4096) {
            dsv4_mhc_post_h4_cpasync_kernel<4096><<<grid_x, 128, 0, stream>>>(
                x_ptr, r_ptr, reinterpret_cast<const float*>(p_ptr), reinterpret_cast<const float*>(c_ptr), o_ptr, M);
        } else {
            dsv4_mhc_post_h4_cpasync_kernel<7168><<<grid_x, 128, 0, stream>>>(
                x_ptr, r_ptr, reinterpret_cast<const float*>(p_ptr), reinterpret_cast<const float*>(c_ptr), o_ptr, M);
        }
    } else if (can_vec8) {
        const int64_t D_vec      = D >> 3;
        const int     threads    = M <= 8 ? 32 : (M <= 64 ? 64 : 128);
        const int64_t col_tiles  = (D_vec + threads - 1) / threads;
        const int64_t total_work = M * col_tiles;
        const int     grid_x     = static_cast<int>(std::min<int64_t>(total_work, 2097152LL));
        const auto    fdiv       = make_fast_div32(col_tiles);
        if (H == 4) {
            dsv4_mhc_post_h4_vec8_kernel<<<grid_x, threads, 0, stream>>>(
                x_ptr, r_ptr, p_ptr, c_ptr, o_ptr, M, D, D_vec, fdiv, post_is_bf16, comb_is_bf16);
        } else {
            dsv4_mhc_post_generic_vec8_kernel<<<grid_x, threads, 0, stream>>>(
                x_ptr, r_ptr, p_ptr, c_ptr, o_ptr, M, H, D, D_vec, fdiv, post_is_bf16, comb_is_bf16);
        }
    } else {
        if (alias_r) {
            TORCH_CHECK(H <= kMaxScalarH, "dsv4_mhc_post_fwd in-place scalar fallback supports H <= 32");
        }
        const int     threads    = D <= 256 ? 128 : 256;
        const int64_t col_tiles  = (D + threads - 1) / threads;
        const int64_t total_work = M * col_tiles;
        const int     grid_x     = static_cast<int>(std::min<int64_t>(total_work, 2097152LL));
        const auto    fdiv       = make_fast_div32(col_tiles);
        dsv4_mhc_post_scalar_kernel<<<grid_x, threads, 0, stream>>>(
            x_ptr, r_ptr, p_ptr, c_ptr, o_ptr, M, H, D, fdiv, post_is_bf16, comb_is_bf16);
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dsv4_mhc_post_fwd launch failed: ", cudaGetErrorString(err));
#else
    (void)x;
    (void)residual;
    (void)post;
    (void)comb;
    (void)out;
    TORCH_CHECK(false, "dsv4_mhc_post_fwd CUDA op is not supported on ROCm");
#endif
}

}  // namespace torch_ext
