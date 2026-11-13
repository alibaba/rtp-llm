/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 */

// Production AttnRes forward for Blackwell (SM100).
//
// Warp-specialized online softmax + residual + RMSNorm:
//   - 1 producer warp issues cp.async.bulk row loads into shared memory.
//   - 8 consumer warps compute reductions and output.
//   - Q=res_weight*rms_weight remains in registers across persistent tokens.
//   - V rows are converted once and cached as FP32 in TMEM between passes.
//
// Supported specialization: H=7168, 0<=num_blocks<=8, and token-major
// block residual storage.

// Adapted from vLLM c3b48446349569512749db7f6e2164aa8a33437d,
// csrc/libtorch_stable/kimi_k3/attn_res_kernel.cu (Apache-2.0).
// Device algorithm unchanged; RTP uses the regular ATen ABI.
#include "rtp_llm/models_py/bindings/cuda/kernels/attn_res.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include "rtp_llm/models_py/bindings/cuda/kernels/nvfp4_quant.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <type_traits>

using bf16_t = __nv_bfloat16;

namespace sm100 {
namespace fwd_prod_v2 {

constexpr int K_TILE                     = 1024;
constexpr int N_CHUNK_DEFAULT            = 4;
constexpr int CHUNK_DEPTH                = 2;
constexpr int BLK                        = 288;       // 1 producer warp + 8 consumer warps
constexpr int CONSUMER_THREADS           = BLK - 32;  // 256
constexpr int CONSUMER_WARPS             = CONSUMER_THREADS / 32;
constexpr int CONSUMER_GROUPS            = 2;  // two 128-thread consumer groups
constexpr int CONSUMER_THREADS_PER_GROUP = CONSUMER_THREADS / CONSUMER_GROUPS;
constexpr int FIRST_USER_NAMED_BARRIER   = 8;

__device__ __forceinline__ const bf16_t* residual_addr(const bf16_t* block_res,
                                                       const bf16_t* layer_res,
                                                       int           source,
                                                       int           N,
                                                       int           token,
                                                       int           block_stride_m,
                                                       int           block_stride_r,
                                                       int           H) {
    if (source < N - 1) {
        return block_res + static_cast<long long>(token) * block_stride_m + source * block_stride_r;
    }
    return layer_res + static_cast<long long>(token) * H;
}

__device__ __forceinline__ uint32_t elect_one_sync() {
    uint32_t pred   = 0;
    uint32_t laneid = 0;
    asm volatile("{\n"
                 ".reg .b32 %%rx;\n"
                 ".reg .pred %%px;\n"
                 "     elect.sync %%rx|%%px, %2;\n"
                 "@%%px mov.s32 %1, 1;\n"
                 "     mov.s32 %0, %%rx;\n"
                 "}\n"
                 : "+r"(laneid), "+r"(pred)
                 : "r"(0xffffffff));
    return pred;
}

__device__ __forceinline__ void mbarrier_init(uint64_t& barrier, int thread_count) {
    uint32_t const barrier_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(barrier_addr), "r"(thread_count));
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t& barrier, uint32_t bytes) {
    uint32_t const barrier_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(barrier_addr), "r"(bytes));
}

__device__ __forceinline__ void mbarrier_wait(uint64_t& barrier, int phase) {
    uint32_t const barrier_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "WAIT:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 "@p bra DONE;\n"
                 "bra WAIT;\n"
                 "DONE:\n"
                 "}\n" ::"r"(barrier_addr),
                 "r"(phase)
                 : "memory");
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t& barrier) {
    uint32_t const barrier_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
    asm volatile("{\n"
                 ".reg .b64 state;\n"
                 "mbarrier.arrive.shared::cta.b64 state, [%0];\n"
                 "}\n" ::"r"(barrier_addr)
                 : "memory");
}

__device__ __forceinline__ void fence_mbarrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void named_barrier_sync(uint32_t num_threads, uint32_t user_barrier_id) {
    asm volatile("bar.sync %0, %1;" ::"r"(user_barrier_id + FIRST_USER_NAMED_BARRIER), "r"(num_threads) : "memory");
}

__device__ __forceinline__ void tmem_allocate(int num_columns, uint32_t* dst) {
    uint32_t const dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(dst_addr), "r"(num_columns));
}

__device__ __forceinline__ void tmem_free(uint32_t tmem_ptr, int num_columns) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_ptr), "r"(num_columns));
}

__device__ __forceinline__ void tmem_release_allocation_lock() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}

__device__ __forceinline__ void tmem_store_wait() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

template<int N, typename T>
__device__ __forceinline__ void tmem_load(uint32_t src_addr, T* dst) {
    uint32_t* values = reinterpret_cast<uint32_t*>(dst);
    if constexpr (N == 8) {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32"
                     "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                     : "=r"(values[0]),
                       "=r"(values[1]),
                       "=r"(values[2]),
                       "=r"(values[3]),
                       "=r"(values[4]),
                       "=r"(values[5]),
                       "=r"(values[6]),
                       "=r"(values[7])
                     : "r"(src_addr));
    } else {
        static_assert(N == 4, "AttnRes TMEM helpers support x4 and x8");
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32"
                     "{%0, %1, %2, %3}, [%4];\n"
                     : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3])
                     : "r"(src_addr));
    }
}

template<int N, typename T>
__device__ __forceinline__ void tmem_store(uint32_t dst_addr, T* src) {
    uint32_t* values = reinterpret_cast<uint32_t*>(src);
    if constexpr (N == 8) {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32"
                     "[%8], {%0, %1, %2, %3, %4, %5, %6, %7};\n" ::"r"(values[0]),
                     "r"(values[1]),
                     "r"(values[2]),
                     "r"(values[3]),
                     "r"(values[4]),
                     "r"(values[5]),
                     "r"(values[6]),
                     "r"(values[7]),
                     "r"(dst_addr));
    } else {
        static_assert(N == 4, "AttnRes TMEM helpers support x4 and x8");
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32"
                     "[%4], {%0, %1, %2, %3};\n" ::"r"(values[0]),
                     "r"(values[1]),
                     "r"(values[2]),
                     "r"(values[3]),
                     "r"(dst_addr));
    }
}

__device__ __forceinline__ float2 float2_add(const float2& a, const float2& b) {
    float2 result;
    asm volatile("add.rn.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(result))
                 : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)));
    return result;
}

__device__ __forceinline__ float2 float2_mul(const float2& a, const float2& b) {
    float2 result;
    asm volatile("mul.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(result))
                 : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)));
    return result;
}

__device__ __forceinline__ float2 float2_fma(const float2& a, const float2& b, const float2& c) {
    float2 result;
    asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(result))
                 : "l"(reinterpret_cast<uint64_t const&>(a)),
                   "l"(reinterpret_cast<uint64_t const&>(b)),
                   "l"(reinterpret_cast<uint64_t const&>(c)));
    return result;
}

template<int NC>
struct FwdSmemPlan {
    alignas(16) uint64_t bar_ready[CHUNK_DEPTH];
    alignas(16) uint64_t bar_consumed[CHUNK_DEPTH];
    alignas(16) uint64_t bar_output_norm_ready;
    alignas(16) float2 ws_stats[CONSUMER_WARPS][NC];
    uint32_t tmem_base;
};

__device__ __forceinline__ void cp_async_bulk(void* smem_dst, const void* gmem_src, int bytes, uint64_t& mbar) {
    uint32_t const s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    uint32_t const m = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], "
                 "[%1], %2, [%3];\n" ::"r"(s),
                 "l"(gmem_src),
                 "r"(bytes),
                 "r"(m)
                 : "memory");
}

template<int H, int VEC, typename Packed>
__device__ __forceinline__ void
store_fp8_group(const bf16_t* values, uint8_t* output, uint8_t* scales, int row, int rows, int col) {
    float amax = 1.e-4f;
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
        amax = fmaxf(amax, fabsf(__bfloat162float(values[j])));
    }
#pragma unroll
    for (int offset = 128 / VEC / 2; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset, 128 / VEC));
    }
    // Match the standalone CUDA quantizer's ceil(log2(x)) on powers of two.
    const uint32_t bits      = __float_as_uint(amax * (1.f / 448.f));
    const uint32_t scale_exp = (bits >> 23) + ((bits & 0x7fffff) != 0);
    const float    inv_scale = __uint_as_float((254 - scale_exp) << 23);
    if (col % 128 == 0) {
        const int group        = col / 128;
        const int aligned_rows = (rows + 3) & ~3;
        uint8_t*  group_scale  = scales + ((long long)(group / 4) * aligned_rows + row) * 4 + group % 4;
        *group_scale           = scale_exp;
        if (row == rows - 1) {
            for (int i = 1; i <= aligned_rows - rows; ++i) {
                group_scale[i * 4] = 127;
            }
        }
    }
    Packed packed = 0;
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
        const float value = fminf(fmaxf(__bfloat162float(values[j]) * inv_scale, -448.f), 448.f);
        packed |= static_cast<Packed>(__nv_fp8_e4m3(value).__x) << (8 * j);
    }
    *reinterpret_cast<Packed*>(output + (long long)row * H + col) = packed;
}

// All CTAs have produced their BF16 rows and maxima. The same grid finishes
// the scale and writes the CUTLASS 128x4 layout directly into the wire buffer.
template<int H>
__device__ __forceinline__ void
attn_res_nvfp4_packet(const bf16_t* output, int rows, const float* amax_partials, uint8_t* wire) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1100
    cooperative_groups::this_grid().sync();
    __shared__ float inverse;
    constexpr int    vectors_per_row = H / CVT_FP4_ELTS_PER_THREAD;
    const int        padded_rows     = (rows + 127) / 128 * 128;
    auto*            codes           = reinterpret_cast<uint32_t*>(wire);
    auto*            scales          = wire + static_cast<int64_t>(rows) * (H / 2);
    auto*            tail            = reinterpret_cast<float*>(scales + static_cast<int64_t>(padded_rows) * (H / 16));
    if (threadIdx.x < 32) {
        uint32_t maximum = 0;
        for (unsigned i = threadIdx.x; i < gridDim.x; i += 32) {
            maximum = max(maximum, __float_as_uint(amax_partials[i]));
        }
        maximum = __reduce_max_sync(0xffffffff, maximum);
        if (threadIdx.x == 0) {
            const float scale = __fmul_rn(fmaxf(__uint_as_float(maximum), 1e-8f), 1.0f / 2688.0f);
            inverse           = __frcp_rn(scale);
            if (blockIdx.x == 0) {
                tail[0] = scale;
                tail[1] = tail[2] = tail[3] = 0.f;
            }
        }
    }
    __syncthreads();
    struct alignas(16) AlignedInput {
        PackedVec<bf16_t> values;
    };
    auto* vectors = reinterpret_cast<const AlignedInput*>(output);
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < static_cast<int64_t>(padded_rows) * vectors_per_row;
         i += gridDim.x * blockDim.x) {
        const int         row = i / vectors_per_row;
        const int         col = i % vectors_per_row;
        PackedVec<bf16_t> values{};
        if (row < rows) {
            values = vectors[i].values;
        }
        auto*      sf     = cvt_quant_to_fp4_get_sf_out_offset<uint8_t, 2>(row, col, H, scales);
        const auto packed = cvt_warp_fp16_to_fp4<bf16_t, false, true>(values, inverse, sf);
        if (row < rows) {
            codes[i] = packed;
        }
    }
#endif
}

// The consumer warps own one complete normalized row. Keep its rounded BF16
// values in registers until the CTA has computed the row's dequant scale.
template<int H, int SLICES>
__device__ __forceinline__ void store_nvfp4_row(const uint4 (&values)[SLICES],
                                                uint32_t  packed_amax,
                                                uint32_t* warp_amax,
                                                int       row,
                                                int       group,
                                                int       comp_wid,
                                                int       ct_in_group,
                                                int       lane,
                                                uint32_t* codes,
                                                uint8_t*  scales,
                                                float*    row_scales) {
    static_assert(H == 7168);
    const uint32_t local   = max(packed_amax & 0xffffu, packed_amax >> 16);
    const uint32_t partial = __reduce_max_sync(0xffffffff, local);
    if (lane == 0) {
        warp_amax[comp_wid] = partial;
    }
    named_barrier_sync(CONSUMER_THREADS, 0);
    const uint32_t maximum = __reduce_max_sync(0xffffffff, lane < CONSUMER_WARPS ? warp_amax[lane] : 0u);
    const float    scale   = __fmul_rn(fmaxf(__uint_as_float(maximum << 16), 1.e-8f), 1.f / 2688.f);
    const float    inverse = __frcp_rn(scale);
    if (comp_wid == 0 && lane == 0) {
        row_scales[row] = scale;
    }
    uint32_t* row_codes = codes + static_cast<int64_t>(row) * (H / 8);
#pragma unroll
    for (int si = 0; si < SLICES - 1; ++si) {
        const int col      = (si * CONSUMER_GROUPS + group) * K_TILE + ct_in_group * 8;
        auto      vec      = *reinterpret_cast<const PackedVec<bf16_t>*>(&values[si]);
        auto*     sf       = cvt_quant_to_fp4_get_sf_out_offset<uint8_t, 2>(row, col / 8, H, scales);
        row_codes[col / 8] = cvt_warp_fp16_to_fp4<bf16_t, false, true>(vec, inverse, sf);
    }

    // The last tile has four values per lane. Two lanes form the same 8-value
    // vector, and the next pair supplies the other half of the 16-value group.
    const uint4    tail    = values[SLICES - 1];
    const uint32_t other_x = __shfl_xor_sync(0xffffffff, tail.x, 1);
    const uint32_t other_y = __shfl_xor_sync(0xffffffff, tail.y, 1);
    const bool     even    = (lane & 1) == 0;
    const uint4    joined =
        even ? make_uint4(tail.x, tail.y, other_x, other_y) : make_uint4(other_x, other_y, tail.x, tail.y);
    auto      vec = *reinterpret_cast<const PackedVec<bf16_t>*>(&joined);
    const int col = 6 * K_TILE + group * (K_TILE / 2) + ct_in_group * 4;
    auto*     sf = (lane & 3) == 0 ? cvt_quant_to_fp4_get_sf_out_offset<uint8_t, 1>(row, col / 16, H, scales) : nullptr;
    const uint32_t packed = cvt_warp_fp16_to_fp4<bf16_t, false, true, 2>(vec, inverse, sf);
    if (even) {
        row_codes[col / 8] = packed;
    }
}

template<int  H,
         int  N,
         int  NC                 = N_CHUNK_DEFAULT,
         int  B                  = 1,
         bool RELEASE_TMEM       = false,
         bool STATIC_COMMON_PATH = false,
         bool HAS_OUTPUT_NORM    = false,
         bool STORE_FP8          = false,
         bool STORE_NVFP4_AMAX   = false,
         bool STORE_NVFP4_ROWS   = false>
__global__ void __launch_bounds__(BLK, 1) attn_res_fwd_online_v2_kernel(bf16_t* __restrict__ block_res,
                                                                        bf16_t* __restrict__ layer_res,
                                                                        const bf16_t* __restrict__ delta,
                                                                        const bf16_t* __restrict__ res_w,
                                                                        const bf16_t* __restrict__ rms_w,
                                                                        bf16_t* __restrict__ output,
                                                                        int   T,
                                                                        int   block_stride_m,
                                                                        int   block_stride_r,
                                                                        float rms_eps,
                                                                        const bf16_t* __restrict__ output_norm_weight,
                                                                        float    output_norm_eps,
                                                                        int      block_write_idx,
                                                                        uint8_t* fp8_output,
                                                                        uint8_t* fp8_scales,
                                                                        float*   amax_partials,
                                                                        uint8_t* nvfp4_packet,
                                                                        const bf16_t* __restrict__ shared_delta) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1100
    constexpr float LOG2_E  = 1.4426950408889634f;
    constexpr int   N_CHUNK = NC;
    // The two-source specialization only consumes half of the TMEM columns.
    constexpr int TMEM_COLS_ALLOC       = NC == 2 ? 128 : 256;
    constexpr int NUM_BUFS              = CHUNK_DEPTH * NC;
    constexpr int NHT                   = H / K_TILE;
    constexpr int SLICES_PER_GROUP      = (NHT + CONSUMER_GROUPS - 1) / CONSUMER_GROUPS;
    constexpr int VEC                   = 8;
    constexpr int ACC_PER_THREAD        = H == 7168 ? 28 : SLICES_PER_GROUP * VEC;
    constexpr int TMEM_V_COLS_PER_GROUP = SLICES_PER_GROUP * N_CHUNK * VEC;
    constexpr int TMEM_V_COLS_TOTAL     = CONSUMER_GROUPS * TMEM_V_COLS_PER_GROUP;
    static_assert(TMEM_V_COLS_TOTAL <= TMEM_COLS_ALLOC);
    static_assert(H >= 4096 && H <= 8192);
    static_assert(H % K_TILE == 0);
    static_assert(!STATIC_COMMON_PATH || HAS_OUTPUT_NORM);
    static_assert(!(STORE_FP8 || STORE_NVFP4_AMAX || STORE_NVFP4_ROWS) || HAS_OUTPUT_NORM);
    __shared__ uint32_t warp_amax[CONSUMER_WARPS];
    uint32_t            packed_amax = 0;

    const int tid              = threadIdx.x;
    const int wid              = tid >> 5;
    const int lane             = tid & 31;
    const int TB               = T * B;
    const int padded_rows      = (TB + 127) / 128 * 128;
    uint8_t*  row_block_scales = nullptr;
    float*    row_scales       = nullptr;
    if constexpr (STORE_NVFP4_ROWS) {
        row_block_scales = nvfp4_packet + static_cast<int64_t>(TB) * (H / 2);
        row_scales       = reinterpret_cast<float*>(row_block_scales + static_cast<int64_t>(padded_rows) * (H / 16));
    }
    const int     num_ctas   = gridDim.x;
    const bool    has_delta  = STATIC_COMMON_PATH || delta != nullptr;
    const int     write_idx  = STATIC_COMMON_PATH ? -1 : block_write_idx;
    constexpr int num_chunks = (N + N_CHUNK - 1) / N_CHUNK;

    const int comp_wid    = wid - 1;
    const int comp_tid    = tid - 32;
    const int group       = (comp_wid >= 4) ? 1 : 0;
    const int ct_in_group = (comp_tid >= 0) ? (comp_tid & (CONSUMER_THREADS_PER_GROUP - 1)) : -1;
    const int k_local     = ct_in_group * VEC;

    constexpr size_t  V_BYTES           = (size_t)NUM_BUFS * H * sizeof(bf16_t);
    constexpr size_t  DELTA_BYTES       = (size_t)CHUNK_DEPTH * H * sizeof(bf16_t);
    constexpr size_t  OUTPUT_NORM_BYTES = HAS_OUTPUT_NORM ? (size_t)H * sizeof(bf16_t) : 0;
    extern __shared__ __align__(16) char smem_raw[];
    bf16_t*                              v_bufs          = reinterpret_cast<bf16_t*>(smem_raw);  // [NUM_BUFS][H]
    bf16_t*                              delta_bufs      = reinterpret_cast<bf16_t*>(smem_raw + V_BYTES);
    bf16_t*                              output_norm_buf = reinterpret_cast<bf16_t*>(smem_raw + V_BYTES + DELTA_BYTES);
    FwdSmemPlan<NC>& plan = *reinterpret_cast<FwdSmemPlan<NC>*>(smem_raw + V_BYTES + DELTA_BYTES + OUTPUT_NORM_BYTES);

    auto slot_of       = [](long long gci, int n) { return (int)(gci % CHUNK_DEPTH) * N_CHUNK + n; };
    auto phase_of      = [](long long gci) { return (int)((gci / CHUNK_DEPTH) & 1); };
    auto buf_ptr       = [&](int slot) -> bf16_t* { return v_bufs + slot * H; };
    auto delta_buf_ptr = [&](int chunk_slot) -> bf16_t* { return delta_bufs + chunk_slot * H; };

    if (wid == 0 && elect_one_sync()) {
#pragma unroll
        for (int i = 0; i < CHUNK_DEPTH; i++) {
            mbarrier_init(plan.bar_ready[i], 1);
            mbarrier_init(plan.bar_consumed[i], CONSUMER_WARPS);
        }
        if constexpr (HAS_OUTPUT_NORM) {
            mbarrier_init(plan.bar_output_norm_ready, 1);
        }
        fence_mbarrier_init();
    }

    if (wid == 1) {
        tmem_allocate(TMEM_COLS_ALLOC, &plan.tmem_base);
        if constexpr (RELEASE_TMEM) {
            tmem_release_allocation_lock();
        }
    }
    __syncthreads();

    if constexpr (HAS_OUTPUT_NORM) {
        if (wid == 0 && elect_one_sync()) {
            mbarrier_expect_tx(plan.bar_output_norm_ready, H * (int)sizeof(bf16_t));
            cp_async_bulk(output_norm_buf, output_norm_weight, H * sizeof(bf16_t), plan.bar_output_norm_ready);
        }
    }

    const uint32_t my_v_tmem = comp_tid >= 0 ? plan.tmem_base + group * TMEM_V_COLS_PER_GROUP : 0;
    float          q_cache[ACC_PER_THREAD];
    if (comp_tid >= 0) {
#pragma unroll
        for (int si = 0; si < SLICES_PER_GROUP; si++) {
            if constexpr (H == 7168) {
                if (si == SLICES_PER_GROUP - 1) {
                    int   h_base = 6 * K_TILE + group * (K_TILE / 2) + ct_in_group * 4;
                    int2  rms_v  = *reinterpret_cast<const int2*>(rms_w + h_base);
                    int2  res_v  = *reinterpret_cast<const int2*>(res_w + h_base);
                    auto* rms2   = reinterpret_cast<__nv_bfloat162*>(&rms_v);
                    auto* res2   = reinterpret_cast<__nv_bfloat162*>(&res_v);
#pragma unroll
                    for (int k = 0; k < 2; k++) {
                        float2 rf                     = __bfloat1622float2(rms2[k]);
                        float2 sf                     = __bfloat1622float2(res2[k]);
                        q_cache[si * VEC + 2 * k]     = rf.x * sf.x;
                        q_cache[si * VEC + 2 * k + 1] = rf.y * sf.y;
                    }
                    continue;
                }
            }
            int dt = si * CONSUMER_GROUPS + group;
            if (dt >= NHT)
                continue;
            int   h_base = dt * K_TILE + k_local;
            int4  rms_v  = *reinterpret_cast<const int4*>(rms_w + h_base);
            int4  res_v  = *reinterpret_cast<const int4*>(res_w + h_base);
            auto* rms2   = reinterpret_cast<__nv_bfloat162*>(&rms_v);
            auto* res2   = reinterpret_cast<__nv_bfloat162*>(&res_v);
#pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 rf                     = __bfloat1622float2(rms2[k]);
                float2 sf                     = __bfloat1622float2(res2[k]);
                q_cache[si * VEC + 2 * k]     = rf.x * sf.x;
                q_cache[si * VEC + 2 * k + 1] = rf.y * sf.y;
            }
        }
    }

    if (wid == 0) {
        if (elect_one_sync()) {
            long long gci        = 0;
            bool      pdl_waited = false;
            for (int tb = blockIdx.x; tb < TB; tb += num_ctas) {
                const int t = tb / B;
                for (int ci = 0; ci < num_chunks; ci++, gci++) {
                    int ns         = ci * N_CHUNK;
                    int an         = min(N_CHUNK, N - ns);
                    int chunk_slot = (int)(gci % CHUNK_DEPTH);
                    int pc         = phase_of(gci);
                    mbarrier_wait(plan.bar_consumed[chunk_slot], pc ^ 1);
                    int transaction_bytes = an * H * (int)sizeof(bf16_t);
                    if (has_delta) {
                        int prefix_n = N - 1 - ns;
                        if (prefix_n >= 0 && prefix_n < an) {
                            transaction_bytes += H * (int)sizeof(bf16_t);
                        }
                    }
                    mbarrier_expect_tx(plan.bar_ready[chunk_slot], transaction_bytes);
#pragma unroll
                    for (int n = 0; n < N_CHUNK; n++) {
                        if (n >= an)
                            continue;
                        int           slot = slot_of(gci, n);
                        const bf16_t* src =
                            residual_addr(block_res, layer_res, ns + n, N, t, block_stride_m, block_stride_r, H);
                        if (!has_delta && ns + n == N - 1 && !pdl_waited) {
                            cudaGridDependencySynchronize();
                            pdl_waited = true;
                        }
                        cp_async_bulk(buf_ptr(slot), src, H * sizeof(bf16_t), plan.bar_ready[chunk_slot]);
                    }
                    if (has_delta) {
                        int prefix_n = N - 1 - ns;
                        if (prefix_n >= 0 && prefix_n < an) {
                            if (!pdl_waited) {
                                cudaGridDependencySynchronize();
                                pdl_waited = true;
                            }
                            cp_async_bulk(delta_buf_ptr(chunk_slot),
                                          delta + (long long)tb * H,
                                          H * sizeof(bf16_t),
                                          plan.bar_ready[chunk_slot]);
                        }
                    }
                }
            }
            cudaTriggerProgrammaticLaunchCompletion();
        }
    } else {
        float acc32[ACC_PER_THREAD] = {};
        float eps_cache;
        asm volatile("mov.b32 %0, %1;" : "=f"(eps_cache) : "f"(rms_eps));

        long long gci = 0;
        for (int tb = blockIdx.x; tb < TB; tb += num_ctas) {
            const bf16_t* shared_row = shared_delta ? shared_delta + static_cast<int64_t>(tb) * H : nullptr;
            float         m_running  = -FLT_MAX;
            float         s_running  = 0.f;
#pragma unroll
            for (int i = 0; i < ACC_PER_THREAD; i++) {
                acc32[i] = 0.f;
            }

#pragma unroll
            for (int ci = 0; ci < num_chunks; ci++, gci++) {
                int ns         = ci * N_CHUNK;
                int an         = min(N_CHUNK, N - ns);
                int chunk_slot = (int)(gci % CHUNK_DEPTH);
                int pr         = phase_of(gci);
                mbarrier_wait(plan.bar_ready[chunk_slot], pr);

                float2 sq_local[N_CHUNK]  = {};
                float2 dot_local[N_CHUNK] = {};

                auto pass_A_body = [&](auto AN_TOK) {
                    constexpr int AN = decltype(AN_TOK)::value;
#pragma unroll
                    for (int si = 0; si < SLICES_PER_GROUP; si++) {
                        if constexpr (H == 7168) {
                            if (si == SLICES_PER_GROUP - 1) {
                                int          h_base = 6 * K_TILE + group * (K_TILE / 2) + ct_in_group * 4;
                                const float* qv     = &q_cache[si * VEC];
#pragma unroll
                                for (int n = 0; n < AN; n++) {
                                    int   slot     = slot_of(gci, n);
                                    int2  vp       = *reinterpret_cast<const int2*>(buf_ptr(slot) + h_base);
                                    auto* v2       = reinterpret_cast<__nv_bfloat162*>(&vp);
                                    int   prefix_n = N - 1 - ns;
                                    if (n == prefix_n && (has_delta || write_idx >= 0)) {
                                        if (has_delta) {
                                            const bf16_t* delta_ptr = delta_buf_ptr(chunk_slot) + h_base;
#pragma unroll
                                            for (int j = 0; j < 2; j++) {
                                                auto delta2 =
                                                    *reinterpret_cast<const __nv_bfloat162*>(delta_ptr + 2 * j);
                                                if (shared_row) {
                                                    delta2 = __hadd2(delta2,
                                                                     *reinterpret_cast<const __nv_bfloat162*>(
                                                                         shared_row + h_base + 2 * j));
                                                }
                                                v2[j] = __hadd2(v2[j], delta2);
                                            }
                                            *reinterpret_cast<int2*>(layer_res + (long long)tb * H + h_base) = vp;
                                        }
                                        if (write_idx >= 0) {
                                            *reinterpret_cast<int2*>(block_res + (long long)(tb / B) * block_stride_m
                                                                     + write_idx * block_stride_r + h_base) = vp;
                                        }
                                    }
                                    float2 f[2] = {__bfloat1622float2(v2[0]), __bfloat1622float2(v2[1])};
                                    tmem_store<4>(my_v_tmem + (si * N_CHUNK + n) * VEC, f);
                                    sq_local[n]  = float2_fma(f[0], f[0], sq_local[n]);
                                    sq_local[n]  = float2_fma(f[1], f[1], sq_local[n]);
                                    dot_local[n] = float2_fma(f[0], make_float2(qv[0], qv[1]), dot_local[n]);
                                    dot_local[n] = float2_fma(f[1], make_float2(qv[2], qv[3]), dot_local[n]);
                                }
                                continue;
                            }
                        }
                        int dt = si * CONSUMER_GROUPS + group;
                        if (dt >= NHT)
                            continue;
                        int          h_base = dt * K_TILE + k_local;
                        const float* qv     = &q_cache[si * VEC];

#pragma unroll
                        for (int n = 0; n < AN; n++) {
                            int   slot     = slot_of(gci, n);
                            int4  vp       = *reinterpret_cast<const int4*>(buf_ptr(slot) + h_base);
                            auto* v2       = reinterpret_cast<__nv_bfloat162*>(&vp);
                            int   prefix_n = N - 1 - ns;
                            if (n == prefix_n && (has_delta || write_idx >= 0)) {
                                if (has_delta) {
                                    const bf16_t* delta_ptr = delta_buf_ptr(chunk_slot) + h_base;
#pragma unroll
                                    for (int j = 0; j < VEC / 2; j++) {
                                        auto delta2 = *reinterpret_cast<const __nv_bfloat162*>(delta_ptr + 2 * j);
                                        if (shared_row) {
                                            delta2 = __hadd2(
                                                delta2,
                                                *reinterpret_cast<const __nv_bfloat162*>(shared_row + h_base + 2 * j));
                                        }
                                        v2[j] = __hadd2(v2[j], delta2);
                                    }
                                    *reinterpret_cast<int4*>(layer_res + (long long)tb * H + h_base) = vp;
                                }
                                if (write_idx >= 0) {
                                    *reinterpret_cast<int4*>(block_res + (long long)(tb / B) * block_stride_m
                                                             + write_idx * block_stride_r + h_base) = vp;
                                }
                            }
                            float2 f[4] = {__bfloat1622float2(v2[0]),
                                           __bfloat1622float2(v2[1]),
                                           __bfloat1622float2(v2[2]),
                                           __bfloat1622float2(v2[3])};
                            tmem_store<VEC>(my_v_tmem + (si * N_CHUNK + n) * VEC, f);
#pragma unroll
                            for (int j = 0; j < VEC / 2; j++) {
                                sq_local[n]  = float2_fma(f[j], f[j], sq_local[n]);
                                dot_local[n] = float2_fma(f[j], make_float2(qv[2 * j], qv[2 * j + 1]), dot_local[n]);
                            }
                        }
                    }
                };
                if constexpr (NC == 4) {
                    switch (an) {
                        case 4:
                            pass_A_body(std::integral_constant<int, 4>{});
                            break;
                        case 3:
                            pass_A_body(std::integral_constant<int, 3>{});
                            break;
                        case 2:
                            pass_A_body(std::integral_constant<int, 2>{});
                            break;
                        case 1:
                            pass_A_body(std::integral_constant<int, 1>{});
                            break;
                        default:
                            __builtin_unreachable();
                    }
                } else if constexpr (NC == 3) {
                    switch (an) {
                        case 3:
                            pass_A_body(std::integral_constant<int, 3>{});
                            break;
                        case 2:
                            pass_A_body(std::integral_constant<int, 2>{});
                            break;
                        case 1:
                            pass_A_body(std::integral_constant<int, 1>{});
                            break;
                        default:
                            __builtin_unreachable();
                    }
                } else {
                    static_assert(NC == 2);
                    switch (an) {
                        case 2:
                            pass_A_body(std::integral_constant<int, 2>{});
                            break;
                        case 1:
                            pass_A_body(std::integral_constant<int, 1>{});
                            break;
                        default:
                            __builtin_unreachable();
                    }
                }
                if (lane == 0) {
                    mbarrier_arrive(plan.bar_consumed[chunk_slot]);
                }
                tmem_store_wait();

                float2 reduce_pair[N_CHUNK];
#pragma unroll
                for (int n = 0; n < N_CHUNK; n++) {
                    reduce_pair[n] = make_float2(sq_local[n].x + sq_local[n].y, dot_local[n].x + dot_local[n].y);
                }
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
                    for (int n = 0; n < N_CHUNK; n++) {
                        uint64_t packed = reinterpret_cast<uint64_t&>(reduce_pair[n]);
                        packed          = __shfl_xor_sync(0xffffffff, packed, offset);
                        float2 other    = reinterpret_cast<float2&>(packed);
                        reduce_pair[n]  = float2_add(reduce_pair[n], other);
                    }
                }
                if (lane == 0) {
#pragma unroll
                    for (int n = 0; n < N_CHUNK; n++) {
                        plan.ws_stats[comp_wid][n] = reduce_pair[n];
                    }
                }
                named_barrier_sync(CONSUMER_THREADS, 0);

                float  local_rsig  = 0.f;
                float  local_logit = 0.f;
                int    stat_n      = lane / CONSUMER_WARPS;
                int    stat_w      = lane % CONSUMER_WARPS;
                float2 totals      = {};
                if (stat_n < N_CHUNK) {
                    totals = plan.ws_stats[stat_w][stat_n];
                }
                // Every consumer must finish reading before the next chunk/token
                // (or the output norm below) reuses the shared statistics.
                named_barrier_sync(CONSUMER_THREADS, 0);
#pragma unroll
                for (int offset = CONSUMER_WARPS / 2; offset > 0; offset >>= 1) {
                    totals.x += __shfl_down_sync(0xffffffff, totals.x, offset, CONSUMER_WARPS);
                    totals.y += __shfl_down_sync(0xffffffff, totals.y, offset, CONSUMER_WARPS);
                }
                if (stat_n < N_CHUNK && stat_w == 0) {
                    local_rsig  = rsqrtf(totals.x / H + eps_cache);
                    local_logit = totals.y * local_rsig;
                }
                float logit_n[N_CHUNK];
#pragma unroll
                for (int n = 0; n < N_CHUNK; n++) {
                    logit_n[n] = __shfl_sync(0xffffffff, local_logit, n * CONSUMER_WARPS);
                }

                float m_chunk = -FLT_MAX;
#pragma unroll
                for (int n = 0; n < N_CHUNK; n++) {
                    if (n < an)
                        m_chunk = fmaxf(m_chunk, logit_n[n]);
                }
                float m_new        = fmaxf(m_running, m_chunk);
                float corr         = exp2f((m_running - m_new) * LOG2_E);
                float w_n[N_CHUNK] = {};
                float w_sum        = 0.f;
#pragma unroll
                for (int n = 0; n < N_CHUNK; n++) {
                    if (n < an) {
                        w_n[n] = exp2f((logit_n[n] - m_new) * LOG2_E);
                        w_sum += w_n[n];
                    }
                }

                auto pass_B_body = [&](auto AN_TOK) {
                    constexpr int AN = decltype(AN_TOK)::value;
#pragma unroll
                    for (int si = 0; si < SLICES_PER_GROUP; si++) {
                        if constexpr (H == 7168) {
                            if (si == SLICES_PER_GROUP - 1) {
                                float2 corr2 = make_float2(corr, corr);
                                float2 a[2];
#pragma unroll
                                for (int j = 0; j < 2; j++) {
                                    float2 old = make_float2(acc32[si * VEC + 2 * j], acc32[si * VEC + 2 * j + 1]);
                                    a[j]       = float2_mul(old, corr2);
                                }
                                float2 f_cache[AN][2];
#pragma unroll
                                for (int n = 0; n < AN; n++) {
                                    tmem_load<4>(my_v_tmem + (si * N_CHUNK + n) * VEC, f_cache[n]);
                                }
#pragma unroll
                                for (int n = 0; n < AN; n++) {
                                    float2 wn = make_float2(w_n[n], w_n[n]);
#pragma unroll
                                    for (int j = 0; j < 2; j++) {
                                        a[j] = float2_fma(wn, f_cache[n][j], a[j]);
                                    }
                                }
#pragma unroll
                                for (int j = 0; j < 2; j++) {
                                    acc32[si * VEC + 2 * j]     = a[j].x;
                                    acc32[si * VEC + 2 * j + 1] = a[j].y;
                                }
                                continue;
                            }
                        }
                        int dt = si * CONSUMER_GROUPS + group;
                        if (dt >= NHT)
                            continue;
                        float2 corr2 = make_float2(corr, corr);
                        float2 a[VEC / 2];
#pragma unroll
                        for (int j = 0; j < VEC / 2; j++) {
                            float2 old = make_float2(acc32[si * VEC + 2 * j], acc32[si * VEC + 2 * j + 1]);
                            a[j]       = float2_mul(old, corr2);
                        }
                        float2 f_cache[AN][VEC / 2];
#pragma unroll
                        for (int n = 0; n < AN; n++) {
                            tmem_load<VEC>(my_v_tmem + (si * N_CHUNK + n) * VEC, f_cache[n]);
                        }
#pragma unroll
                        for (int n = 0; n < AN; n++) {
                            float2 wn = make_float2(w_n[n], w_n[n]);
#pragma unroll
                            for (int j = 0; j < VEC / 2; j++) {
                                a[j] = float2_fma(wn, f_cache[n][j], a[j]);
                            }
                        }
#pragma unroll
                        for (int j = 0; j < VEC / 2; j++) {
                            acc32[si * VEC + 2 * j]     = a[j].x;
                            acc32[si * VEC + 2 * j + 1] = a[j].y;
                        }
                    }
                };
                if constexpr (NC == 4) {
                    switch (an) {
                        case 4:
                            pass_B_body(std::integral_constant<int, 4>{});
                            break;
                        case 3:
                            pass_B_body(std::integral_constant<int, 3>{});
                            break;
                        case 2:
                            pass_B_body(std::integral_constant<int, 2>{});
                            break;
                        case 1:
                            pass_B_body(std::integral_constant<int, 1>{});
                            break;
                        default:
                            __builtin_unreachable();
                    }
                } else if constexpr (NC == 3) {
                    switch (an) {
                        case 3:
                            pass_B_body(std::integral_constant<int, 3>{});
                            break;
                        case 2:
                            pass_B_body(std::integral_constant<int, 2>{});
                            break;
                        case 1:
                            pass_B_body(std::integral_constant<int, 1>{});
                            break;
                        default:
                            __builtin_unreachable();
                    }
                } else {
                    static_assert(NC == 2);
                    switch (an) {
                        case 2:
                            pass_B_body(std::integral_constant<int, 2>{});
                            break;
                        case 1:
                            pass_B_body(std::integral_constant<int, 1>{});
                            break;
                        default:
                            __builtin_unreachable();
                    }
                }

                s_running = s_running * corr + w_sum;
                m_running = m_new;
            }

            float          inv_s          = 1.f / s_running;
            constexpr bool STORE_BF16     = !STORE_NVFP4_ROWS || STORE_FP8;
            bf16_t*        out_ptr        = STORE_BF16 ? output + (long long)tb * H : nullptr;
            float2         output_sq_pair = {};
            // When output RMSNorm is fused, the softmax denominator cancels:
            // (acc / s) * rsqrt(mean((acc / s)^2) + eps)
            //   = acc * rsqrt(mean(acc^2) + eps * s^2).
#pragma unroll
            for (int si = 0; si < SLICES_PER_GROUP; si++) {
                if constexpr (H == 7168) {
                    if (si == SLICES_PER_GROUP - 1) {
                        int    h_base = 6 * K_TILE + group * (K_TILE / 2) + ct_in_group * 4;
                        uint2  packed;
                        auto*  ov2  = reinterpret_cast<__nv_bfloat162*>(&packed);
                        float2 inv2 = make_float2(inv_s, inv_s);
#pragma unroll
                        for (int j = 0; j < 2; j++) {
                            float2 old = make_float2(acc32[si * VEC + 2 * j], acc32[si * VEC + 2 * j + 1]);
                            if constexpr (HAS_OUTPUT_NORM) {
                                output_sq_pair = float2_fma(old, old, output_sq_pair);
                            } else {
                                float2 mixed = float2_mul(old, inv2);
                                ov2[j]       = __float22bfloat162_rn(mixed);
                            }
                        }
                        if constexpr (!HAS_OUTPUT_NORM) {
                            *reinterpret_cast<uint2*>(out_ptr + h_base) = packed;
                        }
                        continue;
                    }
                }
                int dt = si * CONSUMER_GROUPS + group;
                if (dt >= NHT)
                    continue;
                int    h_base = dt * K_TILE + k_local;
                uint4  packed;
                auto*  ov2  = reinterpret_cast<__nv_bfloat162*>(&packed);
                float2 inv2 = make_float2(inv_s, inv_s);
#pragma unroll
                for (int j = 0; j < VEC / 2; j++) {
                    float2 old = make_float2(acc32[si * VEC + 2 * j], acc32[si * VEC + 2 * j + 1]);
                    if constexpr (HAS_OUTPUT_NORM) {
                        output_sq_pair = float2_fma(old, old, output_sq_pair);
                    } else {
                        float2 mixed = float2_mul(old, inv2);
                        ov2[j]       = __float22bfloat162_rn(mixed);
                    }
                }
                if constexpr (!HAS_OUTPUT_NORM) {
                    *reinterpret_cast<uint4*>(out_ptr + h_base) = packed;
                }
            }

            if constexpr (HAS_OUTPUT_NORM) {
                // The immutable weight copy is acquired once, at its first use.
                if (tb == blockIdx.x) {
                    mbarrier_wait(plan.bar_output_norm_ready, 0);
                }
                float output_sq = output_sq_pair.x + output_sq_pair.y;
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    output_sq += __shfl_xor_sync(0xffffffff, output_sq, offset);
                }
                if (lane == 0) {
                    plan.ws_stats[comp_wid][0] = make_float2(output_sq, 0.f);
                }
                named_barrier_sync(CONSUMER_THREADS, 0);
                float total_sq = lane < CONSUMER_WARPS ? plan.ws_stats[lane][0].x : 0.f;
                named_barrier_sync(CONSUMER_THREADS, 0);
#pragma unroll
                for (int offset = CONSUMER_WARPS / 2; offset > 0; offset >>= 1) {
                    total_sq += __shfl_down_sync(0xffffffff, total_sq, offset, CONSUMER_WARPS);
                }
                if (lane == 0) {
                    total_sq = rsqrtf(total_sq / H + output_norm_eps * s_running * s_running);
                }
                float    output_rsigma = __shfl_sync(0xffffffff, total_sq, 0);
                uint4    row_values[SLICES_PER_GROUP];
                uint32_t row_amax = 0;
#pragma unroll
                for (int si = 0; si < SLICES_PER_GROUP; si++) {
                    if constexpr (H == 7168) {
                        if (si == SLICES_PER_GROUP - 1) {
                            int   h_base = 6 * K_TILE + group * (K_TILE / 2) + ct_in_group * 4;
                            uint2 packed;
                            auto* values = reinterpret_cast<bf16_t*>(&packed);
#pragma unroll
                            for (int j = 0; j < 4; j++) {
                                float weight = __bfloat162float(output_norm_buf[h_base + j]);
                                values[j]    = __float2bfloat16(acc32[si * VEC + j] * output_rsigma * weight);
                            }
                            if constexpr (STORE_NVFP4_ROWS) {
                                row_values[si] = make_uint4(packed.x, packed.y, 0, 0);
                                row_amax       = __vmaxu2(row_amax, packed.x & 0x7fff7fff);
                                row_amax       = __vmaxu2(row_amax, packed.y & 0x7fff7fff);
                            }
                            if constexpr (STORE_BF16) {
                                *reinterpret_cast<uint2*>(out_ptr + h_base) = packed;
                            }
                            if constexpr (STORE_NVFP4_AMAX) {
                                // Absolute BF16 values are ordered by their unsigned bits.
                                packed_amax = __vmaxu2(packed_amax, packed.x & 0x7fff7fff);
                                packed_amax = __vmaxu2(packed_amax, packed.y & 0x7fff7fff);
                            }
                            if constexpr (STORE_FP8) {
                                store_fp8_group<H, 4, uint32_t>(values, fp8_output, fp8_scales, tb, TB, h_base);
                            }
                            continue;
                        }
                    }
                    int dt = si * CONSUMER_GROUPS + group;
                    if (dt >= NHT)
                        continue;
                    int   h_base = dt * K_TILE + k_local;
                    uint4 packed;
                    auto* values = reinterpret_cast<bf16_t*>(&packed);
#pragma unroll
                    for (int j = 0; j < VEC; j++) {
                        float weight = __bfloat162float(output_norm_buf[h_base + j]);
                        values[j]    = __float2bfloat16(acc32[si * VEC + j] * output_rsigma * weight);
                    }
                    if constexpr (STORE_NVFP4_ROWS) {
                        row_values[si] = packed;
                        row_amax       = __vmaxu2(row_amax, packed.x & 0x7fff7fff);
                        row_amax       = __vmaxu2(row_amax, packed.y & 0x7fff7fff);
                        row_amax       = __vmaxu2(row_amax, packed.z & 0x7fff7fff);
                        row_amax       = __vmaxu2(row_amax, packed.w & 0x7fff7fff);
                    }
                    if constexpr (STORE_BF16) {
                        *reinterpret_cast<uint4*>(out_ptr + h_base) = packed;
                    }
                    if constexpr (STORE_NVFP4_AMAX) {
                        packed_amax = __vmaxu2(packed_amax, packed.x & 0x7fff7fff);
                        packed_amax = __vmaxu2(packed_amax, packed.y & 0x7fff7fff);
                        packed_amax = __vmaxu2(packed_amax, packed.z & 0x7fff7fff);
                        packed_amax = __vmaxu2(packed_amax, packed.w & 0x7fff7fff);
                    }
                    if constexpr (STORE_FP8) {
                        store_fp8_group<H, VEC, uint64_t>(values, fp8_output, fp8_scales, tb, TB, h_base);
                    }
                }
                if constexpr (STORE_NVFP4_ROWS) {
                    store_nvfp4_row<H>(row_values,
                                       row_amax,
                                       warp_amax,
                                       tb,
                                       group,
                                       comp_wid,
                                       ct_in_group,
                                       lane,
                                       reinterpret_cast<uint32_t*>(nvfp4_packet),
                                       row_block_scales,
                                       row_scales);
                }
            }
        }
    }

    if constexpr (STORE_NVFP4_AMAX) {
        if (wid != 0) {
            uint32_t maximum = __reduce_max_sync(0xffffffff, max(packed_amax & 0xffffu, packed_amax >> 16));
            if (lane == 0) {
                warp_amax[comp_wid] = maximum;
            }
        }
    }
    if constexpr (STORE_NVFP4_ROWS) {
        const int id = blockIdx.x * blockDim.x + tid;
        for (int i = id; i < (padded_rows - TB) * (H / 16); i += gridDim.x * blockDim.x) {
            *cvt_quant_to_fp4_get_sf_out_offset<uint8_t, 1>(TB + i / (H / 16), i % (H / 16), H, row_block_scales) = 0;
        }
        if (id < ((4 - TB % 4) % 4)) {
            row_scales[TB + id] = 0.f;
        }
    }
    __syncthreads();
    if constexpr (STORE_NVFP4_AMAX) {
        if (tid == 0) {
            uint32_t maximum = 0;
#pragma unroll
            for (int warp = 0; warp < CONSUMER_WARPS; ++warp) {
                maximum = max(maximum, warp_amax[warp]);
            }
            amax_partials[blockIdx.x] = __uint_as_float(maximum << 16);
        }
    }
    if (wid == 1) {
        tmem_free(plan.tmem_base, TMEM_COLS_ALLOC);
    }
    if constexpr (STORE_NVFP4_AMAX) {
        if (nvfp4_packet != nullptr) {
            attn_res_nvfp4_packet<H>(output, TB, amax_partials, nvfp4_packet);
        }
    }
#else
    if (threadIdx.x == 0) {
        printf("attn_res_fwd_online_v2_kernel requires sm_10x\n");
    }
#endif
}

template<int  H,
         int  N,
         int  NC                 = N_CHUNK_DEFAULT,
         bool RELEASE_TMEM       = false,
         bool STATIC_COMMON_PATH = false,
         bool HAS_OUTPUT_NORM    = false,
         bool STORE_FP8          = false,
         bool STORE_NVFP4_AMAX   = false,
         bool STORE_NVFP4_ROWS   = false>
static void launch_fwd(bf16_t*       block_residual,
                       bf16_t*       layer_residual,
                       const bf16_t* delta,
                       const bf16_t* res_weight,
                       const bf16_t* rms_weight,
                       bf16_t*       output,
                       int           T,
                       float         rms_eps,
                       int           num_sm,
                       cudaStream_t  stream,
                       const bf16_t* output_norm_weight = nullptr,
                       float         output_norm_eps    = 0.f,
                       int           block_stride_m     = 0,
                       int           block_stride_r     = 0,
                       int           block_write_idx    = -1,
                       uint8_t*      fp8_output         = nullptr,
                       uint8_t*      fp8_scales         = nullptr,
                       float*        amax_partials      = nullptr,
                       uint8_t*      nvfp4_packet       = nullptr,
                       const bf16_t* shared_delta       = nullptr) {
    constexpr size_t smem_size = ((size_t)(CHUNK_DEPTH * (NC + 1) + (HAS_OUTPUT_NORM ? 1 : 0)) * H * sizeof(bf16_t)
                                  + sizeof(FwdSmemPlan<NC>) + 15)
                                 & ~size_t(15);
    auto        kernel    = &attn_res_fwd_online_v2_kernel<H,
                                                           N,
                                                           NC,
                                                           1,
                                                           RELEASE_TMEM,
                                                           STATIC_COMMON_PATH,
                                                           HAS_OUTPUT_NORM,
                                                           STORE_FP8,
                                                           STORE_NVFP4_AMAX,
                                                           STORE_NVFP4_ROWS>;
    static bool attrs_set = false;
    if (!attrs_set) {
        if (smem_size > 48 * 1024) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        }
        attrs_set = true;
    }
    int                grid = RELEASE_TMEM ? num_sm * 2 : num_sm;
    cudaLaunchConfig_t config{};
    config.gridDim          = grid;
    config.blockDim         = BLK;
    config.dynamicSmemBytes = smem_size;
    config.stream           = stream;
    cudaLaunchAttribute attr{};
    if (nvfp4_packet != nullptr && !STORE_NVFP4_ROWS) {
        attr.id              = cudaLaunchAttributeCooperative;
        attr.val.cooperative = 1;
    } else {
        attr.id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;
    }
    config.attrs    = &attr;
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config,
                                      kernel,
                                      block_residual,
                                      layer_residual,
                                      delta,
                                      res_weight,
                                      rms_weight,
                                      output,
                                      T,
                                      block_stride_m,
                                      block_stride_r,
                                      rms_eps,
                                      output_norm_weight,
                                      output_norm_eps,
                                      block_write_idx,
                                      fp8_output,
                                      fp8_scales,
                                      amax_partials,
                                      nvfp4_packet,
                                      shared_delta));
}

}  // namespace fwd_prod_v2
}  // namespace sm100

void attn_res(at::Tensor&               prefix,
              std::optional<at::Tensor> delta,
              at::Tensor&               blocks,
              at::Tensor const&         norm_weight,
              at::Tensor const&         qk_weight,
              std::optional<at::Tensor> output_norm_weight,
              at::Tensor&               output,
              int64_t                   num_blocks,
              int64_t                   block_write_idx,
              double                    eps,
              double                    output_norm_eps,
              std::optional<at::Tensor> shared_delta,
              std::optional<at::Tensor> fp8_output,
              std::optional<at::Tensor> fp8_scales,
              std::optional<at::Tensor> amax_partials,
              std::optional<at::Tensor> nvfp4_packet,
              bool                      nvfp4_rowwise) {
    TORCH_CHECK(prefix.is_cuda() && prefix.dim() == 2 && prefix.scalar_type() == at::kBFloat16,
                "AttnRes prefix must be a CUDA BF16 matrix");
    auto check_tensor = [&](const at::Tensor& tensor, const char* name) {
        TORCH_CHECK(tensor.device() == prefix.device() && tensor.scalar_type() == at::kBFloat16,
                    "AttnRes ",
                    name,
                    " must match prefix device and BF16 dtype");
        TORCH_CHECK(tensor.stride(-1) == 1, "AttnRes requires contiguous inner dimensions");
    };
    check_tensor(prefix, "prefix");
    check_tensor(blocks, "blocks");
    check_tensor(output, "output");
    check_tensor(norm_weight, "norm_weight");
    check_tensor(qk_weight, "qk_weight");
    TORCH_CHECK(blocks.dim() == 3 && blocks.size(0) == prefix.size(0) && blocks.size(2) == prefix.size(1),
                "AttnRes blocks shape mismatch");
    TORCH_CHECK(output.sizes() == prefix.sizes(), "AttnRes output shape mismatch");
    TORCH_CHECK(norm_weight.is_contiguous() && qk_weight.is_contiguous() && norm_weight.numel() == prefix.size(1)
                    && qk_weight.numel() == prefix.size(1),
                "AttnRes weight shape mismatch");
    if (delta.has_value()) {
        check_tensor(*delta, "delta");
        TORCH_CHECK(delta->sizes() == prefix.sizes(), "AttnRes delta shape mismatch");
    }
    if (shared_delta.has_value()) {
        check_tensor(*shared_delta, "shared delta");
        TORCH_CHECK(delta.has_value() && shared_delta->sizes() == prefix.sizes() && shared_delta->is_contiguous(),
                    "AttnRes shared delta requires matching dense MoE deltas");
    }
    if (output_norm_weight.has_value()) {
        check_tensor(*output_norm_weight, "output_norm_weight");
        TORCH_CHECK(output_norm_weight->is_contiguous() && output_norm_weight->numel() == prefix.size(1),
                    "AttnRes output norm shape mismatch");
    }
    int const num_tokens = static_cast<int>(prefix.size(0));
    TORCH_CHECK(fp8_output.has_value() == fp8_scales.has_value(), "AttnRes requires both FP8 outputs");
    if (fp8_output.has_value()) {
        TORCH_CHECK(output_norm_weight.has_value() && fp8_output->device() == prefix.device()
                        && fp8_output->scalar_type() == at::kFloat8_e4m3fn && fp8_output->sizes() == prefix.sizes()
                        && fp8_output->is_contiguous(),
                    "AttnRes FP8 output must be contiguous E4M3 with output norm");
        TORCH_CHECK(fp8_scales->device() == prefix.device() && fp8_scales->scalar_type() == at::kInt
                        && fp8_scales->is_contiguous() && fp8_scales->dim() == 2
                        && fp8_scales->size(0) == (prefix.size(1) + 511) / 512
                        && fp8_scales->size(1) == ((num_tokens + 3) & ~3),
                    "AttnRes FP8 scales must be packed group-128 UE8M0");
    }
    int const                  device = prefix.get_device();
    c10::cuda::CUDAGuard const device_guard(prefix.device());
    cudaDeviceProp const*      properties = at::cuda::getDeviceProperties(device);
    if (amax_partials.has_value()) {
        TORCH_CHECK(output_norm_weight.has_value() && amax_partials->device() == prefix.device()
                        && amax_partials->scalar_type() == at::kFloat && amax_partials->is_contiguous()
                        && amax_partials->dim() == 1 && amax_partials->numel() == properties->multiProcessorCount,
                    "AttnRes NVFP4 amax requires output norm and one FP32 value per SM");
    }
    TORCH_CHECK(properties->major == 10, "AttnRes requires the SM100 family");
    int64_t const hidden_size = prefix.size(1);
    TORCH_CHECK(hidden_size == 7168, "AttnRes requires hidden_size=7168, got ", hidden_size);
    TORCH_CHECK(prefix.stride(0) == hidden_size && (!delta.has_value() || delta->stride(0) == hidden_size)
                    && output.stride(0) == hidden_size,
                "AttnRes requires densely packed rows; got strides ",
                prefix.stride(0),
                ", ",
                delta.has_value() ? delta->stride(0) : 0,
                ", ",
                output.stride(0),
                " for hidden_size ",
                hidden_size);
    TORCH_CHECK(num_blocks >= 0 && num_blocks <= 8 && num_blocks <= blocks.size(1),
                "AttnRes: num_blocks must be 0..min(8, blocks.size(1))");
    TORCH_CHECK(block_write_idx >= -1 && block_write_idx < blocks.size(1),
                "AttnRes: block_write_idx must be -1 or a valid "
                "block row, got ",
                block_write_idx);

    if (nvfp4_rowwise) {
        TORCH_CHECK(nvfp4_packet && output_norm_weight && !amax_partials,
                    "Rowwise AttnRes NVFP4 requires output norm and a packet");
    }
    if (nvfp4_packet.has_value()) {
        const int64_t padded       = (static_cast<int64_t>(num_tokens) + 127) / 128 * 128;
        const int64_t scale_bytes  = nvfp4_rowwise ? ((static_cast<int64_t>(num_tokens) + 3) / 4 * 16) : 16;
        const int64_t packet_bytes = num_tokens * (hidden_size / 2) + padded * (hidden_size / 16) + scale_bytes;
        TORCH_CHECK((nvfp4_rowwise || (amax_partials && properties->cooperativeLaunch))
                        && nvfp4_packet->device() == prefix.device() && nvfp4_packet->scalar_type() == at::kByte
                        && nvfp4_packet->is_contiguous() && nvfp4_packet->numel() == packet_bytes
                        && reinterpret_cast<uintptr_t>(nvfp4_packet->data_ptr()) % 16 == 0,
                    "AttnRes NVFP4 requires an aligned CUTLASS packet and supported launch");
    }

    if (num_tokens == 0) {
        return;
    }

    using namespace sm100::fwd_prod_v2;
    cudaStream_t stream          = at::cuda::getCurrentCUDAStream(device);
    auto         dispatch_blocks = [&](auto static_common_path,
                               auto has_output_norm,
                               auto store_fp8,
                               auto store_nvfp4_amax,
                               auto store_nvfp4_rows) {
        constexpr bool STATIC_COMMON_PATH = decltype(static_common_path)::value;
        constexpr bool HAS_OUTPUT_NORM    = decltype(has_output_norm)::value;
        auto           dispatch           = [&](auto source_count) {
            constexpr int NUM_SOURCES = decltype(source_count)::value;
            launch_fwd<7168,
                                                   NUM_SOURCES,
                                                   3,
                                                   false,
                                                   STATIC_COMMON_PATH,
                                                   HAS_OUTPUT_NORM,
                                                   decltype(store_fp8)::value,
                                                   decltype(store_nvfp4_amax)::value,
                                                   decltype(store_nvfp4_rows)::value>(
                static_cast<bf16_t*>(blocks.data_ptr()),
                static_cast<bf16_t*>(prefix.data_ptr()),
                delta.has_value() ? static_cast<bf16_t const*>(delta->data_ptr()) : nullptr,
                static_cast<bf16_t const*>(qk_weight.data_ptr()),
                static_cast<bf16_t const*>(norm_weight.data_ptr()),
                nvfp4_rowwise && !fp8_output ? nullptr : static_cast<bf16_t*>(output.data_ptr()),
                num_tokens,
                static_cast<float>(eps),
                properties->multiProcessorCount,
                stream,
                output_norm_weight.has_value() ? static_cast<bf16_t const*>(output_norm_weight->data_ptr()) : nullptr,
                static_cast<float>(output_norm_eps),
                static_cast<int>(blocks.stride(0)),
                static_cast<int>(blocks.stride(1)),
                static_cast<int>(block_write_idx),
                fp8_output ? static_cast<uint8_t*>(fp8_output->data_ptr()) : nullptr,
                fp8_scales ? static_cast<uint8_t*>(fp8_scales->data_ptr()) : nullptr,
                amax_partials ? amax_partials->data_ptr<float>() : nullptr,
                nvfp4_packet ? nvfp4_packet->data_ptr<uint8_t>() : nullptr,
                shared_delta ? static_cast<const bf16_t*>(shared_delta->data_ptr()) : nullptr);
        };
        switch (num_blocks) {
            case 0:
                dispatch(std::integral_constant<int, 1>{});
                break;
            case 1:
                dispatch(std::integral_constant<int, 2>{});
                break;
            case 2:
                dispatch(std::integral_constant<int, 3>{});
                break;
            case 3:
                dispatch(std::integral_constant<int, 4>{});
                break;
            case 4:
                dispatch(std::integral_constant<int, 5>{});
                break;
            case 5:
                dispatch(std::integral_constant<int, 6>{});
                break;
            case 6:
                dispatch(std::integral_constant<int, 7>{});
                break;
            case 7:
                dispatch(std::integral_constant<int, 8>{});
                break;
            case 8:
                dispatch(std::integral_constant<int, 9>{});
                break;
        }
    };
    auto dispatch_quantized = [&](auto store_fp8, auto store_nvfp4_amax) {
        if (delta.has_value() && block_write_idx < 0) {
            dispatch_blocks(std::true_type{}, std::true_type{}, store_fp8, store_nvfp4_amax, std::false_type{});
        } else {
            dispatch_blocks(std::false_type{}, std::true_type{}, store_fp8, store_nvfp4_amax, std::false_type{});
        }
    };
    if (nvfp4_rowwise) {
        auto dispatch_rowwise = [&](auto store_fp8) {
            if (delta.has_value() && block_write_idx < 0) {
                dispatch_blocks(std::true_type{}, std::true_type{}, store_fp8, std::false_type{}, std::true_type{});
            } else {
                dispatch_blocks(std::false_type{}, std::true_type{}, store_fp8, std::false_type{}, std::true_type{});
            }
        };
        if (fp8_output.has_value()) {
            dispatch_rowwise(std::true_type{});
        } else {
            dispatch_rowwise(std::false_type{});
        }
    } else if (amax_partials.has_value()) {
        if (fp8_output.has_value()) {
            dispatch_quantized(std::true_type{}, std::true_type{});
        } else {
            dispatch_quantized(std::false_type{}, std::true_type{});
        }
    } else if (fp8_output.has_value()) {
        dispatch_quantized(std::true_type{}, std::false_type{});
    } else if (delta.has_value() && output_norm_weight.has_value() && block_write_idx < 0) {
        dispatch_blocks(std::true_type{}, std::true_type{}, std::false_type{}, std::false_type{}, std::false_type{});
    } else if (output_norm_weight.has_value()) {
        dispatch_blocks(std::false_type{}, std::true_type{}, std::false_type{}, std::false_type{}, std::false_type{});
    } else {
        dispatch_blocks(std::false_type{}, std::false_type{}, std::false_type{}, std::false_type{}, std::false_type{});
    }
    cudaError_t const error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "AttnRes kernel launch failed: ", cudaGetErrorString(error));
}
