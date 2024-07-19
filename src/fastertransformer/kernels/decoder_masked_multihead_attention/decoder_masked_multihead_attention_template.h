/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "src/fastertransformer/kernels/decoder_masked_multihead_attention.h"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/kernels/kv_cache_utils.h"
#include "src/fastertransformer/kernels/rotary_position_embedding.h"

#include "src/fastertransformer/cuda/cuda_type_utils.cuh"
#if USING_CUDA
#include "src/fastertransformer/cuda/memory_utils.h"
// Multi-block mmha kernel can only be selected when CUDA >= 11.7
#if (CUDART_VERSION >= 11070)
#define ENABLE_MULTI_BLOCK_OPTION
#endif

#ifdef ENABLE_MULTI_BLOCK_OPTION
#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>
#include <cuda/std/bit>
#endif  // ENABLE_MULTI_BLOCK_OPTION
#endif

#if USING_ROCM
#include "src/fastertransformer/rocm/hip_utils.h"
#endif

#include <assert.h>
#include <float.h>
#include <type_traits>


namespace fastertransformer {

template<typename T>
inline __device__ float convert_to_float(const T* value) {
    printf("convert type: %s", typeid(T).name());
    return (float)(value[0]);
}

template<>
inline __device__ float convert_to_float(const uint16_t* value) {
    return __half2float(((half*)value)[0]);
}

template<>
inline __device__ float convert_to_float(const float* value) {
    return value[0];
}

template<>
inline __device__ float convert_to_float(const half* value) {
    return __half2float(value[0]);
}

template<>
inline __device__ float convert_to_float(const __nv_bfloat16* value) {
    return (float)(value[0]);
}

template<typename T>
inline __device__ void print_floats(const char* hint, const T* value, const size_t hint_idx = 0, const size_t num = 1) {
    for (size_t i = 0; i < num; i++) {
        const auto ptr = value + i;
        printf("%s[%d](%p): %f\n", hint, hint_idx + i, ptr, convert_to_float(ptr));
    }
}

template<typename T>
__inline__ __host__ __device__ T constexpr flat_index_strided3(
    T const& index_0, T const& index_1, T const& index_2, T const& stride_1, T const& stride_2)
{
    assert(index_1 < stride_1 / stride_2);
    assert(index_2 < stride_2);
    return index_0 * stride_1 + index_1 * stride_2 + index_2;
}

template<typename T>
__inline__ __host__ __device__ T constexpr flat_index2(T const& index_0, T const& index_1, T const& dim_1)
{
    assert(index_1 < dim_1);
    return index_0 * dim_1 + index_1;
}

// Use HMMA to compute with FP16/BF16 inputs and FP32 accumulators.
// #define MMHA_USE_HMMA

// Pre-scale Q or P to reduce number of instructions for dequantizing KV cache.
// If you notice a decrease in accuracy when the fp8 kv cache is enabled,
//  consider disabling the two flags.
#ifdef ENABLE_FP8
// Apply the FP8 scaling to Q instead of K.
#define MMHA_FP8_SCALE_Q_INSTEAD_OF_K
// Apply the FP8 scaling to P instead of V.
#define MMHA_FP8_SCALE_P_INSTEAD_OF_V
#endif  // !defined ENABLE_FP8

// Below are knobs to extend FP32 accumulation for higher FP16 accuracy

// Does not seem to affect the accuracy that much
#define MMHA_USE_FP32_ACCUM_FOR_FMA

// Seems to slightly improve the accuracy
#define MMHA_USE_FP32_ACCUM_FOR_OUT

// #define MMHA_USE_FP32_ACCUM_FOR_LOGITS

#if 0 && defined(MMHA_USE_FP32_ACCUM_FOR_OUT)
 // Does not seem to improve the accuracy
 //#define MMHA_USE_FP32_ACCUM_FOR_LOGITS
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// We use the following terminology to describe the different dimensions.
//
// B:  Batch size (number of sequences),
// L:  Sequence length,
// D:  Hidden dimension,
// H:  Number of heads,
// Dh: Hidden dimension per head - Dh = D / H.
//
// The different kernels assign a threadblock for B x H pair. The grid has size (1, B, H). We use
// 256 threads per block to maximum occupancy and performance.
//
// Each threadblock loads Dh values from Q and its associated bias. The kernels run a loop to
// compute Q * K^T where K is loaded from a cache buffer -- except for the current timestep. The
// cache buffer helps with memory accesses and contains keys with bias.
//
// The layout of the cache buffer for the keys/values is [B, H, L, Dh]
// where the fastest moving dimension (contiguous data) is the rightmost one.
// Contiguous threads will read one hidden_dimension per LDG unless we need more than 32 threads.
//
// The different kernels use 1 ~ 32 threads per key (THREADS_PER_KEY). The size of the LDGs
// is always 16bytes (8 bytes for 8bit cache). Each thread sums Dh / THREADS_PER_KEY elements. At
// the end of each iteration of the Q * K^T loop, we perform a reduction between lanes using an
// HMMA instruction (Tensor Core). Each Q * K^T value is stored in shared memory in FP32.
//
// After that loop, a parallel softmax is computed across the different Q * K^T values stored in
// shared memory.
//
// The kernel ends with a loop over the values in V. We use THREADS_PER_VALUE to control how many
// timesteps are computed by loop iteration. As with the keys, the values are read from a cache
// except for the current timestep. The layout of the cache buffer for the values is same as the key,
// which is [B, H, L, Dh].
//
// Note that we have remapped key layout to make sure it shares the same pattern as value [B, H, L, Dh].
// It helps coalescing memory access, and reducing register pressure.

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Dh_MAX>
struct Qk_vec_m_ {};

template<>
struct Qk_vec_m_<float, 32> {
    using Type = float;
};

template<>
struct Qk_vec_m_<float, 64> {
    using Type = float2;
};

template<>
struct Qk_vec_m_<float, 128> {
    using Type = float4;
};

template<>
struct Qk_vec_m_<float, 256> {
    using Type = float4;
};

template<>
struct Qk_vec_m_<uint16_t, 32> {
    using Type = uint32_t;
};

template<>
struct Qk_vec_m_<uint16_t, 64> {
    using Type = uint32_t;
};

template<>
struct Qk_vec_m_<uint16_t, 128> {
    using Type = uint2;
};

// NOTE: RoPE kernel impelmentation does not work correctly under vector_t=uint4
// here the vec_t is hakced to uint2 for correctness
template<>
struct Qk_vec_m_<uint16_t, 256> {
    using Type = uint2;
};
#ifdef ENABLE_BF16
template<>
struct Qk_vec_m_<__nv_bfloat16, 32> {
    using Type = __nv_bfloat162;
};

template<>
struct Qk_vec_m_<__nv_bfloat16, 64> {
    using Type = __nv_bfloat162;
};

template<>
struct Qk_vec_m_<__nv_bfloat16, 128> {
    using Type = bf16_4_t;
};

template<>
struct Qk_vec_m_<__nv_bfloat16, 256> {
    using Type = bf16_8_t;
};
#endif  // ENABLE_BF16

#ifdef ENABLE_FP8
template<>
struct Qk_vec_m_<__nv_fp8_e4m3, 32> {
    using Type = fp8_4_t;
};

template<>
struct Qk_vec_m_<__nv_fp8_e4m3, 64> {
    using Type = fp8_4_t;
};

template<>
struct Qk_vec_m_<__nv_fp8_e4m3, 128> {
    using Type = fp8_4_t;
};

template<>
struct Qk_vec_m_<__nv_fp8_e4m3, 256> {
    using Type = fp8_4_t;
};
#endif  // ENABLE_FP8

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Dh>
struct Qk_vec_k_ {
    using Type = typename Qk_vec_m_<T, Dh>::Type;
};
#ifdef ENABLE_FP8
template<>
struct Qk_vec_k_<__nv_fp8_e4m3, 32> {
    using Type = float4;
};

template<>
struct Qk_vec_k_<__nv_fp8_e4m3, 64> {
    using Type = float4;
};

template<>
struct Qk_vec_k_<__nv_fp8_e4m3, 128> {
    using Type = float4;
};

template<>
struct Qk_vec_k_<__nv_fp8_e4m3, 256> {
    using Type = float4;
};
#endif  // ENABLE_FP8

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int V_VEC_SIZE>
struct V_vec_m_ {};

template<>
struct V_vec_m_<float, 1> {
    using Type = float;
};

template<>
struct V_vec_m_<float, 2> {
    using Type = float2;
};

template<>
struct V_vec_m_<float, 4> {
    using Type = float4;
};

template<>
struct V_vec_m_<float, 8> {
    using Type = Float8_;
};

template<>
struct V_vec_m_<uint16_t, 2> {
    using Type = uint32_t;
};

template<>
struct V_vec_m_<uint16_t, 4> {
    using Type = uint2;
};

template<>
struct V_vec_m_<uint16_t, 8> {
    using Type = uint4;
};
#ifdef ENABLE_BF16
template<>
struct V_vec_m_<__nv_bfloat16, 2> {
    using Type = __nv_bfloat162;
};

template<>
struct V_vec_m_<__nv_bfloat16, 4> {
    using Type = bf16_4_t;
};

template<>
struct V_vec_m_<__nv_bfloat16, 8> {
    using Type = bf16_8_t;
};
#endif  // ENABLE_BF16

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int V_VEC_SIZE>
struct V_vec_k_ {
    using Type = typename V_vec_m_<T, V_VEC_SIZE>::Type;
};
#ifdef ENABLE_FP8
template<>
struct V_vec_k_<__nv_fp8_e4m3, 4> {
    using Type = float4;
};

template<>
struct V_vec_k_<__nv_fp8_e4m3, 8> {
    using Type = float4;
};

template<>
struct V_vec_k_<__nv_fp8_e4m3, 16> {
    using Type = float4;
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

// Reuse V_vec traits as key and value share the same layout.
template<typename T, int K_VEC_SIZE>
struct K_vec_m_ {
    using Type = typename V_vec_m_<T, K_VEC_SIZE>::Type;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int K_VEC_SIZE>
struct K_vec_k_ {
    using Type = typename K_vec_m_<T, K_VEC_SIZE>::Type;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MMHA_USE_FP32_ACCUM_FOR_FMA
template<typename T>
struct Qk_vec_accum_fp32_ {};

template<>
struct Qk_vec_accum_fp32_<float> {
    using Type = float;
};

template<>
struct Qk_vec_accum_fp32_<float2> {
    using Type = float2;
};

template<>
struct Qk_vec_accum_fp32_<float4> {
    using Type = float4;
};

// template<> struct Qk_vec_accum_fp32_<uint16_t> { using Type = float;        };
template<>
struct Qk_vec_accum_fp32_<uint32_t> {
    using Type = float2;
};

template<>
struct Qk_vec_accum_fp32_<uint2> {
    using Type = Float4_;
};

template<>
struct Qk_vec_accum_fp32_<uint4> {
    using Type = Float8_;
};

template<>
struct Qk_vec_accum_fp32_<__nv_bfloat16> {
    using Type = float;
};

template<>
struct Qk_vec_accum_fp32_<__nv_bfloat162> {
    using Type = float2;
};

#ifdef ENABLE_BF16
template<>
struct Qk_vec_accum_fp32_<bf16_4_t> {
    using Type = Float4_;
};

template<>
struct Qk_vec_accum_fp32_<bf16_8_t> {
    using Type = Float8_;
};
#endif  // ENABLE_BF16

#ifdef ENABLE_FP8
// template<>
// struct Qk_vec_accum_fp32_<fp8_2_t> {
//     using Type = float2;
// };
template<>
struct Qk_vec_accum_fp32_<fp8_4_t> {
    using Type = Float4_;
};

// template<>
// struct Qk_vec_accum_fp32_<fp8_8_t> {
//     using Type = Float4_;
// };
#endif  // ENABLE_FP8

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct K_vec_accum_fp32_ {};

template<>
struct K_vec_accum_fp32_<float> {
    using Type = float;
};

template<>
struct K_vec_accum_fp32_<float2> {
    using Type = float2;
};

template<>
struct K_vec_accum_fp32_<float4> {
    using Type = float4;
};

template<>
struct K_vec_accum_fp32_<Float8_> {
    using Type = Float8_;
};

template<>
struct K_vec_accum_fp32_<uint32_t> {
    using Type = float2;
};

template<>
struct K_vec_accum_fp32_<uint2> {
    using Type = Float4_;
};

template<>
struct K_vec_accum_fp32_<uint4> {
    using Type = Float8_;
};

template<>
struct K_vec_accum_fp32_<__nv_bfloat16> {
    using Type = float;
};

template<>
struct K_vec_accum_fp32_<__nv_bfloat162> {
    using Type = float2;
};

#ifdef ENABLE_BF16
template<>
struct K_vec_accum_fp32_<bf16_4_t> {
    using Type = Float4_;
};

template<>
struct K_vec_accum_fp32_<bf16_8_t> {
    using Type = Float8_;
};
#endif  // ENABLE_BF16
#ifdef ENABLE_FP8
template<>
struct K_vec_accum_fp32_<__nv_fp8_e4m3> {
    using Type = float;
};

template<>
struct K_vec_accum_fp32_<fp8_2_t> {
    using Type = float2;
};

template<>
struct K_vec_accum_fp32_<fp8_4_t> {
    using Type = Float4_;
};

template<>
struct K_vec_accum_fp32_<fp8_8_t> {
    using Type = Float8_;
};
#endif  // ENABLE_FP8

template<>
struct K_vec_accum_fp32_<int8_t> {
    using Type = float;
};

template<>
struct K_vec_accum_fp32_<int16_t> {
    using Type = float2;
};

template<>
struct K_vec_accum_fp32_<int32_t> {
    using Type = Float4_;
};

template<>
struct K_vec_accum_fp32_<int64_t> {
    using Type = Float8_;
};

#endif  // MMHA_USE_FP32_ACCUM_FOR_FMA

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MMHA_USE_FP32_ACCUM_FOR_OUT
template<typename T>
struct V_vec_accum_fp32_ {};

template<>
struct V_vec_accum_fp32_<float> {
    using Type = float;
};

template<>
struct V_vec_accum_fp32_<float2> {
    using Type = float2;
};

template<>
struct V_vec_accum_fp32_<float4> {
    using Type = float4;
};

template<>
struct V_vec_accum_fp32_<uint32_t> {
    using Type = float2;
};

template<>
struct V_vec_accum_fp32_<uint2> {
    using Type = Float4_;
};

template<>
struct V_vec_accum_fp32_<uint4> {
    using Type = Float8_;
};
#ifdef ENABLE_BF16
template<>
struct V_vec_accum_fp32_<__nv_bfloat162> {
    using Type = float2;
};

template<>
struct V_vec_accum_fp32_<bf16_4_t> {
    using Type = Float4_;
};

template<>
struct V_vec_accum_fp32_<bf16_8_t> {
    using Type = Float8_;
};
#endif  // ENABLE_BF16
#ifdef ENABLE_FP8
// template<>
// struct V_vec_accum_fp32_<fp8_2_t> {
//     using Type = float2;
// };
template<>
struct V_vec_accum_fp32_<fp8_4_t> {
    using Type = Float4_;
};

// template<>
// struct V_vec_accum_fp32_<fp8_8_t> {
//     using Type = Float4_;
// };
#endif  // ENABLE_FP8
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Tout, typename Tin>
__inline__ __device__ constexpr Tout vec_conversion(const Tin& x)
{
    static_assert(std::is_same<Tout, Tin>::value, "Type mismatch");
    return x;
}

template<>
__inline__ __device__ Float8_ vec_conversion<Float8_, uint4>(const uint4& a)
{
    Float8_ fc;
    fc.x = half2_to_float2(a.x);
    fc.y = half2_to_float2(a.y);
    fc.z = half2_to_float2(a.z);
    fc.w = half2_to_float2(a.w);
    return fc;
}

#ifdef ENABLE_BF16
template<>
__inline__ __device__ Float8_ vec_conversion<Float8_, bf16_8_t>(const bf16_8_t& a)
{
    Float8_ fc;
    fc.x = bf1622float2(a.x);
    fc.y = bf1622float2(a.y);
    fc.z = bf1622float2(a.z);
    fc.w = bf1622float2(a.w);
    return fc;
}
#endif  // ENABLE_BF16

#ifdef ENABLE_FP8
// fp8_t
template<>
__inline__ __device__ float vec_conversion<float, __nv_fp8_e4m3>(const __nv_fp8_e4m3& a)
{
    return float(a);
}

template<>
__inline__ __device__ __nv_fp8_e4m3 vec_conversion<__nv_fp8_e4m3, float>(const float& a)
{
    return __nv_fp8_e4m3(a);
}

// fp8_2_t
template<>
__inline__ __device__ float2 vec_conversion<float2, fp8_2_t>(const fp8_2_t& a)
{
    return float2(a);
}

template<>
__inline__ __device__ fp8_2_t vec_conversion<fp8_2_t, float2>(const float2& a)
{
    return fp8_2_t(a);
}

// fp8_4_t
template<>
__inline__ __device__ float4 vec_conversion<float4, fp8_4_t>(const fp8_4_t& a)
{
    return float4(a);
}

template<>
__inline__ __device__ fp8_4_t vec_conversion<fp8_4_t, float4>(const float4& a)
{
    return fp8_4_t(a);
}
#endif  // ENABLE_FP8

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS_PER_KEY, typename Q_vec, typename K_vec, int N>
inline __device__ float qk_dot_(const Q_vec (&q)[N], const K_vec (&k)[N])
{
#ifdef MMHA_USE_FP32_ACCUM_FOR_FMA
    using K_vec_accum = typename K_vec_accum_fp32_<K_vec>::Type;
#else
    using K_vec_accum = K_vec;
#endif
    // Compute the parallel products for Q*K^T (treat vector lanes separately).
    K_vec_accum qk_vec = mul<K_vec_accum, Q_vec, K_vec>(q[0], k[0]);
#pragma unroll
    for (int ii = 1; ii < N; ++ii) {
        qk_vec = fma(q[ii], k[ii], qk_vec);
    }

    // Finalize the reduction across lanes.
    float qk = sum(qk_vec);
#pragma unroll
    for (int mask = THREADS_PER_KEY / 2; mask >= 1; mask /= 2) {
        qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
    }
    return qk;
}

template<int THREADS_PER_KEY, typename Q_vec, typename K_vec, int N>
inline __device__ float qk_scale_dot_(const Q_vec (&q)[N], const K_vec (&k)[N], const float k_scale)
{
#ifdef MMHA_USE_FP32_ACCUM_FOR_FMA
    using K_vec_accum = typename K_vec_accum_fp32_<K_vec>::Type;
#else
    using K_vec_accum = K_vec;
#endif
    // Compute the parallel products for Q*K^T (treat vector lanes separately).
    K_vec_accum k_vec  = mul<K_vec_accum, float, K_vec>(k_scale, k[0]);
    K_vec_accum qk_vec = mul<K_vec_accum, Q_vec, K_vec_accum>(q[0], k_vec);
#pragma unroll
    for (int ii = 1; ii < N; ++ii) {
        K_vec_accum k_vec = mul<K_vec_accum, float, K_vec>(k_scale, k[ii]);
        qk_vec            = fma(q[ii], k_vec, qk_vec);
    }

    // Finalize the reduction across lanes.
    float qk = sum(qk_vec);
#pragma unroll
    for (int mask = THREADS_PER_KEY / 2; mask >= 1; mask /= 2) {
        qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
    }
    return qk;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int THREADS_PER_KEY>
struct Qk_dot {
    template<typename Q_vec, typename K_vec, int N>
    static inline __device__ float dot(const Q_vec (&q)[N], const K_vec (&k)[N])
    {
        return qk_dot_<THREADS_PER_KEY>(q, k);
    }

    template<typename Q_vec, typename K_vec, int N>
    static inline __device__ float scale_dot(const Q_vec (&q)[N], const K_vec (&k)[N], const float k_scale)
    {
#ifdef MMHA_USE_HMMA
        static_assert("HMMA doesn't support k scales");
#endif  // MMHA_USE_HMMA
        return qk_scale_dot_<THREADS_PER_KEY>(q, k, k_scale);
    }

    template<int WARP_SIZE = 32>
    static inline __device__ bool is_leader(const int tidx)
    {
        return (tidx % THREADS_PER_KEY) == 0;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename K_vec>
inline __device__ void hmma_fp32(float4& c, const K_vec& a, K_vec b)
{
    // Not supported.
    assert(false);
}
#if USING_CUDA
template<>
inline __device__ void hmma_fp32(float4& c, const uint32_t& a, uint32_t b)
{
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \n"
                 "    {%0, %1, %2, %3}, \n"
                 "    {%4, %5}, \n"
                 "    {%6}, \n"
                 "    {%0, %1, %2, %3}; \n"
                 : "+f"(c.x), "+f"(c.y), "+f"(c.z), "+f"(c.w)
                 : "r"(a), "r"(a), "r"(b));
}
#endif
template<>
inline __device__ void hmma_fp32(float4& c, const uint2& a, uint2 b)
{
    hmma_fp32(c, a.x, b.x);
    hmma_fp32(c, a.y, b.y);
}

template<>
inline __device__ void hmma_fp32(float4& c, const uint4& a, uint4 b)
{
    hmma_fp32(c, a.x, b.x);
    hmma_fp32(c, a.y, b.y);
    hmma_fp32(c, a.z, b.z);
    hmma_fp32(c, a.w, b.w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename K_vec, int THREADS_PER_KEY, int N>
inline __device__ float qk_hmma_dot_(const K_vec (&q)[N], const K_vec (&k)[N])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750

    // Each quad computes its partial result.
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

#pragma unroll
    for (int ii = 0; ii < N; ++ii) {
        hmma_fp32(acc, q[ii], k[ii]);
    }

    // The position inside the warp.
    int lane = threadIdx.x % 32;

    // The position inside the HMMA instruction.
    int row = lane / 4;
    int col = lane % 4 * 2;

    // The result. Only 1 thread in each quad owns a valid value.
    //
    // Row 0, it's lane  0 (col 0) in acc.x.
    // Row 1, it's lane  4 (col 0) in acc.y.
    // Row 2, it's lane  9 (col 2) in acc.x.
    // Row 3, it's lane 13 (col 2) in acc.y.
    // Row 4, it's lane 18 (col 4) in acc.x.
    // Row 5, it's lane 22 (col 4) in acc.y.
    // Row 6, it's lane 27 (col 6) in acc.x.
    // Row 7, it's lane 31 (col 6) in acc.y.
    //
    float result = (row == col) ? acc.x : acc.y;

    // Do the reduction inside the warp.
    if (THREADS_PER_KEY > 4) {
        result += __shfl_xor_sync(unsigned(-1), result, 4);
    }
    if (THREADS_PER_KEY > 8) {
        result += __shfl_xor_sync(unsigned(-1), result, 9);
    }
    if (THREADS_PER_KEY > 16) {
        result += __shfl_xor_sync(unsigned(-1), result, 18);
    }

    // The warp leader has the correct value.
    return result;

#else  // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 750
    return 0.f;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS_PER_KEY>
struct Qk_dot<uint16_t, THREADS_PER_KEY> {
    template<typename Q_vec, typename K_vec, int N>
    static inline __device__ float dot(const Q_vec (&q)[N], const K_vec (&k)[N])
    {
#if __CUDA_ARCH__ >= 750 && defined(MMHA_USE_HMMA)
        return qk_hmma_dot_<K_vec, THREADS_PER_KEY, N>(q, k);
#else
        return qk_dot_<THREADS_PER_KEY>(q, k);
#endif  // defined MMHA_USE_HMMA
    }

    template<typename Q_vec, typename K_vec, int N>
    static inline __device__ float scale_dot(const Q_vec (&q)[N], const K_vec (&k)[N], const float k_scale)
    {
#ifdef MMHA_USE_HMMA
        static_assert("HMMA doesn't support k scales");
#endif  // MMHA_USE_HMMA
        return qk_scale_dot_<THREADS_PER_KEY>(q, k, k_scale);
    }

    template<int WARP_SIZE = 32>
    static inline __device__ bool is_leader(const int tidx)
    {
        // Use HMMA.FP32, leader threads are in the diagonal roughly (0, 4, 9, 13, 18, 22, 27, 31).
#if __CUDA_ARCH__ >= 750 && defined(MMHA_USE_HMMA)
        int leader = 0;
        // The thread position inside the warp.
        int lane = tidx % WARP_SIZE;
        if (THREADS_PER_KEY == 4) {
            leader = int(lane / 8);
        }
        else {
            leader = int(lane / THREADS_PER_KEY) * int(THREADS_PER_KEY / 8);
        }
#else
        const bool leader = 0;
#endif  // defined MMHA_USE_HMMA
        return (tidx % THREADS_PER_KEY) == leader;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int WARPS_PER_BLOCK, int WARP_SIZE = 32>
inline __device__ float block_sum(float* red_smem, float sum)
{

    // Decompose the thread index into warp / lane.
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

// Compute the sum per warp.
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
    }

    // Warp leaders store the data to shared memory.
    if (lane == 0) {
        red_smem[warp] = sum;
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // The warps compute the final sums.
    if (lane < WARPS_PER_BLOCK) {
        sum = red_smem[lane];
    }

// Parallel reduction inside the warp.
#pragma unroll
    for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
        sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
    }

    // Broadcast to other threads.
    return __shfl_sync(uint32_t(-1), sum, 0);
}

#if defined(MMHA_USE_FP32_ACCUM_FOR_LOGITS)

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float cast_to_float(float u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 cast_to_float(float2 u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 cast_to_float(float4 u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ cast_to_float(Float4_ u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ cast_to_float(Float8_ u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 cast_to_float(uint32_t u)
{
    return half2_to_float2(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ cast_to_float(uint2 u)
{
    Float4_ tmp;
    tmp.x = half2_to_float2(u.x);
    tmp.y = half2_to_float2(u.y);
    return tmp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ cast_to_float(uint4 u)
{
    Float8_ tmp;
    tmp.x = half2_to_float2(u.x);
    tmp.y = half2_to_float2(u.y);
    tmp.z = half2_to_float2(u.z);
    tmp.w = half2_to_float2(u.w);
    return tmp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 cast_to_float(__nv_bfloat162 u)
{
    float2 tmp;
    tmp = __bfloat1622float2(u);
    return tmp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ cast_to_float(bf16_4_t u)
{
    Float4_ tmp;
    tmp.x = __bfloat1622float2(u.x);
    tmp.y = __bfloat1622float2(u.y);
    return tmp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ cast_to_float(bf16_8_t u)
{
    Float8_ tmp;
    tmp.x = __bfloat1622float2(u.x);
    tmp.y = __bfloat1622float2(u.y);
    tmp.z = __bfloat1622float2(u.z);
    tmp.w = __bfloat1622float2(u.w);
    return tmp;
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline __device__ __host__ T divUp(T m, T n)
{
    return (m + n - 1) / n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline __device__ __host__ T div(T m, T n)
{
    return m / n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct kernel_type_t {
    using Type = T;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Compute the largest supported head size (dh_max). It must be the smallest power-of-2 that is not strictly smaller
// than the head size (dh).
inline __device__ __host__ constexpr unsigned dh_max(unsigned dh)
{
    return next_power_of_two(const_max(dh, 32u));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline __device__ __host__ constexpr unsigned threads_per_value(unsigned dh_max)
{
    return dh_max * sizeof(T) / 16;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, unsigned Dh_MAX>
inline __device__ __host__ constexpr unsigned threads_per_key()
{
    // Since we want to perform the reduction entirely within a warp, the number of threads per key
    // is capped at 32.
    constexpr unsigned threads = (unsigned)(Dh_MAX * sizeof(T) / 16u);
    if ((threads & (threads - 1)) != 0) {
        assert(false);  // Not a power of two.
    }
    return std::min(32u, threads);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ constexpr uint32_t shfl_mask(int threads)
{
    assert(threads <= 32);
    return threads == 32 ? -1u : (1u << threads) - 1u;
}

inline __device__ constexpr uint32_t shfl_mask_and_index(int threads, int index)
{
    assert(threads <= 32);
    assert((index + 1) * threads <= 32);
    return threads == 32 ? -1u : ((1u << threads) - 1u) << (index * threads);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, typename T_VEC, unsigned VECS_PER_CHUNK>
__device__ inline constexpr uint2 chunk_index(unsigned tidx)
{
    // The chunk associated with the thread.
    auto const idx_chunk = tidx / VECS_PER_CHUNK;

    // The position of the T_VEC vector in that chunk associated with the thread.
    static_assert(sizeof(T_VEC) % sizeof(T) == 0);
    unsigned constexpr kVecSize{sizeof(T_VEC) / sizeof(T)};
    auto const idx_vec = (tidx % VECS_PER_CHUNK) * kVecSize;

    return uint2{idx_chunk, idx_vec};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Tk, typename V_vec_accum, typename V_vec_m, bool INT8_KV_CACHE, bool FP8_KV_CACHE>
inline __device__ void
Logit_value_fma(V_vec_accum& out, const Tk* logits_smem, const V_vec_m& v_vec, const float v_scale, const bool is_mask)
{
#if defined(MMHA_USE_FP32_ACCUM_FOR_LOGITS)
    float logit = is_mask ? 0.f : reinterpret_cast<float*>(const_cast<Tk*>(logits_smem))[0];
    if constexpr (INT8_KV_CACHE) {
        V_vec_accum v_vec_ = mul<V_vec_accum, float, V_vec_m>(v_scale, v_vec);
        out                = fma(logit, cast_to_float(v_vec_), out);
    }
    else if constexpr (FP8_KV_CACHE) {
#ifdef MMHA_FP8_SCALE_P_INSTEAD_OF_V
        out = fma(logit, cast_to_float(v_vec), out);
#else
        V_vec_accum v_vec_ = mul<V_vec_accum, float, V_vec_m>(v_scale, v_vec);
        out                = fma(logit, cast_to_float(v_vec_), out);
#endif  // MMHA_FP8_SCALE_P_INSTEAD_OF_V
    }
    else {
        out = fma(logit, cast_to_float(v_vec), out);
    }
#else  // MMHA_USE_FP32_ACCUM_FOR_LOGITS
    Tk logit = is_mask ? Tk(0.f) : logits_smem[0];
    if constexpr (INT8_KV_CACHE) {
        V_vec_accum v_vec_ = mul<V_vec_accum, float, V_vec_m>(v_scale, v_vec);
        out                = fma(logit, v_vec_, out);
    }
    else if constexpr (FP8_KV_CACHE) {
#ifdef MMHA_FP8_SCALE_P_INSTEAD_OF_V
        out = fma(logit, v_vec, out);
#else
        V_vec_accum v_vec_ = mul<V_vec_accum, float, V_vec_m>(v_scale, v_vec);
        out                = fma(logit, v_vec_, out);
#endif  // MMHA_FP8_SCALE_P_INSTEAD_OF_V
    }
    else {
        out = fma(logit, v_vec, out);
    }
#endif  // MMHA_USE_FP32_ACCUM_FOR_LOGITS
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The type of the inputs. Supported types: float, uint16_t, nv_bfloat16.
    typename T,
    // The type of the cache.
    typename Tcache,
    // Type of struct containing KV cache
    typename KVCacheBuffer,
    // The hidden dimension per head.
    unsigned Dh,
    // The number of threads in a threadblock.
    unsigned THREADS_PER_BLOCK,
    // Whether cross attention is enabled
    bool DO_CROSS_ATTENTION,
    // Whether has beams.
    bool HAS_BEAMS,
    // Whether enable multi-block mode for long-sequence-length.
    bool DO_MULTI_BLOCK = false,
    // The number of threads per key.
    unsigned THREADS_PER_KEY = threads_per_key<T, dh_max(Dh)>(),
    // The number of threads per value.
    unsigned THREADS_PER_VALUE = threads_per_value<T>(dh_max(Dh)),
    // The unroll factor for loading from K cache.
    // Set it default to 4 for higher occupancy (by reducing registers usage).
    unsigned K_LOOP_UNROLL = 4,
    // The unroll factor for loading from V cache.
    unsigned V_LOOP_UNROLL = 8>
__global__ void masked_multihead_attention_kernel(Multihead_attention_params<T, DO_CROSS_ATTENTION> params,
                                                  KVCacheBuffer                                     kvCacheBuffer)
{

    using Tk = typename kernel_type_t<T>::Type;
    // Use 8bit cache.
    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;
    // FP8 KV Cache.
#ifdef ENABLE_FP8
    static constexpr bool FP8_KV_CACHE = std::is_same<Tcache, __nv_fp8_e4m3>::value;
#else
    static constexpr bool FP8_KV_CACHE = false;
#endif
    // INT8 KV Cache.
    static constexpr bool INT8_KV_CACHE = std::is_same<Tcache, int8_t>::value;

    // The size of a warp.
    constexpr unsigned WARP_SIZE{32};
    // The number of warps in a threadblock.
    constexpr unsigned WARPS_PER_BLOCK{THREADS_PER_BLOCK / WARP_SIZE};

    // The maximum hidden size per head.
    constexpr auto Dh_MAX    = dh_max(Dh);
    constexpr bool IS_Dh_MAX = Dh == Dh_MAX;
    static_assert(Dh_MAX >= WARP_SIZE);
    static_assert(Dh_MAX >= Dh);

    // The maximum sequence length in the cyclic kv_cache, i.e., an upper bound on L.
    // Note that the maximum sequence length supported by the model might be greater than this.
    // Note max_kv_cache_length is maximum of cyclic_kv_cache_length among all layers.
    // By default, you can assume that they are the same.
    const auto cyclic_kv_cache_len = static_cast<unsigned>(params.cyclic_kv_cache_length);
    // The current timestep (including paddings).
    // It is only used to calculate the smem stride.
    const auto timestep = static_cast<unsigned>(DO_MULTI_BLOCK ? params.timesteps_per_block : params.timestep);

#ifdef ENABLE_MULTI_BLOCK_OPTION
    constexpr bool MULTI_BLOCK_FLAG = DO_MULTI_BLOCK;
#else
    constexpr bool MULTI_BLOCK_FLAG = false;
#endif

    // Use smem_size_in_bytes (above) to determine the amount of shared memory.
    extern __shared__ char smem_[];

    // The shared memory for the Q*K^T values and partial logits in softmax.
    auto qk_smem = reinterpret_cast<float*>(smem_);

    __shared__ float qk_current_smem[1];

    // The shared memory for the logits. For FP32, that's the same buffer as qk_smem.
    char* logits_smem_ = smem_;
#ifndef MMHA_USE_FP32_ACCUM_FOR_LOGITS
    if (sizeof(Tk) != 4) {
        const auto max_timesteps = DO_CROSS_ATTENTION ? cyclic_kv_cache_len : min(timestep, cyclic_kv_cache_len);
        logits_smem_ += divUp(max_timesteps + 1, 4u) * 16;
    }
    Tk* logits_smem = reinterpret_cast<Tk*>(logits_smem_);
#else
    float* logits_smem = reinterpret_cast<float*>(logits_smem_);
#endif

    __shared__ Tk logits_current_smem[1];

    // The shared memory to do the final reduction for the output values. Reuse qk_smem.
    Tk* out_smem = reinterpret_cast<Tk*>(smem_);

    // The shared memory buffers for the block-wide reductions. One for max, one for sum.
    __shared__ float red_smem[WARPS_PER_BLOCK * 2];

    // A vector of Q or K elements for the current timestep.
    using Qk_vec_m = typename Qk_vec_m_<T, Dh_MAX>::Type;  // with memory-used precision
    using Qk_vec_k = typename Qk_vec_k_<T, Dh_MAX>::Type;  // with kernel-used precision
#ifdef MMHA_USE_FP32_ACCUM_FOR_FMA
    using Qk_vec_accum = typename Qk_vec_accum_fp32_<Qk_vec_k>::Type;
#else
    using Qk_vec_accum = Qk_vec_k;
#endif

    // Make sure the hidden dimension per head is a multiple of the number of threads per key.
    static_assert(Dh_MAX % THREADS_PER_KEY == 0);  // trivially satisfied since THREADS_PER_KEY in {1, 2, 4}

    // The number of elements per vector.
    // Each thread will handle 16 bytes.
    constexpr int K_VEC_SIZE = 16u / sizeof(T);
    // Make sure the hidden size per head is a multiple of the vector size.
    static_assert(Dh_MAX % K_VEC_SIZE == 0);
    // The type of queries and keys for the math in the Q*K^T product.
    using K_vec_k = typename K_vec_k_<T, K_VEC_SIZE>::Type;
    // Only used when key cache is quantized to 8 bits.
    using K_vec_m = typename packed_type<Tcache, num_elems<K_vec_k>::value>::type;
#ifdef MMHA_USE_FP32_ACCUM_FOR_FMA
    using K_vec_accum = typename Qk_vec_accum_fp32_<K_vec_k>::Type;
#else
    using K_vec_accum = K_vec_k;
#endif

    // Use alignment for safely casting the shared buffers as Qk_vec_k and K_vec_k.
    // Shared memory to store Q inputs.
    __shared__ __align__(const_max(sizeof(Qk_vec_k), sizeof(K_vec_k))) Tk q_smem[Dh_MAX];
    __shared__ __align__(const_max(sizeof(Qk_vec_k), sizeof(K_vec_k))) Tk k_smem[Dh_MAX];

    // Make sure the hidden dimension per head is a multiple of the number of threads per value.
    static_assert(Dh_MAX % THREADS_PER_VALUE == 0);  // trivially satisfied since THREADS_PER_VALUE == Dh_MAX / p

    // The number of elements per vector.
    constexpr int V_VEC_SIZE = Dh_MAX / THREADS_PER_VALUE;
    // A vector of V elements for the current timestep.
    using V_vec_k = typename V_vec_k_<T, V_VEC_SIZE>::Type;
    // Only used when value cache is quantized to 8 bits.
    using V_vec_m = typename packed_type<Tcache, num_elems<V_vec_k>::value>::type;
    static_assert(V_VEC_SIZE == sizeof(V_vec_k) / sizeof(T));

    // This could be one of the reasons to have a separate kernel for cross attention
    constexpr auto bias_smem_size = DO_CROSS_ATTENTION ? Dh_MAX : 1u;
    __shared__     __align__(const_max(const_max(sizeof(Qk_vec_k), sizeof(K_vec_k)), sizeof(V_vec_k)))
        Tk         bias_smem[bias_smem_size];

    // The number of elements per vector.
    constexpr unsigned QK_VEC_SIZE{sizeof(Qk_vec_m) / sizeof(T)};
    // Make sure the hidden size per head is a multiple of the vector size.
    static_assert(Dh_MAX % QK_VEC_SIZE == 0);
    // We will use block wide reduction if needed
    // The number of vectors per Dh_MAX.
    constexpr unsigned QK_VECS_PER_Dh_MAX{Dh_MAX / QK_VEC_SIZE};
    static_assert(THREADS_PER_BLOCK >= QK_VECS_PER_Dh_MAX);

    // The batch/beam idx
    const auto batch_beam_idx = blockIdx.y;
    if (params.finished != nullptr && params.finished[batch_beam_idx]) {
        return;
    }

    // The head.
    const unsigned hi{blockIdx.x};
    // The head index of keys and values adjusted for MQA/GQA.
    const int      qhead_per_kv{params.num_heads / params.num_kv_heads};
    const unsigned hi_kv{hi / qhead_per_kv};
    // The number of heads.
    const auto num_heads = static_cast<unsigned>(params.num_heads);
    // The number of heads for keys and values adjusted for MQA/GQA.
    const auto num_heads_kv = static_cast<unsigned>(params.num_kv_heads);

    // The thread in the block.
    const unsigned tidx{threadIdx.x};

    // The column tile along L dimension on K^T -- noted as T_c in flash-attention paper
    const unsigned c_tile{MULTI_BLOCK_FLAG ? blockIdx.z : 0};

    // Indicate if we need to compute the K/V cache element (add KV bias, IA3, RoPE, etc.) and update the cache.
    // For Self-Attention, it's always required.
    // For Cross-Attention, as everything is pre-computed,
    // in the context phase of the encoder, it's not needed in that kernel.
    // Therefore, HANDLE_KV is !DO_CROSS_ATTENTION and irrelevant of timestep.
    static constexpr bool HANDLE_KV{!DO_CROSS_ATTENTION};

    // While doing the product Q*K^T for the different keys we track the max.
    float qk_max = -FLT_MAX;

    float qk = 0.0F;

    // Do we have a relative attention bias?
    bool has_relative_attention_bias = params.relative_attention_bias != nullptr;
    // Compute relative attention bias on the fly, with relative attention table [head_num/TP, num_buckets] passed in.
    // num_buckets passed as relative_attention_bias_stride, max_distance passed as params.max_distance
    // this is a common optimization for both self attention and cross attention
    const bool implicit_rel_attn_bias = params.max_distance != 0 && has_relative_attention_bias;
    int        relative_attention_bias_stride =
        params.relative_attention_bias_stride;  // num_buckets might be modified below, save it beforehand
    int max_distance = params.max_distance;

    // The actual sequence length excluding the paddings.
    // minus 1 because it includes the current timestep while tlength denotes the kv cache length.
    const int tlength             = DO_CROSS_ATTENTION ? params.memory_length_per_sample[batch_beam_idx] - 1 :
                                                         (params.length_per_sample ?
                                                              (params.length_per_sample[batch_beam_idx] + params.max_prefix_prompt_length) :
                                                              static_cast<int>(timestep));
    bool      count_prefix_length = params.count_prefix_length;
    // We will use cyclic kv cache when it exceeds the limit.
    // The length position for storing new key and value.
    const int cyclic_tlength = tlength % cyclic_kv_cache_len;
    // The actual kv cache length.
    // tlength is the past length actually.
    const int kv_loop_length = min(tlength, cyclic_kv_cache_len);
    // The context length for beam searching optimization (all points to beam 0).
    // TODO: with cyclic kv cache, we set it 0 for now (will optimize in the future)
    // as context kv cache might be overwritten by the new kv cache
    const int beam0_context_length =
        HAS_BEAMS && tlength > cyclic_kv_cache_len ? 0 : params.input_lengths[batch_beam_idx];

    // The offset in the Q and K buffer also accounts for the batch.
    const auto qk_vec_idx      = tidx * QK_VEC_SIZE;
    const auto is_valid_qk_vec = qk_vec_idx < Dh;

    const bool load_qkv_quant        = params.qkv_scale_quant_orig != nullptr;
    const bool write_attention_quant = params.attention_out_scale_orig_quant != nullptr;

    // Quant/Dequant scales for 8bits kv cache.
    using T_scale = typename kv_cache_scale_type_t<T, Tcache>::Type;
    T_scale     kv_scale_orig_quant, kv_scale_quant_orig;
    const float kv_scale_quant_orig_f = (ENABLE_8BITS_CACHE ? params.kv_scale_quant_orig[0] : 1.0f);
    convert_from_float(&kv_scale_quant_orig, kv_scale_quant_orig_f);
    convert_from_float(&kv_scale_orig_quant, (ENABLE_8BITS_CACHE ? params.kv_scale_orig_quant[0] : 1.0f));

    // Up to QK_VECS_PER_Dh_MAX threads load Q and K + the bias values for the current timestep.
    // Trigger the loads from the Q and K buffers.
    Qk_vec_k q, k, q_bias, k_bias;
    zero(q);
    zero(k);
    zero(q_bias);
    zero(k_bias);
    float rotary_embedding_base  = params.rotary_embedding_base;
    float rotary_embedding_scale = params.rotary_embedding_scale;
    if (is_valid_qk_vec) {

        // Query
        // The stride between tokens. We may be able to always use params.stride.
        uint32_t q_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads * Dh);
        // The offset.
        const auto q_offset = flat_index_strided3(batch_beam_idx, hi, qk_vec_idx, q_stride, Dh);

        if (load_qkv_quant) {
            using Packed_Int8_t  = typename packed_type<int8_t, num_elems<Qk_vec_m>::value>::type;
            using Packed_Float_t = typename packed_type<float, num_elems<Qk_vec_m>::value>::type;
            const auto q_scaling = params.qkv_scale_quant_orig[0];
            const auto q_quant =
                *reinterpret_cast<const Packed_Int8_t*>(&reinterpret_cast<const int8_t*>(params.q)[q_offset]);
            convert_from_float(&q, mul<Packed_Float_t, float>(q_scaling, float_from_int8(q_quant)));
        }
        else {
            q = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(&params.q[q_offset]));
        }

        if constexpr (DO_CROSS_ATTENTION) {
            const auto k_idx      = QK_VEC_SIZE * tidx;
            const int  inBlockIdx = kvCacheBuffer.getKVLocalIdx(cyclic_tlength, hi, Dh, k_idx);
            Tcache*    k_cache = reinterpret_cast<Tcache*>(kvCacheBuffer.getKBlockPtr(batch_beam_idx, cyclic_tlength));

            k = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(&k_cache[inBlockIdx]));
        }
        else {
            // Key
            // The stride between tokens. We may be able to always use params.stride.
            uint32_t k_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads_kv * Dh);
            // The offset.
            const auto k_offset = flat_index_strided3(batch_beam_idx, hi_kv, qk_vec_idx, k_stride, Dh);

            if (load_qkv_quant) {
                using Packed_Int8_t  = typename packed_type<int8_t, num_elems<Qk_vec_m>::value>::type;
                using Packed_Float_t = typename packed_type<float, num_elems<Qk_vec_m>::value>::type;
                const auto k_scaling = params.qkv_scale_quant_orig[1];
                const auto k_quant =
                    *reinterpret_cast<const Packed_Int8_t*>(&reinterpret_cast<const int8_t*>(params.k)[k_offset]);

                convert_from_float(&k, mul<Packed_Float_t, float>(k_scaling, float_from_int8(k_quant)));
            }
            else {
                k = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(&params.k[k_offset]));
            }
        }

        if (params.q_bias != nullptr) {
            const auto q_bias_offset = flat_index2(hi, qk_vec_idx, Dh);
            q_bias =
                vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(&params.q_bias[q_bias_offset]));
        }
        if (HANDLE_KV && params.k_bias != nullptr) {
            const auto k_bias_offset = flat_index2(hi_kv, qk_vec_idx, Dh);
            k_bias =
                vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(&params.k_bias[k_bias_offset]));
        }
    }

    // Computes the Q/K values with bias.
    q = add(q, q_bias);
    if (HANDLE_KV) {
        k = add(k, k_bias);
    }

    // The width of the beam.
    const auto beam_width = static_cast<unsigned>(params.beam_width);
    // The batch idx.
    const int batch_idx = batch_beam_idx / beam_width;
    // Do we apply IA3?
    const bool do_ia3 = HANDLE_KV && params.ia3_tasks != nullptr;
    // Compute the IA3 task. One per batch index.
    const auto ia3_ti_hi = do_ia3 ? flat_index2(static_cast<unsigned>(params.ia3_tasks[batch_idx]), hi, num_heads) : 0;

    if (do_ia3 && is_valid_qk_vec) {
        k = mul<Qk_vec_k, Qk_vec_k, Qk_vec_k>(k,
                                              vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(
                                                  &params.ia3_key_weights[flat_index2(ia3_ti_hi, qk_vec_idx, Dh)])));
    }
    const int input_len            = (params.input_lengths == nullptr) ? 0 : params.input_lengths[batch_beam_idx];
    int       prefix_prompt_length = (params.prefix_prompt_lengths == nullptr) ? 0 : params.prefix_prompt_lengths[batch_beam_idx];
    if (params.rotary_embedding_dim > 0) {
        const int position_id = params.position_ids == nullptr ? -1 : params.position_ids[batch_beam_idx];
        attention_rope(params.rotary_embedding_style,
                      q,
                      k,
                      reinterpret_cast<T*>(smem_),
                      tidx,
                      tlength,
                      timestep,
                      params.rotary_embedding_dim,
                      params.length_per_sample[batch_beam_idx],
                      params.rotary_embedding_base,
                      params.rotary_embedding_scale,
                      params.rotary_embedding_max_positions,
                      params.original_max_position_embeddings,
                      params.base_scale,
                      position_id,
                      input_len,
                      prefix_prompt_length,
                      count_prefix_length,
                      params.logn_seq_len,
                      HANDLE_KV);
    }
    __syncthreads();

    if (params.use_logn_attn && is_valid_qk_vec) {
        logn_attention(q, tlength, params.logn_seq_len);
    }

    // For the same reason as HANDLE_KV, no compute needed in Cross-Attention's 1st step
    // Store Q K vectors to shared memory, and calculate QK.
    if (qk_vec_idx < Dh_MAX) {

        // Store the Q values to shared memory.
#ifdef MMHA_FP8_SCALE_Q_INSTEAD_OF_K
        if constexpr (FP8_KV_CACHE) {
            // There are many more elements from K than elements from Q so we pre-scale Q instead
            // of scaling all the elements from K. It helps reduce the number of ops.
            Qk_vec_k scaled_q;
            zero(scaled_q);
            if (is_valid_qk_vec) {
                scaled_q = mul<Qk_vec_k, Tk, Qk_vec_k>(kv_scale_quant_orig, q);
            }
            reinterpret_cast<Qk_vec_k*>(&q_smem[qk_vec_idx])[0] = scaled_q;
        }
        else
#endif
        {
            // Set padded Dh to 0 for the correctness of QK (when Dh != Dh_Max).
            Qk_vec_k zero_q;
            zero(zero_q);
            reinterpret_cast<Qk_vec_k*>(&q_smem[qk_vec_idx])[0] = is_valid_qk_vec ? q : zero_q;
        }

        // Store the K values to shared memory.
        // We store K values from shared memory to global memory
        //  when the target position of K cache in global memory has been accessed (in the case of cyclic kv cache)
        reinterpret_cast<Qk_vec_k*>(&k_smem[qk_vec_idx])[0] = k;

        // Compute \sum_i Q[i] * K^T[i] for the current timestep.
        qk = dot<Qk_vec_accum, Qk_vec_k>(q, k);
        if (QK_VECS_PER_Dh_MAX <= WARP_SIZE) {
#pragma unroll
            for (int mask = QK_VECS_PER_Dh_MAX / 2; mask >= 1; mask /= 2) {
                qk += __shfl_xor_sync(shfl_mask(QK_VECS_PER_Dh_MAX), qk, mask);
            }
        }
    }

    if (QK_VECS_PER_Dh_MAX > WARP_SIZE) {
        constexpr int WARPS_PER_RED = (QK_VECS_PER_Dh_MAX + WARP_SIZE - 1) / WARP_SIZE;
        qk                          = block_sum<WARPS_PER_RED>(&red_smem[WARPS_PER_RED], qk);
    }

    // Pre-compute the pointer for the relative attention bias.
    const T* relative_attention_bias_ptr       = nullptr;
    const T* relative_attention_bias_ptr_fixed = nullptr;  // record the base for offset
    if (has_relative_attention_bias) {
        // "hi" is unsigned, subtracting int from unsigned int causes underflow. Cast to int
        int64_t offset                    = implicit_rel_attn_bias ?
                                                ((int64_t)hi * relative_attention_bias_stride - tlength) :
                                                ((int64_t)hi * relative_attention_bias_stride + tlength) * relative_attention_bias_stride;
        relative_attention_bias_ptr       = &params.relative_attention_bias[offset];
        relative_attention_bias_ptr_fixed = &params.relative_attention_bias[offset];
    }

    // Load the value.
    float relative_attention_bias = 0.f;
    if (has_relative_attention_bias && tidx == 0) {
        // TODO: Use a better way to convert from T to float.
        relative_attention_bias = add(relative_attention_bias, relative_attention_bias_ptr[tlength]);
    }

    // Store that value in shared memory. Keep the Q*K^T value in register for softmax.
    if (tidx == 0) {
        // Normalize qk.
        qk = qk * params.inv_sqrt_dh + relative_attention_bias;

        // We don't need to apply the linear position bias here since qi - ki = 0 yields the position bias 0.
        qk_max = qk;

        // Store Q*K^T to shared memory.
        if (MULTI_BLOCK_FLAG) {
            qk_current_smem[0] = qk;
        }
        else {
            // We need to store the qk result to the end of the qk_smem for cyclic kv cache (+ 1 for smem memory
            // allocation) because the previous cache will still write to the new_cache_pos of qk_smem.
            qk_smem[kv_loop_length] = qk;
        }
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    constexpr unsigned K_ELTS_PER_CHUNK{THREADS_PER_KEY * K_VEC_SIZE};

    // The positions of the cache buffer (for this B * H) and the vector within that chunk associated with this
    // thread.
    const auto k_idx = chunk_index<T, K_vec_k, THREADS_PER_KEY>(tidx);

    // The number of vectors per thread.
    constexpr unsigned K_VECS_PER_THREAD{Dh_MAX / K_ELTS_PER_CHUNK};
    static_assert(Dh_MAX == K_ELTS_PER_CHUNK * K_VECS_PER_THREAD);

    // Load the Q values from shared memory. The values are reused during the loop on K.
    K_vec_accum q_vec[K_VECS_PER_THREAD];
#pragma unroll
    for (unsigned ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
        q_vec[ii] = vec_conversion<K_vec_accum, K_vec_k>(
            *reinterpret_cast<const K_vec_k*>(&q_smem[flat_index2(ii, k_idx.y, K_ELTS_PER_CHUNK)]));
    }

    // The number of timesteps loaded per iteration, i.e., (THREADS_PER_BLOCK * THREADS_PER_BLOCK) / 256 <= 256
    constexpr unsigned K_PER_ITER{THREADS_PER_BLOCK / THREADS_PER_KEY};
    // The number of keys per warp.
    constexpr unsigned K_PER_WARP{WARP_SIZE / THREADS_PER_KEY};
    // The number of unrolled keys per warp.
    constexpr unsigned UNROLLED_K_PER_WARP = K_PER_WARP * K_LOOP_UNROLL;
    // The number of unrolled keys per ieration.
    constexpr unsigned UNROLLED_K_PER_ITER = K_PER_ITER * K_LOOP_UNROLL;

    // Base pointer for the row of pointers to k cache blocks
    void** k_cache_base_row_ptr = reinterpret_cast<void**>(kvCacheBuffer.getRowPtr(KVIdxType::K_IDX, batch_beam_idx));

    const auto timesteps_per_block = static_cast<unsigned>(params.timesteps_per_block);

    // Pick a number of keys to make sure all the threads of a warp enter (due to shfl_sync).
    // Take all previous cache as context when we have no beam searching in order to batch as many LDGs as possible.
    const int context_length =
        DO_CROSS_ATTENTION ? kv_loop_length : (HAS_BEAMS ? beam0_context_length : kv_loop_length);
    // Clarifications:
    // - in self attn, input_length is input text length, tlength is current timestep
    // - in cross attn, input_length is *decoder* input length (usually 1), tlength is *encoder* input context length
    // - in beam search, since the cache during generation is organized differently, the following KV compute needs
    // split into context cache compute and generation cache compute
    // - for self attn, no-beam search: entire cache can be treated as context cache --> context_length = tlength
    // - for self attn, beam search: cache of input text length is context cache, other are generation cache -->
    // context_length = input_length
    // - for cross attn, no-beam/beam search: cache length is fixed, not differ context/generation cache -->
    // context_length = tlength Suggestion: we could have a flag HANDLE_GEN_CACHE

    const auto context_ti_end =
        MULTI_BLOCK_FLAG ? divUp(timesteps_per_block, UNROLLED_K_PER_WARP) * UNROLLED_K_PER_WARP :
                           divUp(static_cast<unsigned>(context_length), UNROLLED_K_PER_WARP) * UNROLLED_K_PER_WARP;

    // The generation ti_end.
    const auto generation_ti_end = MULTI_BLOCK_FLAG ?
                                       divUp(timesteps_per_block, K_PER_WARP) * K_PER_WARP :
                                       divUp(static_cast<unsigned>(kv_loop_length), K_PER_WARP) * K_PER_WARP;

    // Iterate over the keys/timesteps to compute the various (Q*K^T)_{ti} values.
    // Note max_kv_cache_length is maximum of cyclic_kv_cache_length among all layers.
    // By default, you can assume that they are the same.
    const auto bi_seq_len_offset = static_cast<std::size_t>(batch_beam_idx) * params.max_kv_cache_length;
    // Beam indices are based on the max_kv_cache_length while each layer may have different cyclic_kv_cache_length
    // So we need to rebuild the beam_indices if max_kv_cache_length is not equal to cyclic_kv_cache_length.
    const int* beam_indices = HAS_BEAMS ? &params.cache_indir[bi_seq_len_offset] : nullptr;

    const auto c_tile_times_timesteps_per_block = c_tile * timesteps_per_block;  // 0 if !MULTI_BLOCK_FLAG

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Key cache loops for dot(Q, K).

    // Is it the leader?
    const bool is_leader = Qk_dot<T, THREADS_PER_KEY>::is_leader(tidx);

    // The slope for ALiBi.
    float linear_bias_slope = 0.f;
    if (params.linear_bias_slopes != nullptr) {
        // TODO: Use a cleaner code to convert from T to float.
        linear_bias_slope = mul<float>(params.linear_bias_slopes[hi], 1.f);
    }

    // Handle only context key cache with beam searching.
    // Handle both context and generation key cache without beam searching.
    // Explicit batching of LDGs (by K_LOOP_UNROLL) as it doesn't depend on indirection tables.
    for (int ti = k_idx.x; ti < context_ti_end; ti += UNROLLED_K_PER_ITER) {
        const int time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;

        // The keys loaded from the key cache.
        K_vec_m k_vec_cache[K_LOOP_UNROLL][K_VECS_PER_THREAD];

#pragma unroll
        for (int k_loop = 0; k_loop < K_LOOP_UNROLL; ++k_loop) {
#pragma unroll
            for (int k_vec_i = 0; k_vec_i < K_VECS_PER_THREAD; ++k_vec_i) {
                // Make sure we read data within the bound.
                // Dh OOB values will be handled by zero_q.
                // Seq OOB values will be masked out when storing back to smem.
                auto const jj             = min(k_idx.y + k_vec_i * K_ELTS_PER_CHUNK, Dh - K_VEC_SIZE);
                const int  valid_time_now = min(time_now + k_loop * K_PER_ITER, context_length - 1);
                const int  seqIdx         = batch_idx * beam_width;

                // Base pointer to k cache block for beam's batch
                Tcache* k_cache_batch = reinterpret_cast<Tcache*>(kvCacheBuffer.getKBlockPtr(seqIdx, valid_time_now));

                int inBlockIdx               = kvCacheBuffer.getKVLocalIdx(valid_time_now, hi_kv, Dh, jj);
                k_vec_cache[k_loop][k_vec_i] = *reinterpret_cast<const K_vec_m*>(&k_cache_batch[inBlockIdx]);
            }
        }

#pragma unroll
        for (int k_loop = 0; k_loop < K_LOOP_UNROLL; ++k_loop) {
            const int local_time_now = time_now + k_loop * K_PER_ITER;
            const int local_ti       = ti + k_loop * K_PER_ITER;
             
            float k_scale = 1.f;

            if constexpr (ENABLE_8BITS_CACHE) {
                const int valid_time_now = min(time_now + k_loop * K_PER_ITER, context_length - 1);
                const int seqIdx         = batch_idx * beam_width;
                float*    k_scale_ptr    = reinterpret_cast<float*>(kvCacheBuffer.getKScalePtr(seqIdx, valid_time_now));
                int       inScaleIdx     = kvCacheBuffer.getKVScaleLocalIdx(valid_time_now, hi_kv);
                k_scale                  = k_scale_ptr[inScaleIdx];
            }

            // Perform the dot product and normalize qk.
            //
            // WARNING: ALL THE THREADS OF A WARP MUST ENTER!!!
            K_vec_m k_vec[K_VECS_PER_THREAD];
#pragma unroll
            for (int k_vec_i = 0; k_vec_i < K_VECS_PER_THREAD; ++k_vec_i) {
                k_vec[k_vec_i] = *reinterpret_cast<K_vec_m*>(&k_vec_cache[k_loop][k_vec_i]);
            }

            // Is it active?
            const bool is_active = local_time_now < context_length;

            if (implicit_rel_attn_bias) {
                // Compute bias value on the fly (See bert_preprocess_kernels.cu::buildRelativeAttentionBias)
                int relative_buckets  = 0;
                int relative_position = local_time_now - tlength;
                int num_buckets       = relative_attention_bias_stride;
                // Special logic in T5 relative attention, both encoder & decoder use this, because
                // relative_attention_bias is pre-computed once and passed around.
                num_buckets /= 2;
                relative_buckets += relative_position > 0 ? num_buckets : 0;
                relative_position = abs(relative_position);
                int  max_exact    = num_buckets / 2;
                bool is_small     = relative_position < max_exact;
                int  relative_position_if_large =
                    max_exact
                    + (int)(logf(relative_position * 1.0f / max_exact) / logf((float)max_distance / max_exact)
                            * (num_buckets - max_exact));
                relative_position_if_large = min(relative_position_if_large, num_buckets - 1);
                relative_buckets += is_small ? relative_position : relative_position_if_large;
                relative_attention_bias_ptr =
                    relative_attention_bias_ptr_fixed + (tlength - local_time_now) + relative_buckets;
            }

            // Prefetch the relative attention bias.
            float relative_attention_bias = 0.f;
            if (is_active && has_relative_attention_bias) {
                // TODO: Use a better way to convert from T to float.
                relative_attention_bias = add(relative_attention_bias, relative_attention_bias_ptr[local_time_now]);
            }

            // Compute the dot product between Q and K.
            // Note that dot will convert 8bit vec to the accumulation data type (float by default).
            float qk_ = 0.f;
#ifdef MMHA_FP8_SCALE_Q_INSTEAD_OF_K
            if constexpr (FP8_KV_CACHE) {
                qk_ = Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k_vec) * params.inv_sqrt_dh;
            }
            else
#endif  // MMHA_FP8_SCALE_Q_INSTEAD_OF_K
            {
                if constexpr (ENABLE_8BITS_CACHE) {
                    qk_ =
                        Qk_dot<T, THREADS_PER_KEY>::scale_dot(q_vec, k_vec, k_scale) * params.inv_sqrt_dh;
                }
                else {
                    qk_ = Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k_vec) * params.inv_sqrt_dh;
                }
            }

            // For multi-block mode, we need to make sure it will not be OOB.
            if (MULTI_BLOCK_FLAG && local_ti >= timesteps_per_block) {
                continue;
            }

            // Add the ALiBi bias. (ki - qi) * slope[hi].
            //
            // The padding tokens are located between the input context and the generated tokens.
            // We need to remove the correct number of padding tokens in the distance computation.
            //
            //   ti   : 0 1 2 3 4 5 6 7 8 9(tlength)
            //   token: i i i i p p p o o o where i=input, p=pad, o=output.
            // e.g. ti = 2, dist = (9 - 3) - 2 = 4.
            //
            // All the threads do the work even if it's not relevant to avoid divergence.
            qk_ += linear_bias_slope * (local_time_now - tlength) + relative_attention_bias;

            // There's one qk value per timestep.
            // Make sure only leader threads stores qk value within the bound.
            if (is_active && is_leader) {
                // Calculate the max for softmax.
                qk_max = fmaxf(qk_max, qk_);
                // Store the product to shared memory.
                qk_smem[local_ti] = qk_;
            }
        }
    }

    // Handle generation key cache with beam searching.
    // Note that it may be overlapped with the context key loop, but it won't impact the corretness.
    // Can skip in cross attention mode.
    if (HAS_BEAMS && !DO_CROSS_ATTENTION
        && (!MULTI_BLOCK_FLAG || (c_tile + 1) * timesteps_per_block > beam0_context_length)) {
        // The input length;
        const int input_length_ = MULTI_BLOCK_FLAG ? beam0_context_length % timesteps_per_block : beam0_context_length;
        // The beginning of the generation.
        const int generation_start_ti = k_idx.x + input_length_ / K_PER_WARP * K_PER_WARP;

        // Iterate over the output tokens.
        for (int ti = generation_start_ti; ti < generation_ti_end; ti += K_PER_ITER) {
            const int time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;

            // The keys loaded from the key cache.
            K_vec_m k_vec[K_VECS_PER_THREAD];

#pragma unroll
            for (int k_vec_i = 0; k_vec_i < K_VECS_PER_THREAD; ++k_vec_i) {
                const int jj             = min(k_idx.y + k_vec_i * K_ELTS_PER_CHUNK, Dh - K_VEC_SIZE);
                const int valid_time_now = min(time_now, kv_loop_length - 1);
                int       beam_offset    = beam_indices[valid_time_now];
                const int seqIdx         = batch_idx * beam_width + beam_offset;
                // Base pointer to k cache block for beam's batch, before offsetting with indirection buffer
                Tcache* k_cache_batch = reinterpret_cast<Tcache*>(kvCacheBuffer.getKBlockPtr(seqIdx, valid_time_now));

                int inBlockIdx = kvCacheBuffer.getKVLocalIdx(valid_time_now, hi_kv, Dh, jj);
                if constexpr (ENABLE_8BITS_CACHE) {
                    float* k_scale_ptr = reinterpret_cast<float*>(kvCacheBuffer.getKScalePtr(seqIdx, valid_time_now));
                    int    inScaleIdx  = kvCacheBuffer.getKVScaleLocalIdx(valid_time_now, hi_kv);
                    load_8bits_kv_cache_vec(
                        &k_vec[k_vec_i], k_cache_batch, inBlockIdx, k_scale_ptr[inBlockIdx - inScaleIdx]);
                }
                else {
                    k_vec[k_vec_i] = (*reinterpret_cast<const K_vec_m*>(&k_cache_batch[inBlockIdx]));
                }
            }

            // Is it active?
            const bool is_active = time_now >= context_length && time_now < kv_loop_length;

            if (implicit_rel_attn_bias) {
                // Compute bias value on the fly (See bert_preprocess_kernels.cu::buildRelativeAttentionBias)
                int relative_buckets  = 0;
                int relative_position = time_now - tlength;
                int num_buckets       = relative_attention_bias_stride;
                // Special logic in T5 relative attention, both encoder & decoder use this, because
                // relative_attention_bias is pre-computed once and passed around.
                num_buckets /= 2;
                relative_buckets += relative_position > 0 ? num_buckets : 0;
                relative_position = abs(relative_position);
                int  max_exact    = num_buckets / 2;
                bool is_small     = relative_position < max_exact;
                int  relative_position_if_large =
                    max_exact
                    + (int)(logf(relative_position * 1.0f / max_exact) / logf((float)max_distance / max_exact)
                            * (num_buckets - max_exact));
                relative_position_if_large = min(relative_position_if_large, num_buckets - 1);
                relative_buckets += is_small ? relative_position : relative_position_if_large;
                relative_attention_bias_ptr =
                    relative_attention_bias_ptr_fixed + (tlength - time_now) + relative_buckets;
            }

            // Prefetch the relative attention bias.
            float relative_attention_bias = 0.f;
            if (is_active && has_relative_attention_bias) {
                // TODO: Use a better way to convert from T to float.
                relative_attention_bias = add(relative_attention_bias, relative_attention_bias_ptr[time_now]);
            }

            // Perform the dot product and normalize qk.
            //
            // WARNING: ALL THE THREADS OF A WARP MUST ENTER!!!
            // Note that dot will convert 8bit vec to the accumulation data type (float by default).
            float qk_ = 0.f;
#ifdef MMHA_FP8_SCALE_Q_INSTEAD_OF_K
            if constexpr (FP8_KV_CACHE) {
                qk_ = Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k_vec) * params.inv_sqrt_dh;
            }
            else
#endif  // MMHA_FP8_SCALE_Q_INSTEAD_OF_K
            {
                if constexpr (ENABLE_8BITS_CACHE) {
                    qk_ =
                        Qk_dot<T, THREADS_PER_KEY>::scale_dot(q_vec, k_vec, kv_scale_quant_orig_f) * params.inv_sqrt_dh;
                }
                else {
                    qk_ = Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k_vec) * params.inv_sqrt_dh;
                }
            }
            // Add the ALiBi bias. (ki - qi) * slope[hi].
            //
            // The padding tokens are located between the input context and the generated tokens.
            // We need to remove the correct number of padding tokens in the distance computation.
            //
            //   ti   : 0 1 2 3 4 5 6 7 8 9(tlength)
            //   token: i i i i p p p o o o where i=input, p=pad, o=output.
            // e.g. ti = 2, dist = (9 - 3) - 2 = 4.
            //
            // All the threads perform that step to avoid divergence.
            qk_ += linear_bias_slope * (time_now - tlength) + relative_attention_bias;

            // There's one qk value per timestep.
            // Make sure only leader threads stores qk value within the bound.
            if (is_active && is_leader) {
                // Calculate the max for softmax.
                qk_max = fmaxf(qk_max, qk_);
                // Store the product to shared memory.
                qk_smem[ti] = qk_;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Softmax.

    // Perform the final reduction to compute the max inside each warp.
    //
    // NOTE: In a group of THREADS_PER_KEY threads, the leader already has the max value for the
    // group so it's not needed to run the reduction inside the group (again).

#if __CUDA_ARCH__ >= 750 && defined(MMHA_USE_HMMA)
    // Leader threads will be in the dignonal when using HMMA.
    if (THREADS_PER_KEY <= 4) {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(unsigned(-1), qk_max, 4));
    }
    if (THREADS_PER_KEY <= 8) {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(unsigned(-1), qk_max, 9));
    }
    if (THREADS_PER_KEY <= 16) {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(unsigned(-1), qk_max, 18));
    }
#else
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= THREADS_PER_KEY; mask /= 2) {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
    }
#endif  // defined MMHA_USE_HMMA

    // Decompose the thread index into warp and lane.
    const auto warp = tidx / WARP_SIZE;
    const auto lane = tidx % WARP_SIZE;

    // The warp leader writes the max to shared memory.
    if (lane == 0) {
        red_smem[warp] = qk_max;
    }

    // Make sure the products are in shared memory.
    __syncthreads();

    // After the syncthreads, the target k position (cyclic kv cache) should also have been used by the k loop.
    // Write the K values to the global memory cache.
    //
    // NOTE: The stores are uncoalesced as we have multiple chunks of 16B spread across the memory
    // system. We designed it this way as it allows much better memory loads (and there are many
    // more loads) + the stores are really "write and forget" since we won't need the ack before
    // the end of the kernel. There's plenty of time for the transactions to complete.

    // For MQA/GQA mode, write only with the first Q head of each group per KV head.
    if (HANDLE_KV && hi == (hi_kv * qhead_per_kv) && qk_vec_idx < Dh) {
        // Trigger the stores to global memory.
        Qk_vec_k   k_vec      = *reinterpret_cast<Qk_vec_k*>(&k_smem[qk_vec_idx]);
        const auto k_idx      = QK_VEC_SIZE * tidx;
        const int  inBlockIdx = kvCacheBuffer.getKVLocalIdx(cyclic_tlength, hi_kv, Dh, k_idx);
        // The base pointer for the value in the cache buffer.
        Tcache* k_cache = reinterpret_cast<Tcache*>(kvCacheBuffer.getKBlockPtr(batch_beam_idx, cyclic_tlength));

        if constexpr (ENABLE_8BITS_CACHE) {
            float k_max = vector_abs_max(k);
#pragma unroll
            for (int mask = QK_VECS_PER_Dh_MAX / 2; mask >= 1; mask /= 2) {
                k_max = fmaxf(k_max, __shfl_xor_sync(shfl_mask(QK_VECS_PER_Dh_MAX), k_max, mask));
            }
            store_8bits_kv_cache_vec(reinterpret_cast<Tcache*>(k_cache), k, inBlockIdx, float(1 << (8 - 1)) / k_max);
            if (k_idx == 0) {
                float* k_scale_ptr = reinterpret_cast<float*>(kvCacheBuffer.getKScalePtr(batch_beam_idx, tlength));
                int    inScaleIdx  = kvCacheBuffer.getKVScaleLocalIdx(tlength, hi_kv);
                *reinterpret_cast<float*>(&k_scale_ptr[inScaleIdx]) = k_max / float(1 << (8 - 1));
            }
        }
        else {
            *reinterpret_cast<Qk_vec_m*>(&k_cache[inBlockIdx]) = vec_conversion<Qk_vec_m, Qk_vec_k>(k_vec);
        }
    }

    // The warps finalize the reduction.
    qk_max = lane < WARPS_PER_BLOCK ? red_smem[lane] : -FLT_MAX;
#pragma unroll
    for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
    }

    // Broadcast to all the threads in the warp.
    qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

    // Compute the logits and start the sum.
    float sum = 0.f;

    // Each thread will handle one float (either qk_smem/logit).
    const int logit_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : kv_loop_length;
    for (int ti = tidx; ti <= logit_loop_end; ti += THREADS_PER_BLOCK) {

        const int time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;

        // For single-block mode, we don't need the mask since it has been skipped.
        if (!MULTI_BLOCK_FLAG) {
            float logit = __expf(qk_smem[time_now] - qk_max);
            sum += logit;
            qk_smem[time_now] = logit;
        }
        else {
            // Not supported yet: multi-block mode with FP8_MHA
            if (time_now < kv_loop_length && ti != timesteps_per_block) {
                float logit = __expf(qk_smem[ti] - qk_max);
                sum += logit;
                qk_smem[ti] = logit;
            }
            else if (time_now == kv_loop_length) {
                float logit = __expf(qk_current_smem[0] - qk_max);
                sum += logit;
                qk_current_smem[0] = logit;
            }
        }
    }

    // Compute the sum.
    sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], sum);

// Normalize the logits.
#ifdef MMHA_FP8_SCALE_P_INSTEAD_OF_V
    // float logit_scale = (FP8_KV_CACHE ? kv_scale_quant_orig_f : 1.0f);
    float logit_scale = 1.0f;
#else
    float logit_scale = 1.f;
#endif  // MMHA_FP8_SCALE_P_INSTEAD_OF_V
    float inv_sum = __fdividef(logit_scale, sum + 1.e-6f);

    const int normlization_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : kv_loop_length;
    for (int ti = tidx; ti <= normlization_loop_end; ti += THREADS_PER_BLOCK) {
        const int time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;

        if (!MULTI_BLOCK_FLAG) {
            convert_from_float(&logits_smem[ti], qk_smem[ti] * inv_sum);
        }
        else {
            // no scaling factor inv_sum applied here, will apply the scaling factor after all blocks finished
            if (time_now < kv_loop_length && ti != timesteps_per_block) {
                convert_from_float(&logits_smem[ti], qk_smem[ti]);
            }
            else if (time_now == kv_loop_length) {
                convert_from_float(&logits_current_smem[0], qk_current_smem[0]);
            }
        }
    }

    // Put Values part below so we leverage __syncthreads
    // from the previous step

    const auto v_idx = chunk_index<T, V_vec_k, THREADS_PER_VALUE>(tidx);
    // The value computed by this thread.
    const auto vo = v_idx.x;
    // The hidden dimensions computed by this particular thread.
    const auto vi = v_idx.y;
    // Base pointer for the row of pointers to v cache blocks
    void** v_cache_base_row_ptr = reinterpret_cast<void**>(kvCacheBuffer.getRowPtr(KVIdxType::V_IDX, batch_beam_idx));
    // Base pointer for the row of pointers to v cache blocks for beam's batch, before offsetting with indirection
    // buffer
    void** v_cache_batch_row_ptr =
        reinterpret_cast<void**>(kvCacheBuffer.getRowPtr(KVIdxType::V_IDX, batch_idx * beam_width));

    // The number of values processed per iteration of the loop.
    constexpr unsigned V_PER_ITER{THREADS_PER_BLOCK / THREADS_PER_VALUE};
    // The number of unrolled keys per ieration.
    constexpr unsigned UNROLLED_V_PER_ITER = V_PER_ITER * V_LOOP_UNROLL;

    bool const is_valid_vi = IS_Dh_MAX || vi < Dh;

    // One group of threads computes the product(s) for the current timestep.
    V_vec_k v_bias;
    zero(v_bias);
    // if( vo == params.timestep % V_PER_ITER ) {
    if (is_valid_vi && HANDLE_KV && vo == kv_loop_length % V_PER_ITER) {
        // Trigger the loads from the V bias buffer.
        if (params.v_bias != nullptr) {
            const auto v_bias_offset = flat_index2(hi_kv, vi, Dh);
            v_bias                   = *reinterpret_cast<const V_vec_k*>(&params.v_bias[v_bias_offset]);
        }

        if (DO_CROSS_ATTENTION) {
            *reinterpret_cast<V_vec_k*>(&bias_smem[vi]) = v_bias;
        }
    }

    // From previous, before values, step
    // Also make sure the logits are in shared memory.
    __syncthreads();

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Value cache loops.

#ifdef MMHA_USE_FP32_ACCUM_FOR_OUT
    using V_vec_accum = typename V_vec_accum_fp32_<V_vec_k>::Type;
#else
    using V_vec_accum = V_vec_k;
#endif
    // The partial outputs computed by each thread.
    V_vec_accum out;
    zero(out);

    // Loop over the timesteps to compute the partial outputs.
    if (is_valid_vi) {
        // Handle only context value cache with beam searching.
        // Handle both context and generation value cache without beam searching.
        // Explicit batching of LDGs (by V_LOOP_UNROLL) as it doesn't depend on indirection tables.
        // Take all previous cache as context when we have no beam searching in order to batch as many LDGs as possible.
        const int context_length =
            DO_CROSS_ATTENTION ? kv_loop_length : (HAS_BEAMS ? beam0_context_length : kv_loop_length);
        int context_v_loop_end    = MULTI_BLOCK_FLAG ? timesteps_per_block : context_length;
        int generation_v_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : kv_loop_length;
        for (int ti = vo; ti < context_v_loop_end; ti += UNROLLED_V_PER_ITER) {
            V_vec_m v_vec_cache[V_LOOP_UNROLL];
#pragma unroll
            for (int v_loop = 0; v_loop < V_LOOP_UNROLL; v_loop++) {
                // Fetch offset based on cache_indir when beam sampling
                int time_idx = ti + v_loop * V_PER_ITER + (MULTI_BLOCK_FLAG ? c_tile_times_timesteps_per_block : 0);
                time_idx     = min(time_idx, kv_loop_length - 1);
                int rowIdx   = batch_idx * beam_width;

                const int inBlockIdx = kvCacheBuffer.getKVLocalIdx(time_idx, hi_kv, Dh, vi);
                // The base pointer for the value in the cache buffer.
                Tcache* v_cache_batch = reinterpret_cast<Tcache*>(kvCacheBuffer.getVBlockPtr(rowIdx, time_idx));

                v_vec_cache[v_loop] = *reinterpret_cast<const V_vec_m*>(&v_cache_batch[inBlockIdx]);
            }

#pragma unroll
            for (int v_loop = 0; v_loop < V_LOOP_UNROLL; v_loop++) {
                float v_scale = 1.f;
                V_vec_m v_vec = reinterpret_cast<V_vec_m*>(&v_vec_cache[v_loop])[0];
                int rowIdx   = batch_idx * beam_width;

                int local_time_idx = ti + v_loop * V_PER_ITER;
                int time_idx       = local_time_idx + (MULTI_BLOCK_FLAG ? c_tile_times_timesteps_per_block : 0);
                const bool is_mask =
                    (MULTI_BLOCK_FLAG && local_time_idx >= timesteps_per_block) || (time_idx >= context_length);

                if constexpr (ENABLE_8BITS_CACHE) {
                    int    local_time  = min(time_idx, tlength - 1);
                    float* v_scale_ptr = reinterpret_cast<float*>(kvCacheBuffer.getVScalePtr(rowIdx, local_time));
                    int    inScaleIdx  = kvCacheBuffer.getKVScaleLocalIdx(local_time, hi_kv);
                    v_scale            = v_scale_ptr[inScaleIdx];
                }

                // Load the logits from shared memory.
                // Note that fma will convert 8bit vec to the accumulation data type (float by default).
                Logit_value_fma<Tk, V_vec_accum, V_vec_m, INT8_KV_CACHE, FP8_KV_CACHE>(
                    out, reinterpret_cast<Tk*>(logits_smem + local_time_idx), v_vec, v_scale, is_mask);
            }
        }

        // Handle generation value cache with beam searching.
        if (HAS_BEAMS && !DO_CROSS_ATTENTION) {
            const auto generation_start_ti =
                MULTI_BLOCK_FLAG ? vo : (vo + (beam0_context_length / V_PER_ITER) * V_PER_ITER);
            // Only the last few blocks need to handle the generation value cache.
            if (!MULTI_BLOCK_FLAG || (c_tile + 1) * timesteps_per_block > beam0_context_length) {
                for (int ti = generation_start_ti; ti < generation_v_loop_end; ti += V_PER_ITER) {
                    // Fetch offset based on cache_indir when beam sampling
                    int time_idx       = ti + (MULTI_BLOCK_FLAG ? c_tile_times_timesteps_per_block : 0);
                    int local_time_idx = ti;
                    if (time_idx < beam0_context_length || (MULTI_BLOCK_FLAG && time_idx >= kv_loop_length)) {
                        continue;
                    }
                    int rowIdx = batch_idx * beam_width + beam_indices[time_idx];

                    const int inBlockIdx = kvCacheBuffer.getKVLocalIdx(time_idx, hi_kv, Dh, vi);
                    // The base pointer for the value in the cache buffer.
                    Tcache* v_cache_batch = reinterpret_cast<Tcache*>(kvCacheBuffer.getVBlockPtr(rowIdx, time_idx));
                    // V_vec_m v_vec;
                    V_vec_m    v_vec = reinterpret_cast<const V_vec_m*>(&v_cache_batch[inBlockIdx])[0];
                    float v_scale = 1.f;
                    if (ENABLE_8BITS_CACHE) {
                        float* v_scale_ptr = reinterpret_cast<float*>(kvCacheBuffer.getVScalePtr(rowIdx, time_idx));
                        int    inScaleIdx  = kvCacheBuffer.getKVScaleLocalIdx(time_idx, hi_kv);
                        v_scale =  v_scale_ptr[inScaleIdx];
                    }

                    // Load the logits from shared memory.
                    // Note that fma will convert 8bit vec to the accumulation data type (float by default).
                    Logit_value_fma<Tk, V_vec_accum, V_vec_m, INT8_KV_CACHE, FP8_KV_CACHE>(
                        out, reinterpret_cast<Tk*>(logits_smem + local_time_idx), v_vec, v_scale, false);
                }
            }
        }
    }

    // Make sure we can overwrite the v cache if using cyclic kv cache.
    __syncthreads();

    // Get the c_tile_id that handles the current timestep.
    const int ctile_idx = tlength / timesteps_per_block;

    // One group of threads computes the product(s) for the current timestep.
    if (vo == kv_loop_length % V_PER_ITER && is_valid_vi && (!MULTI_BLOCK_FLAG || (c_tile == ctile_idx))) {
        const int tokenIdx   = cyclic_tlength;
        const int inBlockIdx = kvCacheBuffer.getKVLocalIdx(tokenIdx, hi_kv, Dh, vi);
        // The base pointer for the value in the cache buffer.
        Tcache* v_cache_base = reinterpret_cast<Tcache*>(kvCacheBuffer.getBlockPtr(v_cache_base_row_ptr, tokenIdx));

        V_vec_k v;
        if (DO_CROSS_ATTENTION) {
            v = vec_conversion<V_vec_k, V_vec_k>(*reinterpret_cast<const V_vec_k*>(&v_cache_base[inBlockIdx]));
        }
        else {
            // Trigger the loads from the V buffer.
            // The stride between tokens. We may be able to always use params.stride.
            uint32_t v_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads_kv * Dh);
            // The offset.
            const auto v_offset = flat_index_strided3(batch_beam_idx, hi_kv, vi, v_stride, Dh);

            if (load_qkv_quant) {
                using Packed_Int8_t  = typename packed_type<int8_t, num_elems<V_vec_k>::value>::type;
                using Packed_Float_t = typename packed_type<float, num_elems<V_vec_k>::value>::type;
                const auto v_scaling = params.qkv_scale_quant_orig[2];
                const auto v_quant =
                    *reinterpret_cast<const Packed_Int8_t*>(&reinterpret_cast<const int8_t*>(params.v)[v_offset]);

                convert_from_float(&v, mul<Packed_Float_t, float>(v_scaling, float_from_int8(v_quant)));
            }
            else {
                v = *reinterpret_cast<const V_vec_k*>(&params.v[v_offset]);
            }
        }

        if (HANDLE_KV) {
            // Compute the V values with bias.
            v = add(v, v_bias);

            if (do_ia3) {
                v = mul<V_vec_k, V_vec_k, V_vec_k>(
                    v, *reinterpret_cast<const V_vec_k*>(&params.ia3_value_weights[flat_index2(ia3_ti_hi, vi, Dh)]));
            }
        }

        // Store the values with bias back to global memory in the cache for V.
        //*reinterpret_cast<V_vec_k*>(&v_cache[params.timestep*Dh]) = v;
        // For MQA/GQA mode, write only with the first Q head of each group per KV head.
        if (hi == (hi_kv * qhead_per_kv)) {
            if constexpr (ENABLE_8BITS_CACHE) {
                float v_max = vector_abs_max(v);
#pragma unroll
                for (int mask = THREADS_PER_VALUE / 2; mask >= 1; mask /= 2) {
                    v_max =
                        fmaxf(v_max,
                              __shfl_xor_sync(
                                  shfl_mask_and_index(THREADS_PER_VALUE, vo % (32 / THREADS_PER_VALUE)), v_max, mask));
                }
                store_8bits_kv_cache_vec(v_cache_base, v, inBlockIdx, float(1 << (8 - 1)) / v_max);
                if (vi == 0) {
                    float* v_scale_ptr = reinterpret_cast<float*>(kvCacheBuffer.getVScalePtr(batch_beam_idx, tokenIdx));
                    int    inScaleIdx  = kvCacheBuffer.getKVScaleLocalIdx(tokenIdx, hi_kv);
                    *reinterpret_cast<float*>(&v_scale_ptr[inScaleIdx]) = v_max / float(1 << (8 - 1));
                }
            }

            else {
                *reinterpret_cast<V_vec_k*>(&v_cache_base[inBlockIdx]) = v;
            }
        }

        // Initialize the output value with the current timestep.
#if defined(MMHA_USE_FP32_ACCUM_FOR_LOGITS)
        // out = fma(logits_smem[params.timestep], cast_to_float(v), out);
        if (!MULTI_BLOCK_FLAG) {
            out = fma(logits_smem[kv_loop_length], cast_to_float(v), out);
        }
        else {
            out = fma(logits_current_smem[0], cast_to_float(v), out);
        }
#else   // MMHA_USE_FP32_ACCUM_FOR_LOGITS
        // out = fma(logits_smem[params.timestep], v, out);
        if (!MULTI_BLOCK_FLAG) {
            out = fma(logits_smem[kv_loop_length], v, out);
        }
        else {  // MULTI_BLOCK_FLAG // Not supported yet: multi-block mode with FP8_MHA
            out = fma(logits_current_smem[0], v, out);
        }
#endif  // MMHA_USE_FP32_ACCUM_FOR_LOGITS
    }
    // Make sure we can start writing to shared memory.
    __syncthreads();

    // Run the final reduction amongst the different groups computing different partial outputs.
#pragma unroll
    for (int active_groups = V_PER_ITER; active_groups >= 2; active_groups /= 2) {

        // The midpoint in the number of active groups.
        int midpoint = active_groups / 2;

        // The upper part of active threads store to shared memory.
        if (vo >= midpoint && vo < active_groups && (Dh == Dh_MAX || vi < Dh)) {
#ifdef MMHA_USE_FP32_ACCUM_FOR_OUT
            convert_from_float(reinterpret_cast<V_vec_k*>(&out_smem[(vo - midpoint) * Dh + vi]), out);
#else
            *reinterpret_cast<V_vec_k*>(&out_smem[(vo - midpoint) * Dh + vi]) = out;
#endif
        }
        __syncthreads();

        // The bottom warps update their values.
        if (vo < midpoint && (Dh == Dh_MAX || vi < Dh)) {
            out = add(*reinterpret_cast<const V_vec_k*>(&out_smem[vo * Dh + vi]), out);
        }
        __syncthreads();
    }

    const auto bhi              = flat_index2(batch_beam_idx, hi, num_heads);
    const auto bhi_seq_len_tile = bhi * params.seq_len_tile;
    // Output the final values.
    if (vo == 0 && (Dh == Dh_MAX || vi < Dh)) {
        const auto bhvi = flat_index2(bhi, vi, Dh);
#ifdef MMHA_USE_FP32_ACCUM_FOR_OUT
        if (write_attention_quant) {
            using Packed_Int8_t = typename packed_type<int8_t, num_elems<V_vec_accum>::value>::type;
            out                 = mul<V_vec_accum, float>(*params.attention_out_scale_orig_quant, out);
            *reinterpret_cast<Packed_Int8_t*>(&(reinterpret_cast<int8_t*>(params.out)[bhvi])) = cast_to_int8(out);
        }
        else {
            if (!MULTI_BLOCK_FLAG) {
                // This makes sure we have coalesced memory access.
                V_vec_k final_out;
                convert_from_float(&final_out, out);
                *reinterpret_cast<V_vec_k*>(&params.out[bhvi]) = final_out;
            }
            else {
                // for write partial output to partial_out
                int partial_out_offset = c_tile * params.batch_size * num_heads * params.hidden_size_per_head;
                // for write partial statistics to partial_max and partial_sum
                int partial_stats_offset = bhi_seq_len_tile + c_tile;

                // This makes sure we have coalesced memory access.
                V_vec_k partial_out;
                convert_from_float(&partial_out, out);
                *reinterpret_cast<V_vec_k*>(&params.partial_out[partial_out_offset + bhvi]) = partial_out;
                convert_from_float(reinterpret_cast<float*>(&params.partial_max[partial_stats_offset]), qk_max);
                convert_from_float(reinterpret_cast<float*>(&params.partial_sum[partial_stats_offset]), sum);
            }
        }
#else   // MMHA_USE_FP32_ACCUM_FOR_OUT
        *reinterpret_cast<V_vec_accum*>(&params.out[bhvi]) = out;
#endif  // MMHA_USE_FP32_ACCUM_FOR_OUT
    }

#ifdef ENABLE_MULTI_BLOCK_OPTION
    if (MULTI_BLOCK_FLAG) {

        cuda::atomic_ref<int, cuda::thread_scope_device> count_ref{params.block_counter[bhi]};
        bool                                             last_block{false};
        if (tidx == 0) {
            if (count_ref.fetch_add(1, cuda::memory_order_acq_rel) == (gridDim.z - 1)) {
                last_block = true;
            }
        }

        ////////////////////
        ////////////////////
        // Make sure every threadblock finishes the previous computation, and enter the last threadblock in the
        // following (for each B and H) Do the final computation in the last threadblock Final reduction computation
        // by combining all the partial max/sum and outputs
        ////////////////////
        ////////////////////
        if (__syncthreads_or(last_block)) {

            ////////////////////
            // Find the global max from all partial max -> use CUB BlockReduce
            ////////////////////

            float final_max          = -FLT_MAX;
            float thread_partial_max = -FLT_MAX;
            thread_partial_max       = params.partial_max[bhi_seq_len_tile + min(tidx, gridDim.x - 1)];

            // Make sure we can start writing to shared memory.
            __syncthreads();

            // Specialize BlockReduce for a 1D block of THREADS_PER_BLOCK threads of type int
            typedef cub::BlockReduce<float, THREADS_PER_BLOCK> BlockReduce;
            // Allocate shared memory for BlockReduce
            __shared__ typename BlockReduce::TempStorage temp_storage;
            // Obtain a segment of consecutive items that are blocked across threads (final_max from above)
            // Compute the block-wide max for thread0
            final_max = BlockReduce(temp_storage).Reduce(thread_partial_max, cub::Max(), gridDim.z);

            __shared__ float final_max_smem;
            if (tidx == 0) {
                final_max_smem = final_max;
            }
            __syncthreads();

            // Finish the final_max computation
            final_max = final_max_smem;

            ////////////////////
            // Reduction for global sum over all partial sum (scaled by the exponential term from global max) -> use
            // gridDim.z threads
            ////////////////////

            float final_sum = 0.f;
            if (tidx < gridDim.z) {
                thread_partial_max            = params.partial_max[bhi_seq_len_tile + tidx];
                const auto thread_partial_sum = params.partial_sum[bhi_seq_len_tile + tidx];
                final_sum += __expf(thread_partial_max - final_max) * thread_partial_sum;
            }

            // Compute the final_sum.
            final_sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], final_sum);

            ////////////////////
            // Reduction for final output (scaled by the exponential term from global max) -> use THREADS_PER_VALUE
            // * gridDim.z threads
            ////////////////////

            // Shared memory to store partial outputs for each oi. -> size: gridDim.z * Dh * 4 Bytes. Reuse qk_smem.
            T* out_oi_smem = reinterpret_cast<T*>(smem_);

            const auto o_idx = chunk_index<T, V_vec_k, THREADS_PER_VALUE>(tidx);
            // The partial output region this thread takes care of
            const auto oo = o_idx.x;
            // The hidden dimensions computed by this particular thread. (refer to vi)
            const auto oi = o_idx.y;

            // Within the bound.
            const bool within_bound = oo < gridDim.z;

            // Load partial output
            int thread_partial_out_offset = oo * params.batch_size * num_heads * params.hidden_size_per_head;
            // Load partial max (different to thread_partial_max since the threadIdx rule changes here)
            float thread_partial_max_for_out = within_bound ? params.partial_max[bhi_seq_len_tile + oo] : final_max;

            // Load the partial outputs.
            V_vec_k zero_k;
            zero(zero_k);
            V_vec_k thread_partial_out =
                within_bound ?
                    *reinterpret_cast<const V_vec_k*>(&params.partial_out[thread_partial_out_offset + bhi * Dh + oi]) :
                    zero_k;

            Tk factor_compute;
            convert_from_float(&factor_compute, __expf(thread_partial_max_for_out - final_max));
            thread_partial_out = mul<V_vec_k, Tk, V_vec_k>(factor_compute, thread_partial_out);

            // Make sure we can start writing to shared memory.
            __syncthreads();

            // The reduction iteration should start with a number which is a power of 2
            const auto reduction_iteration = static_cast<int>(cuda::std::bit_ceil(gridDim.z));

            // Run the final reduction amongst the different groups computing different partial outputs.
#pragma unroll
            for (int active_groups = reduction_iteration; active_groups >= 2; active_groups /= 2) {

                // The midpoint in the number of active groups.
                int midpoint = active_groups / 2;

                // The upper part of active threads store to shared memory.
                if (oo >= midpoint && oo < active_groups && (Dh == Dh_MAX || oi < Dh)) {
                    *reinterpret_cast<V_vec_k*>(&out_oi_smem[(oo - midpoint) * Dh + oi]) = thread_partial_out;
                }
                __syncthreads();

                // The bottom warps update their values.
                if (oo < midpoint && (Dh == Dh_MAX || oi < Dh)) {
                    thread_partial_out =
                        add(thread_partial_out, *reinterpret_cast<const V_vec_k*>(&out_oi_smem[oo * Dh + oi]));
                }
                __syncthreads();
            }

            ////////////////////
            // Final output O * inv_sum
            ////////////////////

            if (oo == 0 && (Dh == Dh_MAX || oi < Dh)) {
                const auto inv_sum = __fdividef(1.f, final_sum + 1.e-6f);

                Tk inv_sum_compute;
                convert_from_float(&inv_sum_compute, inv_sum);

                thread_partial_out = mul<V_vec_k, Tk, V_vec_k>(inv_sum_compute, thread_partial_out);

                *reinterpret_cast<V_vec_k*>(&params.out[bhi * Dh + oi]) = thread_partial_out;
            }

            // Reset qk_current_smem and block_counter for the next timestep
            if (tidx == 0) {
                params.block_counter[bhi] = 0;
            }
        }
    }
#endif  // ENABLE_MULTI_BLOCK_OPTION
}

}  // namespace fastertransformer
