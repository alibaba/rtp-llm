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

#include <stdint.h>
#if USING_CUDA
#include "src/fastertransformer/cuda/cuda_type_utils.cuh"
#include "src/fastertransformer/cuda/cuda_type_utils.cuh"
#include "src/fastertransformer/cuda/cuda_utils.h"
#include "src/fastertransformer/cuda/cuda_utils.h"
#endif

#if USING_ROCM
#include "src/fastertransformer/rocm/hip_type_utils.cuh"
#include "src/fastertransformer/rocm/hip_utils.h"
using namespace fastertransformer::rocm;
#endif

#ifdef ENABLE_BF16
using fastertransformer::bf16hfma2;
using fastertransformer::bf162bf162;
using fastertransformer::bf1622float2;
using fastertransformer::bf16hmul2;
using fastertransformer::bf16hmul;
using fastertransformer::bf16hadd2;
using fastertransformer::float22bf162;
#endif

namespace fastertransformer {

#ifdef ENABLE_FP8
using fp8_2_t = __nv_fp8x2_e4m3;
using fp8_4_t = __nv_fp8x4_e4m3;

struct __align__(8) fp8_8_t {
    __nv_fp8x2_e4m3 x;
    __nv_fp8x2_e4m3 y;
    __nv_fp8x2_e4m3 z;
    __nv_fp8x2_e4m3 w;
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 half2_to_float2(uint32_t v) {
#if USING_ROCM
    return __half22float2(*reinterpret_cast<__half2_raw*>(&v));
#else
    uint16_t lo, hi;
    asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(v));
    return make_float2(half_to_float(lo), half_to_float(hi));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint16_t float_to_half(float f) {
    __half_raw tmp{static_cast<_Float16>(f)};
    return tmp.x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float half_to_float(uint16_t h) {
#if USING_ROCM
    return __half2float(*reinterpret_cast<__half_raw*>(&h));
#else
    float f;
    asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
    return f;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t float2_to_half2(float2 f) {
    union {
        uint32_t u32;
        _Float16_2 data;
    } tmp;
    tmp.data = _Float16_2{static_cast<_Float16>(f.x), static_cast<_Float16>(f.y)};
    return tmp.u32;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t h0_h0(uint16_t a) {
#if USING_ROCM
    __half2 out = __half2half2(*reinterpret_cast<__half_raw*>(&a));
    return *reinterpret_cast<uint32_t*>(&(out.data));
#else
    uint32_t b;
    asm volatile("mov.b32 %0, {%1, %1};" : "=r"(b) : "h"(a));
    return b;
#endif
}


#include "_vector_abs_max.h"
#include "_convert_from_float.h"
#include "_convert_to_float.h"
#include "_cast_to_int8.h"
#include "_add.h"
#include "_mul.h"
#include "_fma.h"
#include "_sum_dot_zero.h"
#include "_logn_attention.h"
#include "_convert_from_fp8.h"
#include "_convert_to_fp8.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Vec_k, typename T, typename T_scale>
inline __device__ void load_8bits_kv_cache_vec(Vec_k* vec, const T* pointer, int idx, T_scale scale) {
    ;  // Not used.
}

template<typename Vec_k, typename T, typename T_scale>
inline __device__ void store_8bits_kv_cache_vec(T* pointer, const Vec_k& vec, int idx, T_scale scale) {
    ;  // Not used.
}


////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Vec_k>
inline __device__ void load_8bits_kv_cache_vec(Vec_k* vec, const int8_t* pointer, int idx, float scale) {
    using Packed_8bits_t = typename packed_type<int8_t, num_elems<Vec_k>::value>::type;
    using Packed_Float_t = typename packed_type<float, num_elems<Vec_k>::value>::type;
    const auto quant     = *reinterpret_cast<const Packed_8bits_t*>(&pointer[idx]);

    // FIXME:
    convert_from_float(vec, mul<Packed_Float_t>(scale, float_from_int8(quant)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_FP8
template<typename Vec_k>
inline __device__ void load_8bits_kv_cache_vec(Vec_k* vec, const __nv_fp8_e4m3* pointer, int idx) {
    using Packed_8bits_t = typename packed_type<__nv_fp8_e4m3, num_elems<Vec_k>::value>::type;
    const auto quant     = *reinterpret_cast<const Packed_8bits_t*>(&pointer[idx]);
    convert_from_fp8(vec, quant);
}

template<typename Vec_k, typename T_scale>
inline __device__ void load_8bits_kv_cache_vec(Vec_k* vec, const __nv_fp8_e4m3* pointer, int idx, T_scale scale) {
    load_8bits_kv_cache_vec(vec, pointer, idx);
    vec[0] = mul<Vec_k>(scale, vec[0]);
}
#endif  // ENABLE_FP8

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Vec_k>
inline __device__ void store_8bits_kv_cache_vec(int8_t* pointer, const Vec_k& vec, int idx, float scale) {
    using Packed_8bits_t     = typename packed_type<int8_t, num_elems<Vec_k>::value>::type;
    using Packed_Float_t     = typename packed_type<float, num_elems<Vec_k>::value>::type;
    Packed_8bits_t out_quant = cast_to_int8(mul<Packed_Float_t>(scale, convert_to_float(vec)));

    *reinterpret_cast<Packed_8bits_t*>(&pointer[idx]) = out_quant;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_FP8
template<typename Vec_k, typename T_scale>
inline __device__ void store_8bits_kv_cache_vec(__nv_fp8_e4m3* pointer, const Vec_k& vec, int idx, T_scale scale) {
    using Packed_8bits_t = typename packed_type<__nv_fp8_e4m3, num_elems<Vec_k>::value>::type;
    Packed_8bits_t out_quant;
    convert_to_fp8(&out_quant, mul<Vec_k>(scale, vec));

    *reinterpret_cast<Packed_8bits_t*>(&pointer[idx]) = out_quant;
}
#endif  // ENABLE_FP8

template<typename Vec_in, typename Vec_out, typename T_cache, typename T_scale>
inline __device__ void convert_from_8bit_kv_cache(Vec_out* vec_o, const Vec_in& vec_i, T_scale scale) {
    if constexpr (std::is_same<T_cache, int8_t>::value) {
        using Packed_Float_t = typename packed_type<float, num_elems<Vec_out>::value>::type;
        convert_from_float(vec_o, mul<Packed_Float_t>(scale, float_from_int8(vec_i)));
    }
#ifdef ENABLE_FP8
    else if constexpr (std::is_same<T_cache, __nv_fp8_e4m3>::value) {
        convert_from_fp8(vec_o, vec_i);
        vec_o[0] = mul<Vec_out>(scale, vec_o[0]);
    }
#endif  // ENABLE_FP8
    else {
        ;  // not supported.
    }
}

template<typename Vec_in, typename Vec_out, typename T_cache>
inline __device__ void convert_from_8bit_kv_cache(Vec_out* vec_o, const Vec_in& vec_i) {
    if constexpr (std::is_same<T_cache, int8_t>::value) {
        using Packed_Float_t = typename packed_type<float, num_elems<Vec_out>::value>::type;
        convert_from_float(vec_o, float_from_int8(vec_i));
    }
#ifdef ENABLE_FP8
    else if constexpr (std::is_same<T_cache, __nv_fp8_e4m3>::value) {
        convert_from_fp8(vec_o, vec_i);
    }
#endif  // ENABLE_FP8
    else {
        ;  // not supported.
    }
}


#if 1
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, bool INT8_KV_CACHE>
struct kv_cache_type_t {
    using Type = T;
};

template<typename T>
struct kv_cache_type_t<T, true> {
    using Type = int8_t;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, typename T_cache>
struct kv_cache_scale_type_t {
    using Type = float;
};

#ifdef ENABLE_FP8
template<>
struct kv_cache_scale_type_t<half, __nv_fp8_e4m3> {
    using Type = uint16_t;
};

template<>
struct kv_cache_scale_type_t<uint16_t, __nv_fp8_e4m3> {
    using Type = uint16_t;
};

template<>
struct kv_cache_scale_type_t<__nv_bfloat16, __nv_fp8_e4m3> {
    using Type = __nv_bfloat16;
};
#endif  // ENALBE_FP8
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Vec_T, typename T>
__device__ __inline__ void vec_from_smem_transpose(Vec_T& vec, T* smem, int transpose_idx, int smem_pitch);

template<>
__device__ __inline__ void vec_from_smem_transpose(float& vec, float* smem, int transpose_idx, int smem_pitch) {
    return;
}

template<>
__device__ __inline__ void vec_from_smem_transpose(uint32_t& vec, uint16_t* smem, int transpose_idx, int smem_pitch) {
    union {
        uint32_t u32;
        uint16_t u16[2];
    } tmp;

    tmp.u16[0] = smem[transpose_idx];
    tmp.u16[1] = smem[smem_pitch + transpose_idx];

    vec = tmp.u32;
}

template<>
__device__ __inline__ void vec_from_smem_transpose(half2& vec, half* smem, int transpose_idx, int smem_pitch) {
    return vec_from_smem_transpose(
        *reinterpret_cast<uint32_t*>(&vec), reinterpret_cast<uint16_t*>(smem), transpose_idx, smem_pitch);
}

template<>
__device__ __inline__ void vec_from_smem_transpose(uint2& vec, uint16_t* smem, int transpose_idx, int smem_pitch) {
    union {
        uint32_t u32;
        uint16_t u16[2];
    } tmp_1, tmp_2;

    tmp_1.u32 = *reinterpret_cast<uint32_t*>(&smem[transpose_idx]);
    tmp_2.u32 = *reinterpret_cast<uint32_t*>(&smem[smem_pitch + transpose_idx]);

    union {
        uint2    u32x2;
        uint16_t u16[4];
    } tmp_3;

    tmp_3.u16[0] = tmp_1.u16[0];
    tmp_3.u16[1] = tmp_2.u16[0];
    tmp_3.u16[2] = tmp_1.u16[1];
    tmp_3.u16[3] = tmp_2.u16[1];

    vec = tmp_3.u32x2;
}

template<>
__device__ __inline__ void vec_from_smem_transpose(uint4& vec, uint16_t* smem, int transpose_idx, int smem_pitch) {
    union {
        uint64_t u64;
        uint16_t u16[4];
    } tmp_1, tmp_2;

    tmp_1.u64 = *reinterpret_cast<uint64_t*>(&smem[transpose_idx]);
    tmp_2.u64 = *reinterpret_cast<uint64_t*>(&smem[smem_pitch + transpose_idx]);

    union {
        uint4    u32x4;
        uint16_t u16[8];
    } tmp_3;

    tmp_3.u16[0] = tmp_1.u16[0];
    tmp_3.u16[1] = tmp_2.u16[0];
    tmp_3.u16[2] = tmp_1.u16[1];
    tmp_3.u16[3] = tmp_2.u16[1];
    tmp_3.u16[4] = tmp_1.u16[2];
    tmp_3.u16[5] = tmp_2.u16[2];
    tmp_3.u16[6] = tmp_1.u16[3];
    tmp_3.u16[7] = tmp_2.u16[3];

    vec = tmp_3.u32x4;
}

#ifdef ENABLE_BF16
template<>
__device__ __inline__ void
vec_from_smem_transpose(bf16_4_t& vec, __nv_bfloat16* smem, int transpose_idx, int smem_pitch) {
    union {
        uint32_t      u32;
        __nv_bfloat16 bf16[2];
    } tmp_1, tmp_2;

    tmp_1.u32 = *reinterpret_cast<uint32_t*>(&smem[transpose_idx]);
    tmp_2.u32 = *reinterpret_cast<uint32_t*>(&smem[smem_pitch + transpose_idx]);

    vec.x = __nv_bfloat162{tmp_1.bf16[0], tmp_2.bf16[0]};
    vec.y = __nv_bfloat162{tmp_1.bf16[1], tmp_2.bf16[1]};
}

template<>
__device__ __inline__ void
vec_from_smem_transpose(bf16_8_t& vec, __nv_bfloat16* smem, int transpose_idx, int smem_pitch) {
    union {
        uint64_t      u64;
        __nv_bfloat16 bf16[4];
    } tmp_1, tmp_2;

    tmp_1.u64 = *reinterpret_cast<uint64_t*>(&smem[transpose_idx]);
    tmp_2.u64 = *reinterpret_cast<uint64_t*>(&smem[smem_pitch + transpose_idx]);

    vec.x = __nv_bfloat162{tmp_1.bf16[0], tmp_2.bf16[0]};
    vec.y = __nv_bfloat162{tmp_1.bf16[1], tmp_2.bf16[1]};
    vec.z = __nv_bfloat162{tmp_1.bf16[2], tmp_2.bf16[2]};
    vec.w = __nv_bfloat162{tmp_1.bf16[3], tmp_2.bf16[3]};
}
#endif  // ENABLE_BF16

template<>
__device__ __inline__ void vec_from_smem_transpose(float4& vec, float* smem, int transpose_idx, int smem_pitch) {
    vec.x = smem[transpose_idx];
    vec.z = smem[transpose_idx + 1];
    vec.y = smem[smem_pitch + transpose_idx];
    vec.w = smem[smem_pitch + transpose_idx + 1];
}

template<>
__device__ __inline__ void vec_from_smem_transpose(uint32_t& vec, half* smem, int transpose_idx, int smem_pitch) {
    union {
        uint32_t u32;
        half     u16[2];
    } tmp;

    tmp.u16[0] = smem[transpose_idx];
    tmp.u16[1] = smem[smem_pitch + transpose_idx];

    vec = tmp.u32;
}

#ifdef ENABLE_BF16
template<>
__device__ __inline__ void
vec_from_smem_transpose(__nv_bfloat162& vec, __nv_bfloat16* smem, int transpose_idx, int smem_pitch) {
    vec.x = smem[transpose_idx];
    vec.y = smem[smem_pitch + transpose_idx];
}
#endif

template<>
__device__ __inline__ void vec_from_smem_transpose(float2& vec, float* smem, int transpose_idx, int smem_pitch) {
    vec.x = smem[transpose_idx];
    vec.y = smem[smem_pitch + transpose_idx];
}

template<typename Vec_T, typename T>
__device__ __inline__ void write_smem_transpose(const Vec_T& vec, T* smem, int transpose_idx, int smem_pitch);

template<>
__device__ __inline__ void write_smem_transpose(const float& vec, float* smem, int transpose_idx, int smem_pitch) {
    return;
}

#ifdef ENABLE_BF16
template<>
__device__ __inline__ void
write_smem_transpose(const bf16_4_t& vec, __nv_bfloat16* smem, int transpose_idx, int smem_pitch) {
    smem[transpose_idx]                  = vec.x.x;
    smem[transpose_idx + 1]              = vec.y.x;
    smem[smem_pitch + transpose_idx]     = vec.x.y;
    smem[smem_pitch + transpose_idx + 1] = vec.y.y;
}

template<>
__device__ __inline__ void
write_smem_transpose(const bf16_8_t& vec, __nv_bfloat16* smem, int transpose_idx, int smem_pitch) {
    smem[transpose_idx]     = vec.x.x;
    smem[transpose_idx + 1] = vec.y.x;
    smem[transpose_idx + 2] = vec.z.x;
    smem[transpose_idx + 3] = vec.w.x;

    smem[smem_pitch + transpose_idx]     = vec.x.y;
    smem[smem_pitch + transpose_idx + 1] = vec.y.y;
    smem[smem_pitch + transpose_idx + 2] = vec.z.y;
    smem[smem_pitch + transpose_idx + 3] = vec.w.y;
}
#endif

#ifdef ENABLE_FP8
template<>
__device__ __inline__ void
vec_from_smem_transpose(float4& vec, __nv_fp8_e4m3* smem, int transpose_idx, int smem_pitch) {
    // TODO
    printf("[ERROR] still no have implementation for vec_from_smem_transpose under __nv_fp8_e4m3 \n");
}
#endif  // ENABLE_FP8

template<>
__device__ __inline__ void write_smem_transpose(const uint4& vec, uint16_t* smem, int transpose_idx, int smem_pitch) {
    union {
        uint64_t u64;
        uint16_t u16[4];
    } tmp_1, tmp_2;

    union {
        uint4    u32x4;
        uint16_t u16[8];
    } tmp_3;

    tmp_3.u32x4  = vec;
    tmp_1.u16[0] = tmp_3.u16[0];
    tmp_2.u16[0] = tmp_3.u16[1];
    tmp_1.u16[1] = tmp_3.u16[2];
    tmp_2.u16[1] = tmp_3.u16[3];
    tmp_1.u16[2] = tmp_3.u16[4];
    tmp_2.u16[2] = tmp_3.u16[5];
    tmp_1.u16[3] = tmp_3.u16[6];
    tmp_2.u16[3] = tmp_3.u16[7];

    *reinterpret_cast<uint64_t*>(&smem[transpose_idx])              = tmp_1.u64;
    *reinterpret_cast<uint64_t*>(&smem[smem_pitch + transpose_idx]) = tmp_2.u64;
}

template<>
__device__ __inline__ void write_smem_transpose(const uint2& vec, uint16_t* smem, int transpose_idx, int smem_pitch) {
    union {
        uint32_t u32;
        uint16_t u16[2];
    } tmp_1, tmp_2;

    union {
        uint2    u32x2;
        uint16_t u16[4];
    } tmp_3;

    tmp_3.u32x2  = vec;
    tmp_1.u16[0] = tmp_3.u16[0];
    tmp_2.u16[0] = tmp_3.u16[1];
    tmp_1.u16[1] = tmp_3.u16[2];
    tmp_2.u16[1] = tmp_3.u16[3];

    *reinterpret_cast<uint32_t*>(&smem[transpose_idx])              = tmp_1.u32;
    *reinterpret_cast<uint32_t*>(&smem[smem_pitch + transpose_idx]) = tmp_2.u32;
}

template<>
__device__ __inline__ void
write_smem_transpose(const uint32_t& vec, uint16_t* smem, int transpose_idx, int smem_pitch) {
    union {
        uint32_t u32;
        uint16_t u16[2];
    } tmp;

    tmp.u32 = vec;

    smem[transpose_idx]              = tmp.u16[0];
    smem[smem_pitch + transpose_idx] = tmp.u16[1];
}

template<>
__device__ __inline__ void write_smem_transpose(const float4& vec, float* smem, int transpose_idx, int smem_pitch) {
    smem[transpose_idx]                  = vec.x;
    smem[transpose_idx + 1]              = vec.z;
    smem[smem_pitch + transpose_idx]     = vec.y;
    smem[smem_pitch + transpose_idx + 1] = vec.w;
}

template<>
__device__ __inline__ void write_smem_transpose(const uint32_t& vec, half* smem, int transpose_idx, int smem_pitch) {
    union {
        uint32_t u32;
        half     u16[2];
    } tmp;

    tmp.u32                          = vec;
    smem[transpose_idx]              = tmp.u16[0];
    smem[smem_pitch + transpose_idx] = tmp.u16[1];
}

template<>
__device__ __inline__ void write_smem_transpose(const half2& vec, half* smem, int transpose_idx, int smem_pitch) {
    return write_smem_transpose(*reinterpret_cast<const uint32_t*>(&vec), smem, transpose_idx, smem_pitch);
}

#ifdef ENABLE_BF16
template<>
__device__ __inline__ void
write_smem_transpose(const __nv_bfloat162& vec, __nv_bfloat16* smem, int transpose_idx, int smem_pitch) {
    smem[transpose_idx]              = vec.x;
    smem[smem_pitch + transpose_idx] = vec.y;
}
#endif

template<>
__device__ __inline__ void write_smem_transpose(const float2& vec, float* smem, int transpose_idx, int smem_pitch) {
    smem[transpose_idx]              = vec.x;
    smem[smem_pitch + transpose_idx] = vec.y;
}

#ifdef ENABLE_FP8
template<>
__device__ __inline__ void
write_smem_transpose(const float4& vec, __nv_fp8_e4m3* smem, int transpose_idx, int smem_pitch) {
    printf("[ERROR] still no have implementation for vec_from_smem_transpose under __nv_fp8_e4m3 \n");
}
#endif  // ENABLE_FP8

// For an explanation of next_power_of_two, see the following references:
// https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
// https://stackoverflow.com/a/1322548
template<typename T>
__device__ __host__ std::enable_if_t<sizeof(T) == 1, T> constexpr next_power_of_two(T v) {
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    return ++v;
}

template<typename T>
__device__ __host__ std::enable_if_t<sizeof(T) == 2, T> constexpr next_power_of_two(T v) {
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    return ++v;
}

template<typename T>
__device__ __host__ std::enable_if_t<sizeof(T) == 4, T> constexpr next_power_of_two(T v) {
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return ++v;
}

template<typename T>
__device__ __host__ std::enable_if_t<sizeof(T) == 8, T> constexpr next_power_of_two(T v) {
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return ++v;
}

template<typename T>
__device__ __host__ constexpr inline T const& const_min(T const& a, T const& b) {
    return b < a ? b : a;
}

template<typename T>
__device__ __host__ constexpr inline T const& const_max(T const& a, T const& b) {
    return b > a ? b : a;
}

#endif // if 0

}  // namespace fastertransformer
