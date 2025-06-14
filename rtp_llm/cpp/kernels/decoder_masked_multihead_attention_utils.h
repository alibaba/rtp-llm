#if USING_CUDA
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

#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"
#include "rtp_llm/cpp/cuda/cuda_utils.h"
#include <stdint.h>

#ifdef ENABLE_BF16
using rtp_llm::bf16hfma2;
using rtp_llm::bf162bf162;
using rtp_llm::bf1622float2;
using rtp_llm::bf16hmul2;
using rtp_llm::bf16hmul;
using rtp_llm::bf16hadd2;
using rtp_llm::float22bf162;
#endif

namespace rtp_llm {

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float half_to_float(uint16_t h) {
    float f;
    asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
    return f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 half2_to_float2(uint32_t v) {
    uint16_t lo, hi;
    asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(v));
    return make_float2(half_to_float(lo), half_to_float(hi));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float add(float a, float b) {
    return a + b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 add(float2 a, float2 b) {
    float2 c;
    c.x = add(a.x, b.x);
    c.y = add(a.y, b.y);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 add(float4 a, float4 b) {
    float4 c;
    c.x = add(a.x, b.x);
    c.y = add(a.y, b.y);
    c.z = add(a.z, b.z);
    c.w = add(a.w, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_FP8
inline __device__ Float8_ add(Float8_ a, Float8_ b) {
    Float8_ c;
    c.x = add(a.x, b.x);
    c.y = add(a.y, b.y);
    c.z = add(a.z, b.z);
    c.w = add(a.w, b.w);
    return c;
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16
inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) {
    return a + b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ __nv_bfloat162 add(__nv_bfloat162 a, __nv_bfloat162 b) {
    return bf16hadd2(a, b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ bf16_4_t add(bf16_4_t a, bf16_4_t b) {
    bf16_4_t c;
    c.x = add(a.x, b.x);
    c.y = add(a.y, b.y);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ bf16_8_t add(bf16_8_t a, bf16_8_t b) {
    bf16_8_t c;
    c.x = add(a.x, b.x);
    c.y = add(a.y, b.y);
    c.z = add(a.z, b.z);
    c.w = add(a.w, b.w);
    return c;
}
#endif  // ENABLE_BF16

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint16_t add(uint16_t a, uint16_t b) {
    uint16_t c;
    asm volatile("add.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t add(uint32_t a, uint32_t b) {
    uint32_t c;
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint2 add(uint2 a, uint2 b) {
    uint2 c;
    c.x = add(a.x, b.x);
    c.y = add(a.y, b.y);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint4 add(uint4 a, uint4 b) {
    uint4 c;
    c.x = add(a.x, b.x);
    c.y = add(a.y, b.y);
    c.z = add(a.z, b.z);
    c.w = add(a.w, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float vector_abs_max(float a) {
    return cuda_abs(a);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float vector_abs_max(float2 a) {
    return cuda_max(cuda_abs(a.x), cuda_abs(a.y));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float vector_abs_max(float4 a) {
    return cuda_max(cuda_max(cuda_abs(a.x), cuda_abs(a.y)), cuda_max(cuda_abs(a.z), cuda_abs(a.w)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint16_t float_to_half(float f) {
    union {
        uint32_t u32;
        uint16_t u16[2];
    } tmp;
#if 0 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800  // Is it better?
    float zero = 0.f;
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(tmp.u32) : "f"(zero), "f"(f));
#else
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f));
#endif
    return tmp.u16[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t float2_to_half2(float2 f) {
    union {
        uint32_t u32;
        uint16_t u16[2];
    } tmp;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(tmp.u32) : "f"(f.y), "f"(f.x));
#else
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f.x));
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[1]) : "f"(f.y));
#endif
    return tmp.u32;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16

inline __device__ __nv_bfloat16 cuda_max_bf162(__nv_bfloat162 val) {
    return (val.x > val.y) ? val.x : val.y;
}

inline __device__ float vector_abs_max(__nv_bfloat16 a) {
    return cuda_cast<float>((cuda_abs(a)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float vector_abs_max(__nv_bfloat162 a) {
    return cuda_cast<float>(cuda_max_bf162(cuda_abs(a)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float vector_abs_max(bf16_4_t a) {
    return cuda_max(vector_abs_max(a.x), vector_abs_max(a.y));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float vector_abs_max(bf16_8_t a) {
    return cuda_max(cuda_max(vector_abs_max(a.x), vector_abs_max(a.y)),
                    cuda_max(vector_abs_max(a.z), vector_abs_max(a.w)));
}
#endif  // ENABLE_BF16

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float vector_abs_max(uint32_t a) {
    return vector_abs_max(half2_to_float2(a));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float vector_abs_max(uint2 a) {
    return cuda_max(vector_abs_max(a.x), vector_abs_max(a.y));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float vector_abs_max(uint4 a) {
    return cuda_max(cuda_max(vector_abs_max(a.x), vector_abs_max(a.y)),
                    cuda_max(vector_abs_max(a.z), vector_abs_max(a.w)));
}

inline __device__ float add(float a, uint16_t b) {
    return a + half_to_float(b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16
inline __device__ float add(float a, __nv_bfloat16 b) {
    return a + __bfloat162float(b);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_FP8
inline __device__ float add(float a, __nv_fp8_e4m3 b) {
    return a + (float)(b);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 add(uint32_t a, float2 fb) {
    float2 fa = half2_to_float2(a);
    return add(fa, fb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ add(uint2 a, Float4_ fb) {
    Float4_ fc;
    fc.x = add(a.x, fb.x);
    fc.y = add(a.y, fb.y);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ add(uint4 a, Float8_ fb) {
    Float8_ fc;
    fc.x = add(a.x, fb.x);
    fc.y = add(a.y, fb.y);
    fc.z = add(a.z, fb.z);
    fc.w = add(a.w, fb.w);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t h0_h0(uint16_t a) {
    uint32_t b;
    asm volatile("mov.b32 %0, {%1, %1};" : "=r"(b) : "h"(a));
    return b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float fma(float a, float b, float c) {
    return a * b + c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(float2 a, float2 b, float2 c) {
    float2 d;
    d.x = fma(a.x, b.x, c.x);
    d.y = fma(a.y, b.y, c.y);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(float a, float2 b, float2 c) {
    float2 d;
    d.x = fma(a, b.x, c.x);
    d.y = fma(a, b.y, c.y);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 fma(float4 a, float4 b, float4 c) {
    float4 d;
    d.x = fma(a.x, b.x, c.x);
    d.y = fma(a.y, b.y, c.y);
    d.z = fma(a.z, b.z, c.z);
    d.w = fma(a.w, b.w, c.w);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ fma(float4 a, Float4_ fb, Float4_ fc) {
    Float4_ fa, fd;
    fa = reinterpret_cast<Float4_&>(a);

    fd.x = fma(fa.x, fb.x, fc.x);
    fd.y = fma(fa.y, fb.y, fc.y);
    return fd;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(Float8_ a, Float8_ b, Float8_ c) {
    Float8_ d;
    d.x = fma(a.x, b.x, c.x);
    d.y = fma(a.y, b.y, c.y);
    d.z = fma(a.z, b.z, c.z);
    d.w = fma(a.w, b.w, c.w);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 fma(float a, float4 b, float4 c) {
    float4 d;
    d.x = fma(a, b.x, c.x);
    d.y = fma(a, b.y, c.y);
    d.z = fma(a, b.z, c.z);
    d.w = fma(a, b.w, c.w);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 fma(float a, float4 b, Float4_ c) {
    float4 d;
    d.x = fma(a, b.x, c.x.x);
    d.y = fma(a, b.y, c.x.y);
    d.z = fma(a, b.z, c.y.x);
    d.w = fma(a, b.w, c.y.y);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ fma(float a, Float4_ b, Float4_ c) {
    Float4_ d;
    d.x = fma(a, b.x, c.x);
    d.y = fma(a, b.y, c.y);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(float a, Float8_ b, Float8_ c) {
    Float8_ d;
    d.x = fma(a, b.x, c.x);
    d.y = fma(a, b.y, c.y);
    d.z = fma(a, b.z, c.z);
    d.w = fma(a, b.w, c.w);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16
inline __device__ float2 add(__nv_bfloat162 a, float2 fb) {
    float2 fa = bf1622float2(a);
    return add(fa, fb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ add(bf16_4_t a, Float4_ fb) {
    Float4_ fc;
    fc.x = add(a.x, fb.x);
    fc.y = add(a.y, fb.y);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ add(bf16_8_t a, Float8_ fb) {
    Float8_ fc;
    fc.x = add(a.x, fb.x);
    fc.y = add(a.y, fb.y);
    fc.z = add(a.z, fb.z);
    fc.w = add(a.w, fb.w);
    return fc;
}
#endif  // ENABLE_BF16

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t fma(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t fma(uint16_t a, uint32_t b, uint32_t c) {
    return fma(h0_h0(a), b, c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint2 fma(uint2 a, uint2 b, uint2 c) {
    uint2 d;
    d.x = fma(a.x, b.x, c.x);
    d.y = fma(a.y, b.y, c.y);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint2 fma(uint16_t a, uint2 b, uint2 c) {
    uint32_t s = h0_h0(a);
    uint2    d;
    d.x = fma(s, b.x, c.x);
    d.y = fma(s, b.y, c.y);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint4 fma(uint4 a, uint4 b, uint4 c) {
    uint4 d;
    d.x = fma(a.x, b.x, c.x);
    d.y = fma(a.y, b.y, c.y);
    d.z = fma(a.z, b.z, c.z);
    d.w = fma(a.w, b.w, c.w);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint4 fma(uint16_t a, uint4 b, uint4 c) {
    uint32_t s = h0_h0(a);
    uint4    d;
    d.x = fma(s, b.x, c.x);
    d.y = fma(s, b.y, c.y);
    d.z = fma(s, b.z, c.z);
    d.w = fma(s, b.w, c.w);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float fma(uint16_t a, uint16_t b, float fc) {
    float fa = half_to_float(a);
    float fb = half_to_float(b);
    return fa * fb + fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(uint32_t a, uint32_t b, float2 fc) {
    float2 fa = half2_to_float2(a);
    float2 fb = half2_to_float2(b);
    return fma(fa, fb, fc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(float2 fa, uint32_t b, float2 fc) {
    float2 fb = half2_to_float2(b);
    return fma(fa, fb, fc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(uint16_t a, uint32_t b, float2 fc) {
    return fma(h0_h0(a), b, fc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ fma(uint2 a, uint2 b, Float4_ fc) {
    Float4_ fd;
    fd.x = fma(a.x, b.x, fc.x);
    fd.y = fma(a.y, b.y, fc.y);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ fma(uint16_t a, uint2 b, Float4_ fc) {
    uint32_t s = h0_h0(a);
    Float4_  fd;
    fd.x = fma(s, b.x, fc.x);
    fd.y = fma(s, b.y, fc.y);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(uint4 a, uint4 b, Float8_ fc) {
    Float8_ fd;
    fd.x = fma(a.x, b.x, fc.x);
    fd.y = fma(a.y, b.y, fc.y);
    fd.z = fma(a.z, b.z, fc.z);
    fd.w = fma(a.w, b.w, fc.w);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(Float8_ fa, uint4 b, Float8_ fc) {
    Float8_ fd;
    fd.x = fma(fa.x, b.x, fc.x);
    fd.y = fma(fa.y, b.y, fc.y);
    fd.z = fma(fa.z, b.z, fc.z);
    fd.w = fma(fa.w, b.w, fc.w);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(uint16_t a, uint4 b, Float8_ fc) {
    uint32_t s = h0_h0(a);
    Float8_  fd;
    fd.x = fma(s, b.x, fc.x);
    fd.y = fma(s, b.y, fc.y);
    fd.z = fma(s, b.z, fc.z);
    fd.w = fma(s, b.w, fc.w);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float fma(uint16_t a, float fb, float fc) {
    float fa = half_to_float(a);
    return fa * fb + fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(uint32_t a, float2 fb, float2 fc) {
    float2 fa = half2_to_float2(a);
    return fma(fa, fb, fc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(uint16_t a, float2 fb, float2 fc) {
    float  fa = half_to_float(a);
    float2 fd;
    fd.x = fma(fa, fb.x, fc.x);
    fd.y = fma(fa, fb.y, fc.y);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ fma(uint2 a, Float4_ fb, Float4_ fc) {
    Float4_ fd;
    fd.x = fma(a.x, fb.x, fc.x);
    fd.y = fma(a.y, fb.y, fc.y);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ fma(uint16_t a, Float4_ fb, Float4_ fc) {
    Float4_ fd;
    fd.x = fma(a, fb.x, fc.x);
    fd.y = fma(a, fb.y, fc.y);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(uint4 a, Float8_ fb, Float8_ fc) {
    Float8_ fd;
    fd.x = fma(a.x, fb.x, fc.x);
    fd.y = fma(a.y, fb.y, fc.y);
    fd.z = fma(a.z, fb.z, fc.z);
    fd.w = fma(a.w, fb.w, fc.w);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(uint16_t a, Float8_ fb, Float8_ fc) {
    Float8_ fd;
    fd.x = fma(a, fb.x, fc.x);
    fd.y = fma(a, fb.y, fc.y);
    fd.z = fma(a, fb.z, fc.z);
    fd.w = fma(a, fb.w, fc.w);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef ENABLE_BF16
inline __device__ __nv_bfloat162 fma(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
    return bf16hfma2(a, b, c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ __nv_bfloat162 fma(__nv_bfloat16 a, __nv_bfloat162 b, __nv_bfloat162 c) {
    return bf16hfma2(bf162bf162(a), b, c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ bf16_4_t fma(bf16_4_t a, bf16_4_t b, bf16_4_t c) {
    bf16_4_t d;
    d.x = fma(a.x, b.x, c.x);
    d.y = fma(a.y, b.y, c.y);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ bf16_4_t fma(__nv_bfloat16 a, bf16_4_t b, bf16_4_t c) {
    __nv_bfloat162 s = bf162bf162(a);
    bf16_4_t       d;
    d.x = fma(s, b.x, c.x);
    d.y = fma(s, b.y, c.y);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ bf16_8_t fma(bf16_8_t a, bf16_8_t b, bf16_8_t c) {
    bf16_8_t d;
    d.x = fma(a.x, b.x, c.x);
    d.y = fma(a.y, b.y, c.y);
    d.z = fma(a.z, b.z, c.z);
    d.w = fma(a.w, b.w, c.w);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ bf16_8_t fma(__nv_bfloat16 a, bf16_8_t b, bf16_8_t c) {
    __nv_bfloat162 s = bf162bf162(a);
    bf16_8_t       d;
    d.x = fma(s, b.x, c.x);
    d.y = fma(s, b.y, c.y);
    d.z = fma(s, b.z, c.z);
    d.w = fma(s, b.w, c.w);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float fma(__nv_bfloat16 a, __nv_bfloat16 b, float fc) {
    return __bfloat162float(a) * __bfloat162float(b) + fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(__nv_bfloat162 a, __nv_bfloat162 b, float2 fc) {
    float2 fa = bf1622float2(a);
    float2 fb = bf1622float2(b);
    return fma(fa, fb, fc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(float2 fa, __nv_bfloat162 b, float2 fc) {
    float2 fb = bf1622float2(b);
    return fma(fa, fb, fc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(__nv_bfloat16 a, __nv_bfloat162 b, float2 fc) {
    return fma(bf162bf162(a), b, fc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ fma(bf16_4_t a, bf16_4_t b, Float4_ fc) {
    Float4_ fd;
    fd.x = fma(a.x, b.x, fc.x);
    fd.y = fma(a.y, b.y, fc.y);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ fma(__nv_bfloat16 a, bf16_4_t b, Float4_ fc) {
    __nv_bfloat162 s = bf162bf162(a);
    Float4_        fd;
    fd.x = fma(s, b.x, fc.x);
    fd.y = fma(s, b.y, fc.y);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(bf16_8_t a, bf16_8_t b, Float8_ fc) {
    Float8_ fd;
    fd.x = fma(a.x, b.x, fc.x);
    fd.y = fma(a.y, b.y, fc.y);
    fd.z = fma(a.z, b.z, fc.z);
    fd.w = fma(a.w, b.w, fc.w);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(Float8_ fa, bf16_8_t b, Float8_ fc) {
    Float8_ fd;
    fd.x = fma(fa.x, b.x, fc.x);
    fd.y = fma(fa.y, b.y, fc.y);
    fd.z = fma(fa.z, b.z, fc.z);
    fd.w = fma(fa.w, b.w, fc.w);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(__nv_bfloat16 a, bf16_8_t b, Float8_ fc) {
    __nv_bfloat162 s = bf162bf162(a);
    Float8_        fd;
    fd.x = fma(s, b.x, fc.x);
    fd.y = fma(s, b.y, fc.y);
    fd.z = fma(s, b.z, fc.z);
    fd.w = fma(s, b.w, fc.w);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float fma(__nv_bfloat16 a, float fb, float fc) {
    float fa = __bfloat162float(a);
    return fa * fb + fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(__nv_bfloat162 a, float2 fb, float2 fc) {
    float2 fa = bf1622float2(a);
    return fma(fa, fb, fc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(__nv_bfloat16 a, float2 fb, float2 fc) {
    float fa = __bfloat162float(a);
    return fma(fa, fb, fc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ fma(bf16_4_t a, Float4_ fb, Float4_ fc) {
    Float4_ fd;
    fd.x = fma(a.x, fb.x, fc.x);
    fd.y = fma(a.y, fb.y, fc.y);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ fma(__nv_bfloat16 a, Float4_ fb, Float4_ fc) {
    Float4_ fd;
    fd.x = fma(a, fb.x, fc.x);
    fd.y = fma(a, fb.y, fc.y);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(bf16_8_t a, Float8_ fb, Float8_ fc) {
    Float8_ fd;
    fd.x = fma(a.x, fb.x, fc.x);
    fd.y = fma(a.y, fb.y, fc.y);
    fd.z = fma(a.z, fb.z, fc.z);
    fd.w = fma(a.w, fb.w, fc.w);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(__nv_bfloat16 a, Float8_ fb, Float8_ fc) {
    Float8_ fd;
    fd.x = fma(a, fb.x, fc.x);
    fd.y = fma(a, fb.y, fc.y);
    fd.z = fma(a, fb.z, fc.z);
    fd.w = fma(a, fb.w, fc.w);
    return fd;
}

#endif  // ENABLE_BF16

#ifdef ENABLE_FP8

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 fma(float4 a, fp8_4_t b, float4 fc) {
    float4 fd;

    union {
        fp8_4_t fp8_4;
        fp8_2_t fp8_2[2];
    };

    fp8_4      = b;
    float2 fb0 = float2(fp8_2[0]);
    float2 fb1 = float2(fp8_2[1]);

    fd.x = fma(a.x, fb0.x, fc.x);
    fd.y = fma(a.y, fb0.y, fc.y);
    fd.z = fma(a.z, fb1.x, fc.z);
    fd.w = fma(a.w, fb1.y, fc.w);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 fma(float a, fp8_4_t b, float4 fc) {
    float4 fd;

    union {
        fp8_4_t fp8_4;
        fp8_2_t fp8_2[2];
    };

    fp8_4      = b;
    float2 fb0 = float2(fp8_2[0]);
    float2 fb1 = float2(fp8_2[1]);

    fd.x = fma(a, fb0.x, fc.x);
    fd.y = fma(a, fb0.y, fc.y);
    fd.z = fma(a, fb1.x, fc.z);
    fd.w = fma(a, fb1.y, fc.w);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ fma(float4 a, fp8_4_t b, Float4_ fc) {
    float4 fd;
    fd = fma(a, b, reinterpret_cast<float4&>(fc));

    return reinterpret_cast<Float4_&>(fd);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(uint4 a, fp8_8_t b, Float8_ fc) {
    Float8_ fd;

    union {
        fp8_8_t fp8_8;
        fp8_2_t fp8_2[4];
    };

    fp8_8 = b;

    fd.x = fma(a.x, float2(fp8_2[0]), fc.x);
    fd.y = fma(a.y, float2(fp8_2[1]), fc.y);
    fd.z = fma(a.z, float2(fp8_2[2]), fc.z);
    fd.w = fma(a.w, float2(fp8_2[3]), fc.w);

    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(Float8_ fa, fp8_8_t b, Float8_ fc) {
    Float8_ fd;

    union {
        fp8_8_t fp8_8;
        fp8_2_t fp8_2[4];
    };

    fp8_8 = b;

    fd.x = fma(fa.x, float2(fp8_2[0]), fc.x);
    fd.y = fma(fa.y, float2(fp8_2[1]), fc.y);
    fd.z = fma(fa.z, float2(fp8_2[2]), fc.z);
    fd.w = fma(fa.w, float2(fp8_2[3]), fc.w);

    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(float a, fp8_8_t b, Float8_ fc) {
    Float8_ fd;

    union {
        fp8_8_t fp8_8;
        fp8_2_t fp8_2[4];
    };

    fp8_8 = b;

    fd.x = fma(a, float2(fp8_2[0]), fc.x);
    fd.y = fma(a, float2(fp8_2[1]), fc.y);
    fd.z = fma(a, float2(fp8_2[2]), fc.z);
    fd.w = fma(a, float2(fp8_2[3]), fc.w);

    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(uint16_t a, fp8_8_t b, Float8_ fc) {
    return fma(half_to_float(a), b, fc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(bf16_8_t a, fp8_8_t b, Float8_ fc) {
    Float8_ fd;

    union {
        fp8_8_t fp8_8;
        fp8_2_t fp8_2[4];
    };

    fp8_8 = b;

    fd.x = fma(a.x, float2(fp8_2[0]), fc.x);
    fd.y = fma(a.y, float2(fp8_2[1]), fc.y);
    fd.z = fma(a.z, float2(fp8_2[2]), fc.z);
    fd.w = fma(a.w, float2(fp8_2[3]), fc.w);

    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(__nv_bfloat16 a, fp8_8_t b, Float8_ fc) {
    return fma(__bfloat162float(a), b, fc);
}

#endif  // ENABLE_FP8

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 fma(float4 a, int32_t b, float4 fc) {
    float4 fd;

    union {
        int32_t int32;
        ;
        int8_t int8[4];
    };

    int32 = b;

    fd.x = fma(a.x, int8[0], fc.x);
    fd.y = fma(a.y, int8[1], fc.y);
    fd.z = fma(a.z, int8[2], fc.z);
    fd.w = fma(a.w, int8[3], fc.w);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ fma(float4 a, int32_t b, Float4_ fc) {
    float4 fd;
    fd = fma(a, b, reinterpret_cast<float4&>(fc));

    return reinterpret_cast<Float4_&>(fd);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 fma(float a, int32_t b, float4 fc) {
    float4 fd;

    union {
        int32_t int32;
        ;
        int8_t int8[4];
    };

    int32 = b;

    fd.x = fma(a, int8[0], fc.x);
    fd.y = fma(a, int8[1], fc.y);
    fd.z = fma(a, int8[2], fc.z);
    fd.w = fma(a, int8[3], fc.w);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(uint4 a, int64_t b, Float8_ fc) {
    Float8_ fd;

    union {
        int64_t int64;
        int8_t  int8[8];
    };

    int64 = b;

    fd.x = fma(a.x, make_float2(int8[0], int8[1]), fc.x);
    fd.y = fma(a.y, make_float2(int8[2], int8[3]), fc.y);
    fd.z = fma(a.z, make_float2(int8[4], int8[5]), fc.z);
    fd.w = fma(a.w, make_float2(int8[6], int8[7]), fc.w);

    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(Float8_ fa, int64_t b, Float8_ fc) {
    Float8_ fd;

    union {
        int64_t int64;
        int8_t  int8[8];
    };

    int64 = b;

    fd.x = fma(fa.x, make_float2(int8[0], int8[1]), fc.x);
    fd.y = fma(fa.y, make_float2(int8[2], int8[3]), fc.y);
    fd.z = fma(fa.z, make_float2(int8[4], int8[5]), fc.z);
    fd.w = fma(fa.w, make_float2(int8[6], int8[7]), fc.w);

    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(float a, int64_t b, Float8_ fc) {
    Float8_ fd;
    float2  fa = make_float2(a, a);

    union {
        int64_t int64;
        int8_t  int8[8];
    };

    int64 = b;

    fd.x = fma(fa, make_float2(int8[0], int8[1]), fc.x);
    fd.y = fma(fa, make_float2(int8[2], int8[3]), fc.y);
    fd.z = fma(fa, make_float2(int8[4], int8[5]), fc.z);
    fd.w = fma(fa, make_float2(int8[6], int8[7]), fc.w);

    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(uint16_t a, int64_t b, Float8_ fc) {
    return fma(half_to_float(a), b, fc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(bf16_8_t a, int64_t b, Float8_ fc) {
    Float8_ fd;

    union {
        int64_t int64;
        int8_t  int8[8];
    };

    int64 = b;

    fd.x = fma(a.x, make_float2(int8[0], int8[1]), fc.x);
    fd.y = fma(a.y, make_float2(int8[2], int8[3]), fc.y);
    fd.z = fma(a.z, make_float2(int8[4], int8[5]), fc.z);
    fd.w = fma(a.w, make_float2(int8[6], int8[7]), fc.w);

    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(__nv_bfloat16 a, int64_t b, Float8_ fc) {
    return fma(__bfloat162float(a), b, fc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Acc, typename A, typename B>
inline __device__ Acc mul(A a, B b) {
    // This will error out when multiply operation is not supported.
    return Acc(a * b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float mul<float, float>(float a, float b) {
    return a * b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 mul(float2 a, float2 b) {
    float2 c;
    c.x = a.x * b.x;
    c.y = a.y * b.y;
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 mul(float a, float2 b) {
    float2 c;
    c.x = a * b.x;
    c.y = a * b.y;
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float4 mul(float4 a, float4 b) {
    float4 c;
    c.x = a.x * b.x;
    c.y = a.y * b.y;
    c.z = a.z * b.z;
    c.w = a.w * b.w;
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float4_ mul(float4 a, Float4_ b) {
    float4 c;
    c = mul<float4, float4, float4>(a, reinterpret_cast<float4&>(b));
    return reinterpret_cast<Float4_&>(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float4 mul(float a, float4 b) {
    float4 c;
    c.x = a * b.x;
    c.y = a * b.y;
    c.z = a * b.z;
    c.w = a * b.w;
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float4_ mul(float a, Float4_ b) {
    float4 c = mul<float4, float, float4>(a, reinterpret_cast<float4&>(b));
    return reinterpret_cast<Float4_&>(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(float a, Float8_ b) {
    Float8_ c;
    c.x = mul<float2, float, float2>(a, b.x);
    c.y = mul<float2, float, float2>(a, b.y);
    c.z = mul<float2, float, float2>(a, b.z);
    c.w = mul<float2, float, float2>(a, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint16_t mul(uint16_t a, uint16_t b) {
    uint16_t c;
    asm volatile("mul.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint32_t mul(uint32_t a, uint32_t b) {
    uint32_t c;
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint32_t mul(uint16_t a, uint32_t b) {
    return mul<uint32_t, uint32_t, uint32_t>(h0_h0(a), b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint2 mul(uint2 a, uint2 b) {
    uint2 c;
    c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
    c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint2 mul(uint16_t a, uint2 b) {
    uint32_t s = h0_h0(a);
    uint2    c;
    c.x = mul<uint32_t, uint32_t, uint32_t>(s, b.x);
    c.y = mul<uint32_t, uint32_t, uint32_t>(s, b.y);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint4 mul(uint4 a, uint4 b) {
    uint4 c;
    c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
    c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
    c.z = mul<uint32_t, uint32_t, uint32_t>(a.z, b.z);
    c.w = mul<uint32_t, uint32_t, uint32_t>(a.w, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint4 mul(uint16_t a, uint4 b) {
    uint32_t s = h0_h0(a);
    uint4    c;
    c.x = mul<uint32_t, uint32_t, uint32_t>(s, b.x);
    c.y = mul<uint32_t, uint32_t, uint32_t>(s, b.y);
    c.z = mul<uint32_t, uint32_t, uint32_t>(s, b.z);
    c.w = mul<uint32_t, uint32_t, uint32_t>(s, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float mul(uint16_t a, uint16_t b) {
    float fa = half_to_float(a);
    float fb = half_to_float(b);
    return fa * fb;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float mul(uint16_t a, float b) {
    return half_to_float(a) * b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 mul(uint32_t a, uint32_t b) {
    float2 fa = half2_to_float2(a);
    float2 fb = half2_to_float2(b);
    return mul<float2, float2, float2>(fa, fb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 mul(uint32_t a, float2 fb) {
    float2 fa = half2_to_float2(a);
    return mul<float2, float2, float2>(fa, fb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 mul(float2 fa, uint32_t b) {
    float2 fb = half2_to_float2(b);
    return mul<float2, float2, float2>(fa, fb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 mul(uint16_t a, uint32_t b) {
    return mul<float2, uint32_t, uint32_t>(h0_h0(a), b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float4_ mul(uint2 a, uint2 b) {
    Float4_ fc;
    fc.x = mul<float2, uint32_t, uint32_t>(a.x, b.x);
    fc.y = mul<float2, uint32_t, uint32_t>(a.y, b.y);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float4_ mul(uint16_t a, uint2 b) {
    uint32_t s = h0_h0(a);
    Float4_  fc;
    fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);
    fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(uint4 a, uint4 b) {
    Float8_ fc;
    fc.x = mul<float2, uint32_t, uint32_t>(a.x, b.x);
    fc.y = mul<float2, uint32_t, uint32_t>(a.y, b.y);
    fc.z = mul<float2, uint32_t, uint32_t>(a.z, b.z);
    fc.w = mul<float2, uint32_t, uint32_t>(a.w, b.w);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(Float8_ fa, uint4 b) {
    Float8_ fc;
    fc.x = mul<float2, float2, uint32_t>(fa.x, b.x);
    fc.y = mul<float2, float2, uint32_t>(fa.y, b.y);
    fc.z = mul<float2, float2, uint32_t>(fa.z, b.z);
    fc.w = mul<float2, float2, uint32_t>(fa.w, b.w);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(Float8_ fa, Float8_ fb) {
    Float8_ fc;
    fc.x = mul<float2, float2, float2>(fa.x, fb.x);
    fc.y = mul<float2, float2, float2>(fa.y, fb.y);
    fc.z = mul<float2, float2, float2>(fa.z, fb.z);
    fc.w = mul<float2, float2, float2>(fa.w, fb.w);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(uint4 a, Float8_ fb) {
    Float8_ fc;
    fc.x = mul<float2, uint32_t, float2>(a.x, fb.x);
    fc.y = mul<float2, uint32_t, float2>(a.y, fb.y);
    fc.z = mul<float2, uint32_t, float2>(a.z, fb.z);
    fc.w = mul<float2, uint32_t, float2>(a.w, fb.w);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(uint16_t a, uint4 b) {
    uint32_t s = h0_h0(a);
    Float8_  fc;
    fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);
    fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);
    fc.z = mul<float2, uint32_t, uint32_t>(s, b.z);
    fc.w = mul<float2, uint32_t, uint32_t>(s, b.w);
    return fc;
}
////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ uint2 mul(float a, uint2 b)
{
    uint16_t h = float_to_half(a);
    uint2 c = mul<uint2, uint16_t, uint2>(h, b);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(float a, uint4 b) {
    uint16_t h0 = float_to_half(a);
    uint32_t s  = h0_h0(h0);
    Float8_  fc;
    fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);
    fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);
    fc.z = mul<float2, uint32_t, uint32_t>(s, b.z);
    fc.w = mul<float2, uint32_t, uint32_t>(s, b.w);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint4 mul(float a, uint4 b) {
    uint16_t h = float_to_half(a);
    uint4    c = mul<uint4, uint16_t, uint4>(h, b);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16
template<>
inline __device__ __nv_bfloat16 mul(__nv_bfloat16 a, __nv_bfloat16 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __hmul(a, b);
#else
    return bf16hmul(a, b);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ float mul(float a, __nv_bfloat16 b) {
    return mul<float>(a, __bfloat162float(b));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ __nv_bfloat162 mul(__nv_bfloat162 a, __nv_bfloat162 b) {
    return bf16hmul2(a, b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ __nv_bfloat162 mul(float a, __nv_bfloat162 b) {
    return mul<__nv_bfloat162>(__float2bfloat162_rn(a), b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ __nv_bfloat162 mul(__nv_bfloat16 a, __nv_bfloat162 b) {
    return mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(bf162bf162(a), b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ bf16_4_t mul(bf16_4_t a, bf16_4_t b) {
    bf16_4_t c;
    c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);
    c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ bf16_4_t mul(__nv_bfloat16 a, bf16_4_t b) {
    __nv_bfloat162 s = bf162bf162(a);
    bf16_4_t       c;
    c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.x);
    c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.y);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ bf16_4_t mul(float a, bf16_4_t b) {
    return mul<bf16_4_t>(__float2bfloat16(a), b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ bf16_8_t mul(bf16_8_t a, bf16_8_t b) {
    bf16_8_t c;
    c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);
    c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);
    c.z = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.z, b.z);
    c.w = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.w, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ bf16_8_t mul(float a, bf16_8_t b) {
    __nv_bfloat162 a_ = float22bf162(make_float2(a, a));
    bf16_8_t       c;
    c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a_, b.x);
    c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a_, b.y);
    c.z = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a_, b.z);
    c.w = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a_, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ bf16_8_t mul(__nv_bfloat16 a, bf16_8_t b) {
    __nv_bfloat162 s = bf162bf162(a);
    bf16_8_t       c;
    c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.x);
    c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.y);
    c.z = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.z);
    c.w = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float mul(__nv_bfloat16 a, __nv_bfloat16 b) {
    float fa = (float)a;
    float fb = (float)b;
    return fa * fb;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float mul(__nv_bfloat16 a, float b) {
    return __bfloat162float(a) * b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 mul(__nv_bfloat162 a, __nv_bfloat162 b) {
    float2 fa = bf1622float2(a);
    float2 fb = bf1622float2(b);
    return mul<float2, float2, float2>(fa, fb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 mul(__nv_bfloat162 a, float2 fb) {
    float2 fa = bf1622float2(a);
    return mul<float2, float2, float2>(fa, fb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 mul(float2 fa, __nv_bfloat162 b) {
    float2 fb = bf1622float2(b);
    return mul<float2, float2, float2>(fa, fb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 mul(__nv_bfloat16 a, __nv_bfloat162 b) {
    return mul<float2, __nv_bfloat162, __nv_bfloat162>(bf162bf162(a), b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float4_ mul(bf16_4_t a, bf16_4_t b) {
    Float4_ fc;
    fc.x = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);
    fc.y = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float4_ mul(__nv_bfloat16 a, bf16_4_t b) {
    __nv_bfloat162 s = bf162bf162(a);
    Float4_        fc;
    fc.x = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.x);
    fc.y = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.y);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(bf16_8_t a, bf16_8_t b) {
    Float8_ fc;
    fc.x = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);
    fc.y = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);
    fc.z = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.z, b.z);
    fc.w = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.w, b.w);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(bf16_8_t a, Float8_ fb) {
    Float8_ fc;
    fc.x = mul<float2, __nv_bfloat162, float2>(a.x, fb.x);
    fc.y = mul<float2, __nv_bfloat162, float2>(a.y, fb.y);
    fc.z = mul<float2, __nv_bfloat162, float2>(a.z, fb.z);
    fc.w = mul<float2, __nv_bfloat162, float2>(a.w, fb.w);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(Float8_ fa, bf16_8_t b) {
    Float8_ fc;
    fc.x = mul<float2, float2, __nv_bfloat162>(fa.x, b.x);
    fc.y = mul<float2, float2, __nv_bfloat162>(fa.y, b.y);
    fc.z = mul<float2, float2, __nv_bfloat162>(fa.z, b.z);
    fc.w = mul<float2, float2, __nv_bfloat162>(fa.w, b.w);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(__nv_bfloat16 a, bf16_8_t b) {
    __nv_bfloat162 s = bf162bf162(a);
    Float8_        fc;
    fc.x = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.x);
    fc.y = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.y);
    fc.z = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.z);
    fc.w = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.w);
    return fc;
}
#endif  // ENABLE_BF16

#ifdef ENABLE_FP8

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(uint4 a, fp8_8_t b) {
    Float8_ fc;

    union {
        fp8_8_t fp8_8;
        fp8_2_t fp8_2[4];
    };

    fp8_8 = b;

    fc.x = mul<float2, uint32_t, float2>(a.x, float2(fp8_2[0]));
    fc.y = mul<float2, uint32_t, float2>(a.y, float2(fp8_2[1]));
    fc.z = mul<float2, uint32_t, float2>(a.z, float2(fp8_2[2]));
    fc.w = mul<float2, uint32_t, float2>(a.w, float2(fp8_2[3]));

    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(Float8_ fa, fp8_8_t b) {
    Float8_ fc;

    union {
        fp8_8_t fp8_8;
        fp8_2_t fp8_2[4];
    };

    fp8_8 = b;

    fc.x = mul<float2, float2, float2>(fa.x, float2(fp8_2[0]));
    fc.y = mul<float2, float2, float2>(fa.y, float2(fp8_2[1]));
    fc.z = mul<float2, float2, float2>(fa.z, float2(fp8_2[2]));
    fc.w = mul<float2, float2, float2>(fa.w, float2(fp8_2[3]));

    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float4_ mul(float fa, fp8_4_t b) {
    Float4_ fc;

    union {
        fp8_4_t fp8_4;
        fp8_2_t fp8_2[2];
    };

    fp8_4      = b;
    float2 fa2 = make_float2(fa, fa);

    fc.x = mul<float2, float2, float2>(fa2, float2(fp8_2[0]));
    fc.y = mul<float2, float2, float2>(fa2, float2(fp8_2[1]));

    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float4 mul(float fa, fp8_4_t b) {
    Float4_ fc = mul<Float4_, float, fp8_4_t>(fa, b);
    return reinterpret_cast<float4&>(fc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(float fa, fp8_8_t b) {
    Float8_ fc;

    union {
        fp8_8_t fp8_8;
        fp8_2_t fp8_2[4];
    };

    fp8_8      = b;
    float2 fa2 = make_float2(fa, fa);

    fc.x = mul<float2, float2, float2>(fa2, float2(fp8_2[0]));
    fc.y = mul<float2, float2, float2>(fa2, float2(fp8_2[1]));
    fc.z = mul<float2, float2, float2>(fa2, float2(fp8_2[2]));
    fc.w = mul<float2, float2, float2>(fa2, float2(fp8_2[3]));

    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(bf16_8_t a, fp8_8_t b) {
    Float8_ fc;

    union {
        fp8_8_t fp8_8;
        fp8_2_t fp8_2[4];
    };

    fp8_8 = b;

    fc.x = mul<float2, __nv_bfloat162, float2>(a.x, float2(fp8_2[0]));
    fc.y = mul<float2, __nv_bfloat162, float2>(a.y, float2(fp8_2[1]));
    fc.z = mul<float2, __nv_bfloat162, float2>(a.z, float2(fp8_2[2]));
    fc.w = mul<float2, __nv_bfloat162, float2>(a.w, float2(fp8_2[3]));
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float4 mul(float4 fa, fp8_4_t b) {
    float4 fc;

    union {
        fp8_4_t fp8_4;
        fp8_2_t fp8_2[2];
    };

    fp8_4 = b;

    float2 fb0 = float2(fp8_2[0]);
    float2 fb1 = float2(fp8_2[1]);

    fc.x = fa.x * fb0.x;
    fc.y = fa.y * fb0.y;
    fc.z = fa.z * fb1.x;
    fc.w = fa.w * fb1.y;

    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float4_ mul(float4 fa, fp8_4_t b) {
    float4 fc = mul<float4, float4, fp8_4_t>(fa, b);
    return reinterpret_cast<Float4_&>(fc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#endif  // ENABLE_FP8

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(uint4 a, int64_t b) {
    Float8_ fc;

    union {
        int64_t int64;
        int8_t  int8[8];
    };

    int64 = b;

    fc.x = mul<float2, uint32_t, float2>(a.x, make_float2(int8[0], int8[1]));
    fc.y = mul<float2, uint32_t, float2>(a.y, make_float2(int8[2], int8[3]));
    fc.z = mul<float2, uint32_t, float2>(a.z, make_float2(int8[4], int8[5]));
    fc.w = mul<float2, uint32_t, float2>(a.w, make_float2(int8[6], int8[7]));

    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(Float8_ fa, int64_t b) {
    Float8_ fc;

    union {
        int64_t int64;
        int8_t  int8[8];
    };

    int64 = b;

    fc.x = mul<float2, float2, float2>(fa.x, make_float2(int8[0], int8[1]));
    fc.y = mul<float2, float2, float2>(fa.y, make_float2(int8[2], int8[3]));
    fc.z = mul<float2, float2, float2>(fa.z, make_float2(int8[4], int8[5]));
    fc.w = mul<float2, float2, float2>(fa.w, make_float2(int8[6], int8[7]));

    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(float fa, int64_t b) {
    Float8_ fc;

    union {
        int64_t int64;
        int8_t  int8[8];
    };

    int64      = b;
    float2 fa2 = make_float2(fa, fa);

    fc.x = mul<float2, float2, float2>(fa2, make_float2(int8[0], int8[1]));
    fc.y = mul<float2, float2, float2>(fa2, make_float2(int8[2], int8[3]));
    fc.z = mul<float2, float2, float2>(fa2, make_float2(int8[4], int8[5]));
    fc.w = mul<float2, float2, float2>(fa2, make_float2(int8[6], int8[7]));
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float4_ mul(float fa, int32_t b) {
    Float4_ fc;

    union {
        int32_t int32;
        int8_t  int8[4];
    };

    int32      = b;
    float2 fa2 = make_float2(fa, fa);

    fc.x = mul<float2, float2, float2>(fa2, make_float2(int8[0], int8[1]));
    fc.y = mul<float2, float2, float2>(fa2, make_float2(int8[2], int8[3]));
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float4 mul(float fa, int32_t b) {
    Float4_ fc = mul<Float4_, float, int32_t>(fa, b);
    return reinterpret_cast<float4&>(fc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16

template<>
inline __device__ Float8_ mul(bf16_8_t a, int64_t b) {
    Float8_ fc;

    union {
        int64_t int64;
        int8_t  int8[8];
    };

    int64 = b;

    fc.x = mul<float2, __nv_bfloat162, float2>(a.x, make_float2(int8[0], int8[1]));
    fc.y = mul<float2, __nv_bfloat162, float2>(a.y, make_float2(int8[2], int8[3]));
    fc.z = mul<float2, __nv_bfloat162, float2>(a.z, make_float2(int8[4], int8[5]));
    fc.w = mul<float2, __nv_bfloat162, float2>(a.w, make_float2(int8[6], int8[7]));

    return fc;
}

#endif  // ENABLE_BF16

///////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float4 mul(float4 a, int32_t b) {
    float4 fc;

    union {
        int32_t int32;
        int8_t  int8[4];
    };

    int32 = b;

    fc.x = a.x * float(int8[0]);
    fc.y = a.y * float(int8[1]);
    fc.z = a.z * float(int8[2]);
    fc.w = a.w * float(int8[3]);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(float v) {
    return v;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(float2 v) {
    return v.x + v.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(float4 v) {
    return v.x + v.y + v.z + v.w;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(Float4_ v) {
    return v.x.x + v.x.y + v.y.x + v.y.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(Float8_ v) {
    float out = 0.f;

    out += sum(v.x);
    out += sum(v.y);
    out += sum(v.z);
    out += sum(v.w);

    return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16
inline __device__ float sum(__nv_bfloat162 v) {
    float2 vf = bf1622float2(v);
    return vf.x + vf.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(bf16_4_t v) {
    return sum(v.x) + sum(v.y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(bf16_8_t v) {
    return sum(v.x) + sum(v.y) + sum(v.z) + sum(v.w);
}
#endif  // ENABLE_BF16
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(uint16_t v) {
    return half_to_float(v);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(uint32_t v) {
    float2 tmp = half2_to_float2(v);
    return tmp.x + tmp.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(uint2 v) {
    uint32_t c = add(v.x, v.y);
    return sum(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(uint4 v) {
#if 1
    uint32_t c = add(v.x, v.y);
    c          = add(c, v.z);
    c          = add(c, v.w);
#else
    uint32_t c = add(v.x, v.y);
    uint32_t d = add(v.z, v.w);
    c          = add(c, d);
#endif
    return sum(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline __device__ float dot(T a, T b) {
    return sum(mul<T, T, T>(a, b));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename A, typename T>
inline __device__ float dot(T a, T b) {
    return sum(mul<A, T, T>(a, b));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void zero(uint16_t& dst) {
    dst = uint16_t(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline __device__ void zero(T& dst) {
    constexpr int WORDS = sizeof(T) / 4;

    union {
        T        raw;
        uint32_t words[WORDS];
    } tmp;

#pragma unroll
    for (int ii = 0; ii < WORDS; ++ii) {
        tmp.words[ii] = 0u;
    }
    dst = tmp.raw;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __inline__ void logn_attention(float& vec, const int seq_idx, const int logn_seq_len) {
    float logn = logf(seq_idx + 1) / logf(logn_seq_len);
    if (seq_idx > logn_seq_len) {
        vec = vec * logn;
    }
}

__device__ __inline__ void logn_attention(float2& vec, const int seq_idx, const int logn_seq_len) {
    float logn = logf(seq_idx + 1) / logf(logn_seq_len);
    if (seq_idx > logn_seq_len) {
        vec.x = vec.x * logn;
        vec.y = vec.y * logn;
    }
}

__device__ __inline__ void logn_attention(uint32_t& vec, const int seq_idx, const int logn_seq_len) {
    float2 result = half2_to_float2(vec);
    float  logn   = logf(seq_idx + 1) / logf(logn_seq_len);
    if (seq_idx > logn_seq_len) {
        result.x = result.x * logn;
        result.y = result.y * logn;
    }
    vec = float2_to_half2(result);
}

__device__ __inline__ void logn_attention(float4& vec, const int seq_idx, const int logn_seq_len) {
    float logn = logf(seq_idx + 1) / logf(logn_seq_len);
    if (seq_idx > logn_seq_len) {
        vec.x = vec.x * logn;
        vec.y = vec.y * logn;
        vec.z = vec.z * logn;
        vec.w = vec.w * logn;
    }
}

__device__ __inline__ void logn_attention(uint2& vec, const int seq_idx, const int logn_seq_len) {
    float2 result0 = half2_to_float2(vec.x);
    float2 result1 = half2_to_float2(vec.y);
    float  logn    = logf(seq_idx + 1) / logf(logn_seq_len);
    if (seq_idx > logn_seq_len) {
        result0.x = result0.x * logn;
        result0.y = result0.y * logn;
        result1.x = result1.x * logn;
        result1.y = result1.y * logn;
    }
    vec.x = float2_to_half2(result0);
    vec.y = float2_to_half2(result1);
}

__device__ __inline__ void logn_attention(uint4& vec, const int seq_idx, const int logn_seq_len) {
    float2 result0 = half2_to_float2(vec.x);
    float2 result1 = half2_to_float2(vec.y);
    float2 result2 = half2_to_float2(vec.z);
    float2 result3 = half2_to_float2(vec.w);
    float  logn    = logf(seq_idx + 1) / logf(logn_seq_len);
    if (seq_idx > logn_seq_len) {
        result0.x = result0.x * logn;
        result0.y = result0.y * logn;
        result1.x = result1.x * logn;
        result1.y = result1.y * logn;
        result2.x = result2.x * logn;
        result2.y = result2.y * logn;
        result3.x = result3.x * logn;
        result3.y = result3.y * logn;
    }
    vec.x = float2_to_half2(result0);
    vec.y = float2_to_half2(result1);
    vec.z = float2_to_half2(result2);
    vec.w = float2_to_half2(result3);
}

#ifdef ENABLE_BF16

__device__ __inline__ void logn_attention(__nv_bfloat162& vec, const int seq_idx, const int logn_seq_len) {
    if (seq_idx > logn_seq_len) {
        __nv_bfloat16 scalar = __nv_bfloat16((logf(seq_idx + 1) / logf(logn_seq_len)));
        vec.x                = vec.x * scalar;
        vec.y                = vec.y * scalar;
    }
}

__device__ __inline__ void logn_attention(__nv_bfloat16& vec, const int seq_idx, const int logn_seq_len) {
    if (seq_idx > logn_seq_len) {
        __nv_bfloat16 scalar = __nv_bfloat16((logf(seq_idx + 1) / logf(logn_seq_len)));
        vec                  = vec * scalar;
    }
}

__device__ __inline__ void logn_attention(bf16_8_t& vec, const int seq_idx, const int logn_seq_len) {
    if (seq_idx > logn_seq_len) {
        __nv_bfloat16  scalar  = __nv_bfloat16((logf(seq_idx + 1) / logf(logn_seq_len)));
        __nv_bfloat162 scalar2 = __nv_bfloat162(scalar, scalar);
        vec.x                  = vec.x * scalar2;
        vec.y                  = vec.y * scalar2;
        vec.z                  = vec.z * scalar2;
        vec.w                  = vec.w * scalar2;
    }
}

__device__ __inline__ void logn_attention(bf16_4_t& vec, const int seq_idx, const int logn_seq_len) {
    if (seq_idx > logn_seq_len) {
        __nv_bfloat16  scalar  = __nv_bfloat16((logf(seq_idx + 1) / logf(logn_seq_len)));
        __nv_bfloat162 scalar2 = __nv_bfloat162(scalar, scalar);
        vec.x                  = vec.x * scalar2;
        vec.y                  = vec.y * scalar2;
    }
}

#endif
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(float* dst, float src) {
    *dst = src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint16_t* dst, float src) {
    *dst = float_to_half(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint32_t* dst, float2 src) {
    *dst = float2_to_half2(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef ENABLE_BF16
inline __device__ void convert_from_float(__nv_bfloat16* dst, float src) {
    *dst = __float2bfloat16(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(__nv_bfloat162* dst, float2 src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    *dst = __float22bfloat162_rn(src);
#else
    *dst   = __floats2bfloat162_rn(src.x, src.y);
#endif
}
#endif  // ENABLE_BF16
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint2* dst, Float4_ src) {
    dst->x = float2_to_half2(src.x);
    dst->y = float2_to_half2(src.y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint2* dst, float4 src) {
    convert_from_float(dst, Float4_{make_float2(src.x, src.y), make_float2(src.z, src.w)});
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint4* dst, Float8_ src) {
    dst->x = float2_to_half2(src.x);
    dst->y = float2_to_half2(src.y);
    dst->z = float2_to_half2(src.z);
    dst->w = float2_to_half2(src.w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16
inline __device__ void convert_from_float(bf16_4_t* dst, Float4_ src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    dst->x = __float22bfloat162_rn(src.x);
    dst->y = __float22bfloat162_rn(src.y);
#else
    dst->x = __floats2bfloat162_rn(src.x.x, src.x.y);
    dst->y = __floats2bfloat162_rn(src.y.x, src.y.y);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(bf16_4_t* dst, float4 src) {
    convert_from_float(dst, Float4_{make_float2(src.x, src.y), make_float2(src.z, src.w)});
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(bf16_8_t* dst, Float8_ src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    dst->x = __float22bfloat162_rn(src.x);
    dst->y = __float22bfloat162_rn(src.y);
    dst->z = __float22bfloat162_rn(src.z);
    dst->w = __float22bfloat162_rn(src.w);
#else
    dst->x = __floats2bfloat162_rn(src.x.x, src.x.y);
    dst->y = __floats2bfloat162_rn(src.y.x, src.y.y);
    dst->z = __floats2bfloat162_rn(src.z.x, src.z.y);
    dst->w = __floats2bfloat162_rn(src.w.x, src.w.y);
#endif
}
#endif  // ENABLE_BF16

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_FP8
inline __device__ void convert_from_float(fp8_4_t* dst, float4 src) {
    *dst = fp8_4_t(src);
}

inline __device__ void convert_from_float(fp8_2_t* dst, float2 src) {
    *dst = fp8_2_t(src);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(float2* dst, float2 src) {
    *dst = src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(float4* dst, float4 src) {
    *dst = src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(Float8_* dst, Float8_ src) {
    *dst = src;
}

inline __device__ void convert_from_float(int32_t* dst, float2 src) {
    *dst = float2_to_half2(src);
}

inline __device__ void convert_from_float(int64_t* dst, float4 src) {
    uint2 tmp;
    convert_from_float(&tmp, src);
    *dst = *(reinterpret_cast<int64_t*>(&tmp));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename A>
inline __device__ typename packed_type<float, num_elems<A>::value>::type convert_to_float(A u) {
    return {};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ convert_to_float(Float8_ u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float4 convert_to_float(float4 u) {
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 convert_to_float(float2 u) {
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float convert_to_float(float u) {
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ convert_to_float(uint4 u) {
    Float8_ f8;
    f8.x = half2_to_float2(u.x);
    f8.y = half2_to_float2(u.y);
    f8.z = half2_to_float2(u.z);
    f8.w = half2_to_float2(u.w);
    return f8;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float4 convert_to_float(uint2 u) {
    float4 ret;
    float2 f2x = half2_to_float2(u.x);
    float2 f2y = half2_to_float2(u.y);
    ret.x      = f2x.x;
    ret.y      = f2x.y;
    ret.z      = f2y.x;
    ret.w      = f2y.y;
    return ret;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 convert_to_float(uint32_t u) {
    return half2_to_float2(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float convert_to_float(half u) {
    return static_cast<float>(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16
template<>
inline __device__ float convert_to_float(__nv_bfloat16 u) {
    return static_cast<float>(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 convert_to_float(__nv_bfloat162 u) {
    return bf1622float2(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float4 convert_to_float(bf16_4_t u) {
    float4 ret;
    float2 f2x = bf1622float2(u.x);
    float2 f2y = bf1622float2(u.y);
    ret.x      = f2x.x;
    ret.y      = f2x.y;
    ret.z      = f2y.x;
    ret.w      = f2y.y;
    return ret;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ convert_to_float(bf16_8_t u) {
    Float8_ f8;
    f8.x = bf1622float2(u.x);
    f8.y = bf1622float2(u.y);
    f8.z = bf1622float2(u.z);
    f8.w = bf1622float2(u.w);
    return f8;
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_FP8
inline __device__ void convert_from_fp8(uint16_t* v, const __nv_fp8_e4m3 u)
{
    half h = half(u);
    v[0] = reinterpret_cast<uint16_t&>(h);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_fp8(uint32_t* v, const fp8_2_t u)
{
    half2 h = half2(u);
    v[0] = reinterpret_cast<uint32_t&>(h);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_fp8(uint2* v, const fp8_4_t u)
{
    uint32_t* v_ptr = reinterpret_cast<uint32_t*>(v);
    fp8_2_t const* u_ptr = reinterpret_cast<fp8_2_t const*>(&u);

    convert_from_fp8(v_ptr + 0, u_ptr[0]);
    convert_from_fp8(v_ptr + 1, u_ptr[1]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_fp8(uint4* v, const fp8_8_t u)
{
    uint32_t* v_ptr = reinterpret_cast<uint32_t*>(v);
    fp8_2_t const* u_ptr = reinterpret_cast<fp8_2_t const*>(&u);

    convert_from_fp8(v_ptr + 0, u_ptr[0]);
    convert_from_fp8(v_ptr + 1, u_ptr[1]);
    convert_from_fp8(v_ptr + 2, u_ptr[2]);
    convert_from_fp8(v_ptr + 3, u_ptr[3]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_fp8(__nv_bfloat16* v, const __nv_fp8_e4m3 u)
{
    v[0] = __nv_bfloat16(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_fp8(__nv_bfloat162* v, const fp8_2_t u)
{
    union
    {
        __nv_fp8_e4m3 fp8[2];
        fp8_2_t fp8_2;
    };

    fp8_2 = u;
    v[0].x = __nv_bfloat16(fp8[0]);
    v[0].y = __nv_bfloat16(fp8[1]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_fp8(bf16_4_t* v, const fp8_4_t u)
{

    __nv_bfloat162* v2 = reinterpret_cast<__nv_bfloat162*>(v);
    fp8_2_t const* u2 = reinterpret_cast<fp8_2_t const*>(&u);
    convert_from_fp8(v2, u2[0]);
    convert_from_fp8(v2 + 1, u2[1]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_fp8(bf16_8_t* v, const fp8_8_t u)
{
    __nv_bfloat162* v2 = reinterpret_cast<__nv_bfloat162*>(v);
    convert_from_fp8(v2 + 0, u.x);
    convert_from_fp8(v2 + 1, u.y);
    convert_from_fp8(v2 + 2, u.z);
    convert_from_fp8(v2 + 3, u.w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_fp8(float* v, const __nv_fp8_e4m3 u)
{
    v[0] = float(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_fp8(float2* v, const fp8_2_t u)
{
    v[0] = float2(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_fp8(float4* v, const fp8_4_t u)
{
    v[0] = float4(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_fp8(Float8_* v, const fp8_8_t u)
{
    v[0].x = float2(u.x);
    v[0].y = float2(u.y);
    v[0].z = float2(u.z);
    v[0].w = float2(u.w);
}

#endif // ENALBE_FP8

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float float_from_int8(int8_t u) {
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 float_from_int8(int16_t u) {
    union {
        int16_t int16;
        int8_t  int8[2];
    };

    int16 = u;
    return make_float2(int8[0], int8[1]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 float_from_int8(int32_t u) {
    union {
        int32_t int32;
        int8_t  int8[4];
    };

    int32 = u;
    return make_float4(int8[0], int8[1], int8[2], int8[3]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// clang-format off
inline __device__ Float8_ float_from_int8(int64_t u)
{
    union {
        int64_t int64;
        int16_t int16[4];
    };
    int64 = u;
    return Float8_ {float_from_int8(int16[0]),
                    float_from_int8(int16[1]),
                    float_from_int8(int16[2]),
                    float_from_int8(int16[3])};
}

// clang-format on

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_FP8
inline __device__ void convert_to_fp8(__nv_fp8_e4m3* v, const __nv_bfloat16 u) {
    v[0] = __nv_fp8_e4m3(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(fp8_2_t* v, const __nv_bfloat162 u) {
    v[0] = fp8_2_t(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(fp8_4_t* v, const bf16_4_t u) {
    reinterpret_cast<fp8_2_t*>(v)[0] = fp8_2_t(u.x);
    reinterpret_cast<fp8_2_t*>(v)[1] = fp8_2_t(u.y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(fp8_8_t* v, const bf16_8_t u) {
    v[0].x = fp8_2_t(u.x);
    v[0].y = fp8_2_t(u.y);
    v[0].z = fp8_2_t(u.z);
    v[0].w = fp8_2_t(u.w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(__nv_fp8_e4m3* v, const half u) {
    v[0] = __nv_fp8_e4m3(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(__nv_fp8_e4m3* v, const uint16_t u) {
    v[0] = __nv_fp8_e4m3(reinterpret_cast<const half&>(u));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(fp8_2_t* v, const uint32_t u) {
    v[0] = fp8_2_t(reinterpret_cast<const half2&>(u));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(fp8_4_t* v, const uint2 u) {
    union {
        uint2 u2;
        half2 h2[2];
    };

    u2 = u;

    reinterpret_cast<fp8_2_t*>(v)[0] = fp8_2_t(h2[0]);
    reinterpret_cast<fp8_2_t*>(v)[1] = fp8_2_t(h2[1]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(fp8_8_t* v, const uint4 u) {
    union {
        uint4 u4;
        half2 h2[4];
    };

    u4 = u;

    v[0].x = fp8_2_t(h2[0]);
    v[0].y = fp8_2_t(h2[1]);
    v[0].z = fp8_2_t(h2[2]);
    v[0].w = fp8_2_t(h2[3]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(__nv_fp8_e4m3* v, const float u) {
    v[0] = __nv_fp8_e4m3(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(fp8_2_t* v, const float2 u) {
    v[0] = fp8_2_t(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(fp8_4_t* v, const float4 u) {
    v[0] = fp8_4_t(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_to_fp8(fp8_8_t* v, const Float8_ u) {
    v[0].x = fp8_2_t(u.x);
    v[0].y = fp8_2_t(u.y);
    v[0].z = fp8_2_t(u.z);
    v[0].w = fp8_2_t(u.w);
}
#endif  // ENABLE_FP8

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ int8_t cast_to_int8(float val) {
    union {
        int8_t  int8[2];
        int16_t int16;
    };

    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=h"(int16) : "f"(val));
    return int8[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ int32_t cast_to_int8(float2 val) {
    union {
        int8_t  int8[2];
        int32_t int32;
    };

    int8[0] = cast_to_int8(val.x);
    int8[1] = cast_to_int8(val.y);
    return int32;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ int32_t cast_to_int8(float4 val) {
    union {
        int8_t  int8[4];
        int32_t int32;
    };

    int8[0] = cast_to_int8(val.x);
    int8[1] = cast_to_int8(val.y);
    int8[2] = cast_to_int8(val.z);
    int8[3] = cast_to_int8(val.w);
    return int32;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ int64_t cast_to_int8(Float8_ val) {
    union {
        int8_t  int8[8];
        int64_t int64;
    };

    int8[0] = cast_to_int8(val.x.x);
    int8[1] = cast_to_int8(val.x.y);
    int8[2] = cast_to_int8(val.y.x);
    int8[3] = cast_to_int8(val.y.y);
    int8[4] = cast_to_int8(val.z.x);
    int8[5] = cast_to_int8(val.z.y);
    int8[6] = cast_to_int8(val.w.x);
    int8[7] = cast_to_int8(val.w.y);
    return int64;
}

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

////////////////////////////////////////////////////////////////////////////////////////////////////

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

}  // namespace rtp_llm
#endif

#if USING_ROCM
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
#include "rtp_llm/cpp/cuda/cuda_type_utils.cuh"
#if USING_CUDA
#include "rtp_llm/cpp/cuda/cuda_utils.h"
#endif

#if USING_ROCM
#include "rtp_llm/cpp/rocm/hip_utils.h"
using namespace rtp_llm::rocm;
#endif

#ifdef ENABLE_BF16
using rtp_llm::bf16hfma2;
using rtp_llm::bf162bf162;
using rtp_llm::bf1622float2;
using rtp_llm::bf16hmul2;
using rtp_llm::bf16hmul;
using rtp_llm::bf16hadd2;
using rtp_llm::float22bf162;
#endif

namespace rtp_llm {

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

}  // namespace rtp_llm
#endif
