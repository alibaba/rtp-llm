#pragma once

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
#if USING_ROCM
    __half_raw out = __hmul(*reinterpret_cast<__half_raw*>(&a), *reinterpret_cast<__half_raw*>(&b));
    return *reinterpret_cast<uint16_t*>(&(out.data));
#else
    uint16_t c;
    asm volatile("mul.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
    return c;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint32_t mul(uint32_t a, uint32_t b) {
#if USING_ROCM
    __half2 out = __hmul2(*reinterpret_cast<__half2_raw*>(&a), *reinterpret_cast<__half2_raw*>(&b));
    return *reinterpret_cast<uint32_t*>(&(out.data));
#else
    uint32_t c;
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
    return c;
#endif
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

template<>
inline __device__ __nv_bfloat162 mul(__nv_bfloat162 a, __nv_bfloat162 b) {
    return bf16hmul2(a, b);
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
