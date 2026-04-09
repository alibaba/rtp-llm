#pragma once

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

#ifdef ENABLE_BF16
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
#endif
