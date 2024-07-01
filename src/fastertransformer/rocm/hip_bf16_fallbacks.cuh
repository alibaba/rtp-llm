#pragma once

#include "src/fastertransformer/rocm/hip_utils.h"

namespace fastertransformer {


#ifdef ENABLE_BF16
static inline __device__ __host__ __nv_bfloat162 __floats2bfloat162_rn(float x, float y) {
    return {__float2bfloat16(x), __float2bfloat16(y)};
}
static inline __device__ __host__ __nv_bfloat162 __float2bfloat162_rn(float x) {
    return __floats2bfloat162_rn(x, x);
}

inline __device__ __nv_bfloat162 make_bfloat162(const __nv_bfloat16 x, const __nv_bfloat16 y) {
    __nv_bfloat162 t;
    t.x.data = x.data;
    t.y.data = y.data;
    return t;
}

inline __device__ __nv_bfloat162 make_bfloat162(float x, float y) {
    return make_bfloat162(__float2bfloat16(x), __float2bfloat16(y));
}

inline __device__ float2 bf1622float2(const __nv_bfloat162 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float2 f_val;
    f_val.x = __low2float(val);
    f_val.y = __high2float(val);
    return f_val;
#else
    return __bfloat1622float2(val);
#endif
}

inline __device__ int16_t bf1622int16(__nv_bfloat162 val) {
    val = __hmin2(val, make_bfloat162(127., 127.));
    val = __hmax2(val, make_bfloat162(-128., -128.));
    union {
        int8_t  int8[2];
        int16_t int16;
    };
    int8[0] = static_cast<int8_t>(val.x.data);
    int8[1] = static_cast<int8_t>(val.y.data);
    return int16;
}

inline __device__ __nv_bfloat162 float22bf162(const float2 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __floats2bfloat162_rn(val.x, val.y);
#else
    return __float22bfloat162_rn(val);
#endif
}

inline __device__ __nv_bfloat162 bf162bf162(const __nv_bfloat16 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    __nv_bfloat162 val2;
    val2.x = val;
    val2.y = val;
    return val2;
#else
    return __bfloat162bfloat162(val);
#endif
}

inline __device__ __nv_bfloat162 bf16hadd2(const __nv_bfloat162 x, const __nv_bfloat162 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fxl, fxh, fyl, fyh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    fyl = __low2float(y);
    fyh = __high2float(y);
    return __floats2bfloat162_rn(fxl + fyl, fxh + fyh);
#else
    return __hadd2(x, y);
#endif
}

inline __device__ __nv_bfloat16 bf16hadd(const __nv_bfloat16 x, const __nv_bfloat16 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __float2bfloat16(__bfloat162float(x) + __bfloat162float(y));
#else
    return __hadd(x, y);
#endif
}

inline __device__ __nv_bfloat162 bf16hsub2(const __nv_bfloat162 x, const __nv_bfloat162 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fxl, fxh, fyl, fyh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    fyl = __low2float(y);
    fyh = __high2float(y);
    return __floats2bfloat162_rn(fxl - fyl, fxh - fyh);
#else
    return __hsub2(x, y);
#endif
}

inline __device__ __nv_bfloat16 bf16hsub(const __nv_bfloat16 x, const __nv_bfloat16 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __float2bfloat16(__bfloat162float(x) - __bfloat162float(y));
#else
    return __hsub(x, y);
#endif
}

inline __device__ __nv_bfloat162 bf16hmul2(const __nv_bfloat162 x, const __nv_bfloat162 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fxl, fxh, fyl, fyh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    fyl = __low2float(y);
    fyh = __high2float(y);
    return __floats2bfloat162_rn(fxl * fyl, fxh * fyh);
#else
    return __hmul2(x, y);
#endif
}

inline __device__ __nv_bfloat16 bf16hmul(const __nv_bfloat16 x, const __nv_bfloat16 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __float2bfloat16(__bfloat162float(x) * __bfloat162float(y));
#else
    return __hmul(x, y);
#endif
}

inline __device__ __nv_bfloat162 bf16hfma2(const __nv_bfloat162 x, const __nv_bfloat162 y, const __nv_bfloat162 z) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fxl, fxh, fyl, fyh, fzl, fzh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    fyl = __low2float(y);
    fyh = __high2float(y);
    fzl = __low2float(z);
    fzh = __high2float(z);
    return __floats2bfloat162_rn(fxl * fyl + fzl, fxh * fyh + fzh);
#else
    return __hfma2(x, y, z);
#endif
}

inline __device__ __nv_bfloat16 bf16hfma(const __nv_bfloat16 x, const __nv_bfloat16 y, const __nv_bfloat16 z) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __float2bfloat16(__bfloat162float(x) * __bfloat162float(y) + __bfloat162float(z));
#else
    return __hfma(x, y, z);
#endif
}

inline __device__ __nv_bfloat162 bf16exp2(const __nv_bfloat162 x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fxl, fxh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    ;
    return __floats2bfloat162_rn(expf(fxl), expf(fxh));
#else
    return h2exp(x);
#endif
}

inline __device__ __nv_bfloat16 bf16hadd(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b) + __bfloat162float(c));
#else
    return a + b + c;
#endif
}

inline __device__ __nv_bfloat16 bf16hadd(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c, __nv_bfloat16 d) {
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b) + __bfloat162float(c) + __bfloat162float(d));
}

inline __device__ __nv_bfloat162 bf16hadd2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fal, fah, fbl, fbh, fcl, fch;
    fal = __low2float(a);
    fah = __high2float(a);
    fbl = __low2float(b);
    fbh = __high2float(b);
    fcl = __low2float(c);
    fch = __high2float(c);
    return __floats2bfloat162_rn(fal + fbl + fcl, fah + fbh + fch);
#else
    return __hadd2(__hadd2(a, b), c);
#endif
}

inline __device__ __nv_bfloat16 bf16hmul(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b) * __bfloat162float(c));
#else
    return a * b * c;
#endif
}

inline __device__ __nv_bfloat162 bf16hmul2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fal, fah, fbl, fbh, fcl, fch;
    fal = __low2float(a);
    fah = __high2float(a);
    fbl = __low2float(b);
    fbh = __high2float(b);
    fcl = __low2float(c);
    fch = __high2float(c);
    return __floats2bfloat162_rn(fal * fbl * fcl, fah * fbh * fch);
#else
    return __hmul2(__hmul2(a, b), c);
#endif
}

inline __device__ __nv_bfloat162 bf16hfma2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c, __nv_bfloat162 d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float fal, fah, fbl, fbh, fcl, fch, fdl, fdh;
    fal = __low2float(a);
    fah = __high2float(a);
    fbl = __low2float(b);
    fbh = __high2float(b);
    fcl = __low2float(c);
    fch = __high2float(c);
    fdl = __low2float(d);
    fdh = __high2float(d);
    return __floats2bfloat162_rn(fal * fbl * fcl + fdl, fah * fbh * fch + fdh);
#else
    return __hadd2(bf16hmul2(a, b, c), d);
#endif
}

#endif  // ENABLE_BF16

}  // namespace fastertransformer