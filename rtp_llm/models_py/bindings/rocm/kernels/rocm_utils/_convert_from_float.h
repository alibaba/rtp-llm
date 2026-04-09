#pragma once

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
    *dst = __float22bfloat162_rn(src);
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
    dst->x = __float22bfloat162_rn(src.x);
    dst->y = __float22bfloat162_rn(src.y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(bf16_4_t* dst, float4 src) {
    convert_from_float(dst, Float4_{make_float2(src.x, src.y), make_float2(src.z, src.w)});
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(bf16_8_t* dst, Float8_ src) {
    dst->x = __float22bfloat162_rn(src.x);
    dst->y = __float22bfloat162_rn(src.y);
    dst->z = __float22bfloat162_rn(src.z);
    dst->w = __float22bfloat162_rn(src.w);
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
    uint2* tmp;
    convert_from_float(tmp, src);
    dst = reinterpret_cast<int64_t*>(tmp);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
