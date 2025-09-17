#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename A>
inline __device__ typename packed_type<float, num_elems<A>::value>::type convert_to_float(A u) {
    return {};
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
