#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_FP8
inline __device__ void convert_from_fp8(uint16_t* v, const __nv_fp8_e4m3 u) {
    half h = half(u);
    v[0]   = reinterpret_cast<uint16_t&>(h);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_fp8(uint32_t* v, const fp8_2_t u) {
    half2 h = half2(u);
    v[0]    = reinterpret_cast<uint32_t&>(h);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_fp8(uint2* v, const fp8_4_t u) {
    uint32_t*      v_ptr = reinterpret_cast<uint32_t*>(v);
    const fp8_2_t* u_ptr = reinterpret_cast<const fp8_2_t*>(&u);

    convert_from_fp8(v_ptr + 0, u_ptr[0]);
    convert_from_fp8(v_ptr + 1, u_ptr[1]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_fp8(uint4* v, const fp8_8_t u) {
    uint32_t*      v_ptr = reinterpret_cast<uint32_t*>(v);
    const fp8_2_t* u_ptr = reinterpret_cast<const fp8_2_t*>(&u);

    convert_from_fp8(v_ptr + 0, u_ptr[0]);
    convert_from_fp8(v_ptr + 1, u_ptr[1]);
    convert_from_fp8(v_ptr + 2, u_ptr[2]);
    convert_from_fp8(v_ptr + 3, u_ptr[3]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_fp8(__nv_bfloat16* v, const __nv_fp8_e4m3 u) {
    v[0] = __nv_bfloat16(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_fp8(__nv_bfloat162* v, const fp8_2_t u) {
    union {
        __nv_fp8_e4m3 fp8[2];
        fp8_2_t       fp8_2;
    };

    fp8_2  = u;
    v[0].x = __nv_bfloat16(fp8[0]);
    v[0].y = __nv_bfloat16(fp8[1]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_fp8(bf16_4_t* v, const fp8_4_t u) {

    __nv_bfloat162* v2 = reinterpret_cast<__nv_bfloat162*>(v);
    const fp8_2_t*  u2 = reinterpret_cast<const fp8_2_t*>(&u);
    convert_from_fp8(v2, u2[0]);
    convert_from_fp8(v2 + 1, u2[1]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_fp8(bf16_8_t* v, const fp8_8_t u) {
    __nv_bfloat162* v2 = reinterpret_cast<__nv_bfloat162*>(v);
    convert_from_fp8(v2 + 0, u.x);
    convert_from_fp8(v2 + 1, u.y);
    convert_from_fp8(v2 + 2, u.z);
    convert_from_fp8(v2 + 3, u.w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_fp8(float* v, const __nv_fp8_e4m3 u) {
    v[0] = float(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_fp8(float2* v, const fp8_2_t u) {
    v[0] = float2(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_fp8(float4* v, const fp8_4_t u) {
    v[0] = float4(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_fp8(Float8_* v, const fp8_8_t u) {
    v[0].x = float2(u.x);
    v[0].y = float2(u.y);
    v[0].z = float2(u.z);
    v[0].w = float2(u.w);
}
#endif  // ENALBE_FP8

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
