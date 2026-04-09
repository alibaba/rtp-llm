#pragma once

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
