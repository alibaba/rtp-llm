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
