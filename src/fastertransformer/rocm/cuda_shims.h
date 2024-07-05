/*
 * map cudaXxx APIs and types into hipXxx
 * because kernels/ uses cudaXxxx only
 */
#pragma once

#include <hip/hip_fp16.h>
#if ENABLE_BF16
#include <hip/hip_bf16.h>  // __hip_bfloat16
// #include <hip/hip_bfloat16.h> // hip_bfloat16
#endif

template<typename T_OUT, typename T_IN>
__host__ __device__ inline T_OUT special_cast(T_IN val) {
    return val;
}
#if ENABLE_BF16
static inline __device__ __host__ __hip_bfloat162 __float2bfloat162_rn(float x) {
    return {__float2bfloat16(x), __float2bfloat16(x)};
}
static inline __device__ __host__ __hip_bfloat162 __floats2bfloat162_rn(float x, float y) {
    return {__float2bfloat16(x), __float2bfloat16(y)};
}
template<>
__host__ __device__ inline __hip_bfloat16 special_cast<__hip_bfloat16, float>(float val) {
    return __float2bfloat16(val);
};
template<>
__host__ __device__ inline float special_cast<float, __hip_bfloat16>(__hip_bfloat16 val) {
    return __bfloat162float(val);
};
#endif

/* **************************** cuda warpper ***************************** */
#if ENABLE_BF16
#define __nv_bfloat16 __hip_bfloat16
#define __nv_bfloat162 __hip_bfloat162
#endif
#define cudaStream_t hipStream_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpy hipMemcpy
#define sync_check_cuda_error sync_check_hip_error
#define check_cuda_error check_hip_error
