#pragma once

#include <cstdint>
#include "amd_bfloat16.h"
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel_rocm.h>

#define __nv_bfloat16 amd_bfloat16
#define __nv_bfloat162 amd_bfloat162
#define __nv_fp8_e4m3 __hip_fp8_e4m3_fnuz

static inline __device__ __host__ __nv_bfloat162 __float2bfloat162_rn(float x) {
    return {__nv_bfloat16(x), __nv_bfloat16(x)};
}
static inline __device__ __host__ __nv_bfloat162 __floats2bfloat162_rn(float x, float y) {
    return {__nv_bfloat16(x), __nv_bfloat16(y)};
}
static inline __device__ __host__ __nv_bfloat162 __ldg(const __nv_bfloat162* ptr) {
    return *ptr;
}
static inline __device__ __host__ __nv_bfloat16 __ldg(const __nv_bfloat16* ptr) {
    return *ptr;
}

template<typename T>
__device__ inline T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = 32) {
    (void)mask;
    return __shfl_xor(var, laneMask, width);
}

template<typename T>
__device__ inline T __shfl_sync(unsigned mask, T var, int laneMask, int width = 32) {
    (void)mask;
    return __shfl(var, laneMask, width);
}

__device__ inline unsigned __ballot_sync(unsigned mask, int predicate) {
    (void)mask;
    return __ballot(predicate);
}

template<typename T_OUT, typename T_IN>
__host__ __device__ inline T_OUT special_cast(T_IN val) {
    return val;
}
#ifdef ENABLE_BF16
template<>
__host__ __device__ inline amd_bfloat16 special_cast<amd_bfloat16, float>(float val) {
    return __float2bfloat16(val);
};
template<>
__host__ __device__ inline float special_cast<float, amd_bfloat16>(amd_bfloat16 val) {
    return __bfloat162float(val);
};
#endif

#define curand_uniform hiprand_uniform
#define curand_init hiprand_init
#define curandState_t hiprandState_t

#define cub hipcub
#define check_cuda_value ROCM_CHECK

#define cudaStream_t hipStream_t
#define cudaStreamSynchronize hipStreamSynchronize

#define cudaEvent_t hipEvent_t

#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrMaxSharedMemoryPerMultiprocessor hipDeviceAttributeMaxSharedMemoryPerMultiprocessor

#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaDeviceSynchronize hipDeviceSynchronize

#define cudaFree hipFree
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMalloc hipMalloc
#define cudaMallocAsync hipMallocAsync
#define cudaFree hipFree
#define cudaFreeAsync hipFreeAsync

#define cudaEventSynchronize hipEventSynchronize
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEventDestroy hipEventDestroy
#define cudaEventElapsedTime hipEventElapsedTime

#define cudaPeekAtLastError hipPeekAtLastError
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define check_cuda_error() rtp_llm::rocm::syncAndCheckInDebug(__FILE__, __LINE__)
#define cudaDeviceProp hipDeviceProp_t

#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled

#define cudaStreamQuery hipStreamQuery
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamDefault hipStreamDefault

#define CUDA_IPC_HANDLE_SIZE HIP_IPC_HANDLE_SIZE
#define cudaIpcGetMemHandle hipIpcGetMemHandle
#define cudaIpcOpenMemHandle hipIpcOpenMemHandle
#define cudaIpcCloseMemHandle hipIpcCloseMemHandle
#define cudaIpcMemHandle_t hipIpcMemHandle_t
#define cudaIpcOpenMemHandle hipIpcOpenMemHandle
#define cudaIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess

// Taken from cuda_utils.h