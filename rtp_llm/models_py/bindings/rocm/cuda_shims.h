#pragma once

#include <cstdint>
#include "amd_bfloat16.h"
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel_rocm.h>

#define __nv_bfloat16 amd_bfloat16
#define __nv_bfloat162 amd_bfloat162
// gfx950 (MI355) uses standard IEEE FP8 E4M3, while gfx942 (MI300) uses the
// FNUZ (Finite, No Unsigned Zero) variant. This is a hardware-level difference.
// No fat binary concern: RTP-LLM builds separate binaries per GPU architecture,
// so this compile-time #ifdef is sufficient.
#ifdef ROCM_GFX950
#define __nv_fp8_e4m3 __hip_fp8_e4m3
#else
#define __nv_fp8_e4m3 __hip_fp8_e4m3_fnuz
#endif

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
#define check_cuda_value(val) rtp_llm::rocm::check((val), __FILE__, __LINE__)

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
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeMaxSharedMemoryPerBlock

template<typename T>
inline hipError_t cudaFuncSetAttribute(T func, hipFuncAttribute attr, int value) {
    return hipFuncSetAttribute((const void*)func, attr, value);
}

template<typename T>
inline hipError_t cudaFuncGetAttributes(hipFuncAttributes* attr, T func) {
    return hipFuncGetAttributes(attr, (const void*)func);
}

#define cudaFuncAttributes hipFuncAttributes
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

// Shim for cuda::atomic_ref used in topkLastDim.cu
// Provides minimal interface matching libcu++ cuda::atomic_ref
namespace cuda {

// Map cuda::std to ::std for ROCm
namespace std = ::std;

enum thread_scope {
    thread_scope_block,
    thread_scope_device,
    thread_scope_system
};

enum memory_order {
    memory_order_relaxed
    // TODO: support other memory orders
};

// TODO: currently only support device scope and relaxed memory order
template<typename T, thread_scope Scope = thread_scope_device>
struct atomic_ref {
    T* ptr_;
    __device__ explicit atomic_ref(T& ref): ptr_(&ref) {}
    __device__ T load(memory_order = memory_order_relaxed) const {
        return __atomic_load_n(ptr_, __ATOMIC_RELAXED);
    }
    // TTAS: cheap cached load filters out obvious mismatches before the expensive atomicCAS
    __device__ bool compare_exchange_weak(T& expected,
                                          T  desired,
                                          memory_order = memory_order_relaxed,
                                          memory_order = memory_order_relaxed) {
        T old = __atomic_load_n(ptr_, __ATOMIC_RELAXED);
        if (old == expected) {
            T prev = atomicCAS(ptr_, expected, desired);
            if (prev == expected) {
                return true;
            }
            expected = prev;
            return false;
        }
        expected = old;
        return false;
    }
};

}  // namespace cuda
