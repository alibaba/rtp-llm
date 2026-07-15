#pragma once

#include "rtp_llm/cpp/utils/Logger.h"

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#elif USING_ROCM
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {

inline void pinThreadToDeviceOnce(int device_id) {
    if (device_id < 0) {
        return;
    }

#if USING_CUDA
    thread_local int pinned_device = -1;
    if (pinned_device == device_id) {
        return;
    }
    const auto rc = cudaSetDevice(device_id);
    if (rc != cudaSuccess) {
        RTP_LLM_LOG_WARNING("cudaSetDevice(%d) failed: %s", device_id, cudaGetErrorString(rc));
        return;
    }
    at::cuda::set_device(device_id);
    at::cuda::setCurrentCUDAStream(at::cuda::getDefaultCUDAStream(device_id));
    pinned_device = device_id;
#elif USING_ROCM
    thread_local int pinned_device = -1;
    if (pinned_device == device_id) {
        return;
    }
    const auto rc = hipSetDevice(device_id);
    if (rc != hipSuccess) {
        RTP_LLM_LOG_WARNING("hipSetDevice(%d) failed: %s", device_id, hipGetErrorString(rc));
        return;
    }
    at::hip::set_device(device_id);
    pinned_device = device_id;
#else
    (void)device_id;
#endif
}

}  // namespace rtp_llm
