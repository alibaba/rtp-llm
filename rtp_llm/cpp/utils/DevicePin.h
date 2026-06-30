#pragma once

#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#elif USING_ROCM
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {

namespace detail {

template<typename SetDevice>
inline void setCurrentThreadDeviceIfNeededImpl(int device_id, int& current_device, SetDevice&& set_device) {
    if (device_id < 0) {
        return;
    }

    if (current_device == device_id) {
        return;
    }

    std::forward<SetDevice>(set_device)(device_id);
    // Cache only successful backend calls so invalid devices still retry later.
    current_device = device_id;
}

}  // namespace detail

inline void setCurrentThreadDeviceIfNeeded(int device_id) {
    if (device_id < 0) {
        return;
    }

#if USING_CUDA || USING_ROCM
    thread_local int current_device = -1;

    // Thread-pool workers may serve different cache stores over time; a new
    // device_id intentionally retargets the current thread instead of no-oping.
    // ROCm PyTorch exposes ordinary GPU tensors as CUDA; only pinning uses HIP APIs.
#if USING_CUDA
    detail::setCurrentThreadDeviceIfNeededImpl(
        device_id, current_device, [](int device) { at::cuda::set_device(device); });
#elif USING_ROCM
    detail::setCurrentThreadDeviceIfNeededImpl(
        device_id, current_device, [](int device) { at::hip::set_device(device); });
#endif
#else
    // CPU-only builds intentionally no-op; production cache-store builds pin to a GPU backend.
#endif
}

}  // namespace rtp_llm
