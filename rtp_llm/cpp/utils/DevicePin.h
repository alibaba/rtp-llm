#pragma once

#include "rtp_llm/cpp/utils/Logger.h"

#include <c10/core/DeviceType.h>

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#elif USING_ROCM
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {

inline c10::DeviceType getPinnedTorchDeviceType() {
#if USING_ROCM
    return c10::DeviceType::HIP;
#elif USING_CUDA
    return c10::DeviceType::CUDA;
#else
    return c10::DeviceType::CPU;
#endif
}

inline void setCurrentThreadDeviceIfNeeded(int device_id) {
    if (device_id < 0) {
        return;
    }

#if USING_CUDA || USING_ROCM
    thread_local int current_device = -1;
    if (current_device == device_id) {
        return;
    }

    // Thread-pool workers may serve different cache stores over time; a new
    // device_id intentionally retargets the current thread instead of no-oping.
#if USING_CUDA
    at::cuda::set_device(device_id);
#elif USING_ROCM
    at::hip::set_device(device_id);
#endif
    // Keep fail-fast semantics: set_device exceptions propagate, and the
    // cached thread state is updated only after success so later calls retry.
    current_device = device_id;
#else
    // CPU-only builds intentionally no-op; production cache-store builds pin to a GPU backend.
#endif
}

}  // namespace rtp_llm
