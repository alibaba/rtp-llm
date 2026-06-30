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
    // ROCm PyTorch exposes ordinary GPU tensors as CUDA; only pinning uses HIP APIs.
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
