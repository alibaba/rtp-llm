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

    thread_local int pinned_device = -1;
    if (pinned_device == device_id) {
        return;
    }

#if USING_CUDA
    at::cuda::set_device(device_id);
#elif USING_ROCM
    at::hip::set_device(device_id);
#endif
    pinned_device = device_id;
}

}  // namespace rtp_llm
