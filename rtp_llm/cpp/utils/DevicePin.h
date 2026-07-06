#pragma once

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <utility>

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#elif USING_ROCM
#include <c10/hip/HIPFunctions.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {

namespace detail {

template<typename SetDeviceContext, typename SetDefaultStream>
inline void setCurrentThreadDeviceIfNeededImpl(int                device_id,
                                               int&               current_device,
                                               SetDeviceContext&& set_device_context,
                                               SetDefaultStream&& set_default_stream) {
    if (device_id < 0) {
        return;
    }

    const bool device_changed = current_device != device_id;
    if (device_changed) {
        std::forward<SetDeviceContext>(set_device_context)(device_id);
    }

    std::forward<SetDefaultStream>(set_default_stream)(device_id);
    // Cache only fully successful backend setup so failures still retry later.
    current_device = device_id;
}

}  // namespace detail

inline void setCurrentThreadDeviceContext(int device_id) {
    if (device_id < 0) {
        return;
    }

#if USING_CUDA
    auto err = cudaSetDevice(device_id);
    RTP_LLM_CHECK_WITH_INFO(err == cudaSuccess, "cudaSetDevice(%d) failed: %s", device_id, cudaGetErrorString(err));
    at::cuda::set_device(device_id);
#elif USING_ROCM
    auto err = hipSetDevice(device_id);
    RTP_LLM_CHECK_WITH_INFO(err == hipSuccess, "hipSetDevice(%d) failed: %s", device_id, hipGetErrorString(err));
    c10::hip::set_device(device_id);
#else
    // CPU-only builds intentionally no-op; production cache-store builds pin to a GPU backend.
#endif
}

inline void setCurrentThreadDefaultStream(int device_id) {
    if (device_id < 0) {
        return;
    }

#if USING_CUDA
    at::cuda::setCurrentCUDAStream(at::cuda::getDefaultCUDAStream(device_id));
#elif USING_ROCM
    c10::hip::setCurrentHIPStream(c10::hip::getDefaultHIPStream(device_id));
#else
    // CPU-only builds intentionally no-op; production cache-store builds pin to a GPU backend.
#endif
}

inline void setCurrentThreadDevice(int device_id) {
    setCurrentThreadDeviceContext(device_id);
    setCurrentThreadDefaultStream(device_id);
}

inline void setCurrentThreadDeviceIfNeeded(int device_id) {
    if (device_id < 0) {
        return;
    }

#if USING_CUDA || USING_ROCM
    thread_local int current_device = -1;

    // Thread-pool workers may serve different cache stores over time; a new
    // device_id intentionally retargets the current thread instead of no-oping.
    // Callers must not bypass this helper to change the same worker thread's
    // current device, unless they restore it before returning to pinned work.
    // The default stream is restored on every call because same-device tasks
    // may follow work that temporarily changed PyTorch's current stream.
    detail::setCurrentThreadDeviceIfNeededImpl(
        device_id,
        current_device,
        [](int device) { setCurrentThreadDeviceContext(device); },
        [](int device) { setCurrentThreadDefaultStream(device); });
#else
    // CPU-only builds intentionally no-op; production cache-store builds pin to a GPU backend.
#endif
}

}  // namespace rtp_llm
