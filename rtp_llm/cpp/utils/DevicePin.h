#pragma once

#include <stdexcept>
#include <string>
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

template<typename SetDeviceContext, typename SetDefaultStream, typename GetCurrentDevice>
inline void setCurrentThreadDeviceIfNeededImpl(int                device_id,
                                               int&               current_device,
                                               SetDeviceContext&& set_device_context,
                                               SetDefaultStream&& set_default_stream,
                                               GetCurrentDevice&& get_current_device) {
    if (device_id < 0) {
        return;
    }

    bool need_set_device = current_device != device_id;
    if (!need_set_device) {
        need_set_device = std::forward<GetCurrentDevice>(get_current_device)() != device_id;
    }
    if (need_set_device) {
        std::forward<SetDeviceContext>(set_device_context)(device_id);
    }

    std::forward<SetDefaultStream>(set_default_stream)(device_id);
    // Cache only fully successful backend setup so failures still retry later.
    current_device = device_id;
}

template<typename SetDeviceContext, typename SetDefaultStream>
inline void setCurrentThreadDeviceIfNeededImpl(int                device_id,
                                               int&               current_device,
                                               SetDeviceContext&& set_device_context,
                                               SetDefaultStream&& set_default_stream) {
    detail::setCurrentThreadDeviceIfNeededImpl(device_id,
                                               current_device,
                                               std::forward<SetDeviceContext>(set_device_context),
                                               std::forward<SetDefaultStream>(set_default_stream),
                                               [&current_device]() { return current_device; });
}

}  // namespace detail

inline void setCurrentThreadDeviceContext(int device_id) {
    if (device_id < 0) {
        return;
    }

#if USING_CUDA
    auto err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaSetDevice(" + std::to_string(device_id) + ") failed: " + cudaGetErrorString(err));
    }
    at::cuda::set_device(device_id);
#elif USING_ROCM
    auto err = hipSetDevice(device_id);
    if (err != hipSuccess) {
        throw std::runtime_error("hipSetDevice(" + std::to_string(device_id) + ") failed: " + hipGetErrorString(err));
    }
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

inline int getCurrentThreadDeviceContext() {
#if USING_CUDA
    int  device_id = -1;
    auto err       = cudaGetDevice(&device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDevice failed: ") + cudaGetErrorString(err));
    }
    return device_id;
#elif USING_ROCM
    int  device_id = -1;
    auto err       = hipGetDevice(&device_id);
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("hipGetDevice failed: ") + hipGetErrorString(err));
    }
    return device_id;
#else
    return -1;
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
    // Cache hits still verify the actual runtime device so a direct
    // cudaSetDevice/hipSetDevice from shared worker code is corrected here.
    // The default stream is restored on every call because same-device tasks
    // may follow work that temporarily changed PyTorch's current stream.
    detail::setCurrentThreadDeviceIfNeededImpl(
        device_id,
        current_device,
        [](int device) { setCurrentThreadDeviceContext(device); },
        [](int device) { setCurrentThreadDefaultStream(device); },
        []() { return getCurrentThreadDeviceContext(); });
#else
    // CPU-only builds intentionally no-op; production cache-store builds pin to a GPU backend.
#endif
}

}  // namespace rtp_llm
