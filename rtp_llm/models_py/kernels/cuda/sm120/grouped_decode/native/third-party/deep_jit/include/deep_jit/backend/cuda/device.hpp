#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include <cuda_runtime.h>

#include <deep_jit/utils/exception.hpp>

namespace deep_jit::cuda {

inline void check_cuda_runtime(const cudaError_t error, const char* expression) {
    if (error != cudaSuccess) {
        DJ_PANIC("{} failed with CUDA error {} ({}): {}",
                 expression,
                 static_cast<int>(error),
                 cudaGetErrorName(error),
                 cudaGetErrorString(error));
    }
}

#ifndef DJ_CUDA_RUNTIME_CHECK
#define DJ_CUDA_RUNTIME_CHECK(expr) ::deep_jit::cuda::check_cuda_runtime((expr), #expr)
#endif

class Device {
    cudaDeviceProp prop{};
    int64_t clock_rate = 0;
    bool initialized = false;

public:
    const cudaDeviceProp& get_prop() {
        if (not initialized) {
            // `cudaFree(nullptr)` is to ensure the current CUDA context exists before later driver API calls
            int device_index = 0;
            DJ_CUDA_RUNTIME_CHECK(cudaGetDevice(&device_index));
            DJ_CUDA_RUNTIME_CHECK(cudaFree(nullptr));
            DJ_CUDA_RUNTIME_CHECK(cudaGetDeviceProperties(&prop, device_index));
            initialized = true;
        }
        return prop;
    }

    int get_num_sms() { return get_prop().multiProcessorCount; }

    int get_num_l2_cache_bytes() { return get_prop().l2CacheSize; }

    int get_num_smem_bytes() { return static_cast<int>(get_prop().sharedMemPerBlockOptin); }

    int64_t get_clock_rate() {
        if (clock_rate == 0) {
            int device_index = 0;
            int rate = 0;
            DJ_CUDA_RUNTIME_CHECK(cudaGetDevice(&device_index));
            DJ_CUDA_RUNTIME_CHECK(cudaDeviceGetAttribute(&rate, cudaDevAttrClockRate, device_index));
            clock_rate = static_cast<int64_t>(rate) * 1000;
        }
        return clock_rate;
    }

    int get_arch_major() { return get_prop().major; }

    int get_arch_minor() { return get_prop().minor; }

    std::pair<int, int> get_arch_pair() { return {get_arch_major(), get_arch_minor()}; }

    std::string get_arch(const bool use_arch_family = true, const bool number_only = false) {
        const auto [major, minor] = get_arch_pair();
        const auto arch = std::to_string(major * 10 + minor);

        // E.g., sm_80, sm_90, sm_100
        if (number_only or major < 9)
            return arch;

        // E.g., sm_90a, sm_100f, sm_100a
        if (major > 9)
            return arch + (use_arch_family ? "f" : "a");
        return arch + "a";
    }
};

}  // namespace deep_jit::cuda
