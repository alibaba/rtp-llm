#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <optional>

#include <cutlass/version.h>
#include <deep_jit/backend/cuda/backend.hpp>

namespace deep_gemm {

using GroupedJitRuntime = deep_jit::Runtime<deep_jit::CUDA>;

class GroupedJit {
public:
    void init(deep_jit::Config new_config) {
        DJ_HOST_ASSERT(not config, "grouped JIT is already initialized");
        config.emplace(std::move(new_config));
    }

    GroupedJitRuntime* operator->() {
        // The pybind launcher holds the GIL when resolving the device runtime.
        // Device pointers remain stable while DeepJIT releases the GIL to compile.
        DJ_HOST_ASSERT(config.has_value(), "grouped JIT has no configured source paths");
        int device_index = 0;
        DJ_CUDA_RUNTIME_CHECK(cudaGetDevice(&device_index));
        const auto it = devices.find(device_index);
        if (it != devices.end())
            return it->second.get();

        auto runtime = std::make_unique<GroupedJitRuntime>(*config);
        const auto capability = runtime->device.get_arch_pair();
        DJ_HOST_ASSERT(capability.first == 12 and capability.second == 0, "grouped JIT requires SM120");
        auto& options = runtime->default_compiler_options;
        options.nvcc_flags->emplace_back("--diag-suppress=39,161,174,177,186,940");
        options.nvcc_flags->emplace_back("--compiler-options=-Wno-deprecated-declarations,-Wno-abi");
        options.arch = "120f";
        options.nvcc_flags->emplace_back("-gencode=arch=compute_120f,code=sm_120f");
        return devices.emplace(device_index, std::move(runtime)).first->second.get();
    }

private:
    std::optional<deep_jit::Config> config;
    std::map<int, std::unique_ptr<GroupedJitRuntime>> devices;
};

inline GroupedJit jit;

}  // namespace deep_gemm
