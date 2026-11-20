#pragma once

#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fmt/format.h>
#include <memory>
#include <type_traits>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>

#include <deep_jit/backend/cuda/driver.hpp>
#include <deep_jit/backend/cuda/options.hpp>
#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/gil.hpp>
#include <deep_jit/utils/no_ref_ptr.hpp>

namespace deep_jit::cuda {

template <typename T>
inline void* kernel_arg_pointer(const T& value) {
    if constexpr (std::is_base_of_v<NoRefPtr, std::decay_t<T>>) {
        return value.ptr;
    } else {
        return const_cast<void*>(static_cast<const void*>(&value));
    }
}

// Immutable CUDA kernel handles with shared ownership. Driver resources are
// unloaded when the last shared owner is destroyed.
class Kernel {
public:
    CUlibrary library_handle{};
    CUfunction kernel_handle{};

    Kernel(const CUlibrary& library_handle, const CUfunction& kernel_handle)
        : library_handle(library_handle), kernel_handle(kernel_handle) {}

    ~Kernel() { unload(); }
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;
    Kernel(Kernel&&) = delete;
    Kernel& operator=(Kernel&&) = delete;

    static std::shared_ptr<Kernel> load(const std::filesystem::path& dir, const Env& env) {
        // Release GIL to let other Python threads run
        GilScopedRelease gil_release;

        // Check existence
        const auto cubin_path = dir / "kernel.cubin";
        if (not std::filesystem::is_regular_file(cubin_path))
            DJ_PANIC("missing CUDA CUBIN: {}", cubin_path.string());

        // Record start time
        const bool debug = env.get<bool>("JIT_DEBUG", false);
        const bool print_load_time = debug or env.get<bool>("JIT_PRINT_LOAD_TIME", false);
        if (debug)
            std::fputs(fmt::format("Loading CUBIN: {}\n", cubin_path.string()).c_str(), stdout);
        const auto start_time = std::chrono::steady_clock::now();

        // Load kernel
        CUlibrary library_handle{};
        CUfunction kernel_handle{};
        DJ_CUDA_DRIVER_CHECK(driver::lazy_cuLibraryLoadFromFile(
            &library_handle, cubin_path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
        try {
            unsigned int num_kernels = 0;
            DJ_CUDA_DRIVER_CHECK(driver::lazy_cuLibraryGetKernelCount(&num_kernels, library_handle));
            if (num_kernels != 1)
                DJ_PANIC("expected exactly one kernel in {}", cubin_path.string());

            CUkernel library_kernel_handle{};
            DJ_CUDA_DRIVER_CHECK(driver::lazy_cuLibraryEnumerateKernels(&library_kernel_handle, 1, library_handle));
            DJ_CUDA_DRIVER_CHECK(driver::lazy_cuKernelGetFunction(&kernel_handle, library_kernel_handle));
        } catch (...) {
            driver::lazy_cuLibraryUnload(library_handle);
            throw;
        }

        // Print and return
        if (print_load_time) {
            const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start_time;
            std::fputs(fmt::format("Load time ({}): {:.2f} ms\n", dir.string(), elapsed.count()).c_str(), stdout);
        }
        return std::make_shared<Kernel>(library_handle, kernel_handle);
    }

    template <typename... Args>
    void launch(const LaunchOptions& launch_options, const Args&... args) const {
        // Release GIL to let other Python threads run
        GilScopedRelease gil_release;

        // Checks
        DJ_HOST_ASSERT(kernel_handle != nullptr, "kernel must be loaded before launch");
        DJ_HOST_ASSERT(launch_options.num_smem_bytes.has_value(), "CUDA dynamic shared-memory size must be specified");
        DJ_HOST_ASSERT(launch_options.grid_dim.has_value(), "CUDA grid dimension must be specified");
        DJ_HOST_ASSERT(launch_options.block_dim.has_value(), "CUDA block dimension must be specified");
        DJ_HOST_ASSERT(launch_options.cluster_dim.has_value(), "CUDA cluster dimension must be specified");
        DJ_HOST_ASSERT(launch_options.cooperative.has_value(), "CUDA cooperative option must be specified");
        DJ_HOST_ASSERT(launch_options.enable_pdl.has_value(), "CUDA PDL option must be specified");
        DJ_HOST_ASSERT(launch_options.nonportable_cluster_size_allowed.has_value(),
                       "CUDA non-portable cluster option must be specified");
        DJ_HOST_ASSERT(*launch_options.num_smem_bytes >= 0, "CUDA dynamic shared-memory size must not be negative");
        DJ_HOST_ASSERT(launch_options.grid_dim->x > 0 and launch_options.grid_dim->y > 0 and launch_options.grid_dim->z > 0,
                       "CUDA grid dimensions must be positive");
        DJ_HOST_ASSERT(launch_options.block_dim->x > 0 and launch_options.block_dim->y > 0 and launch_options.block_dim->z > 0,
                       "CUDA block dimensions must be positive");
        DJ_HOST_ASSERT(launch_options.cluster_dim->x > 0, "CUDA cluster dimension must be positive");
        DJ_HOST_ASSERT(launch_options.cluster_dim->y == 1 and launch_options.cluster_dim->z == 1,
                       "only one-dimensional CUDA clusters are supported");

        // Set **maximum** dynamic shared memory
        if (*launch_options.num_smem_bytes > 0) {
            DJ_CUDA_DRIVER_CHECK(driver::lazy_cuFuncSetAttribute(
                kernel_handle, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                *launch_options.num_smem_bytes));
        }

        // Set non-portable cluster size
        // NOTES: you can not write `if (nonportable_cluster_size_allowed) ...`,
        // this may lead to a bug, e.g., launch with `= true` then `= false`,
        // in such case, the second launch will still use the first attribute
        // But, we still write this, because the launch overhead is important
        // And for `= true`'s side-effects for `= false` can be ignored
        if (*launch_options.nonportable_cluster_size_allowed) {
            DJ_CUDA_DRIVER_CHECK(driver::lazy_cuFuncSetAttribute(
                kernel_handle, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
        }

        // NOTES: please enlarge the array size if you want more attributes
        std::array<CUlaunchAttribute, 3> attributes{};
        unsigned int num_attributes = 0;
        if (*launch_options.cooperative) {
            auto& attribute = attributes[num_attributes ++];
            attribute.id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
            attribute.value.cooperative = 1;
        }
        if (launch_options.cluster_dim->x > 1) {
            auto& attribute = attributes[num_attributes ++];
            attribute.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
            attribute.value.clusterDim.x = launch_options.cluster_dim->x;
            attribute.value.clusterDim.y = 1;
            attribute.value.clusterDim.z = 1;
        }
        if (*launch_options.enable_pdl) {
            auto& attribute = attributes[num_attributes ++];
            attribute.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
            attribute.value.programmaticStreamSerializationAllowed = 1;
        }

        // Set launch config
        void* kernel_arg_ptrs[sizeof...(Args) + 1] = {kernel_arg_pointer(args)..., nullptr};
        CUlaunchConfig config{};
        config.gridDimX = launch_options.grid_dim->x;
        config.gridDimY = launch_options.grid_dim->y;
        config.gridDimZ = launch_options.grid_dim->z;
        config.blockDimX = launch_options.block_dim->x;
        config.blockDimY = launch_options.block_dim->y;
        config.blockDimZ = launch_options.block_dim->z;
        config.sharedMemBytes = *launch_options.num_smem_bytes;
        config.hStream = launch_options.stream
            ? *launch_options.stream
            : at::cuda::getCurrentCUDAStream().stream();
        config.attrs = num_attributes == 0 ? nullptr : attributes.data();
        config.numAttrs = num_attributes;
        DJ_CUDA_DRIVER_CHECK(driver::lazy_cuLaunchKernelEx(
            &config, kernel_handle, sizeof...(Args) == 0 ? nullptr : kernel_arg_ptrs, nullptr));
    }

    void unload() noexcept {
        if (library_handle == nullptr)
            return;

        try {
            DJ_CUDA_DRIVER_CHECK(driver::lazy_cuLibraryUnload(library_handle));
        } catch (...) {
        }
        library_handle = nullptr;
        kernel_handle = nullptr;
    }
};

}  // namespace deep_jit::cuda
