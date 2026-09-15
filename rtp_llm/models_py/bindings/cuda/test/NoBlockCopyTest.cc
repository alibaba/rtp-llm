#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <gtest/gtest.h>
#include <thread>

namespace rtp_llm {
namespace {

void checkDriver(CUresult result) {
    TORCH_CHECK(result == CUDA_SUCCESS, "CUDA driver error: ", static_cast<int>(result));
}

// Unlike cudaMalloc memory, sleep's VMM storage is accessible only on its
// mapped device. Using a different device's copy stream must not be hidden
// by the driver's ordinary cross-device cudaMemcpy support.
torch::Tensor makeVmmTensor(int device, size_t bytes) {
    c10::cuda::CUDAGuard guard(device);
    CUmemAllocationProp  prop{};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = device;
    size_t granularity = 0;
    checkDriver(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    const size_t                 size = ((bytes + granularity - 1) / granularity) * granularity;
    CUdeviceptr                  ptr{};
    CUmemGenericAllocationHandle handle{};
    checkDriver(cuMemAddressReserve(&ptr, size, 0, 0, 0));
    checkDriver(cuMemCreate(&handle, size, &prop, 0));
    checkDriver(cuMemMap(ptr, size, 0, handle, 0));
    checkDriver(cuMemRelease(handle));
    CUmemAccessDesc access{};
    access.location = prop.location;
    access.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    checkDriver(cuMemSetAccess(ptr, size, &access, 1));
    return torch::from_blob(
        reinterpret_cast<void*>(ptr),
        {static_cast<int64_t>(bytes)},
        [ptr, size](void*) {
            (void)cuMemUnmap(ptr, size);
            (void)cuMemAddressFree(ptr, size);
        },
        torch::TensorOptions().device(torch::Device(torch::kCUDA, device)).dtype(torch::kUInt8));
}

TEST(NoBlockCopyTest, VmmCopiesOnFreshAndReusedRpcThreads) {
    if (at::cuda::device_count() < 2) {
        GTEST_SKIP() << "The lazy-device and cross-device stream regression requires two GPUs";
    }
    ASSERT_FALSE(c10::cuda::hasPrimaryContext(0));
    auto gpu = makeVmmTensor(1, 1024 * 1024);
    gpu.fill_(37);
    auto host = torch::empty_like(gpu, torch::TensorOptions().device(torch::kCPU).pinned_memory(true));
    {
        c10::cuda::CUDAGuard guard(1);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    ASSERT_FALSE(c10::cuda::hasPrimaryContext(0));

    std::exception_ptr failure;
    std::thread        worker([&]() {
        try {
            // A from_blob/temporary guard on a new RPC thread leaves CUDA on
            // device 1 while PyTorch remembers the uninitialized device 0.
            { c10::cuda::CUDAGuard guard(1); }
            int runtime_device = -1;
            TORCH_CHECK(cudaGetDevice(&runtime_device) == cudaSuccess);
            TORCH_CHECK(runtime_device == 1 && c10::cuda::current_device() == 0);
            execNoBlockCopy(MultiCopyParams{{host}, {gpu}});
            TORCH_CHECK(host.eq(37).all().item<bool>());
            TORCH_CHECK(c10::cuda::current_device() == 0);
            TORCH_CHECK(!c10::cuda::hasPrimaryContext(0));

            // Reusing the same worker for another device must not reuse the
            // first device's thread-local stream. Check both copy directions.
            for (const int device : {0, 1, 0, 1}) {
                auto other = makeVmmTensor(device, host.nbytes());
                host.fill_(53 + device);
                execNoBlockCopy(MultiCopyParams{{other}, {host}});
                host.zero_();
                execNoBlockCopy(MultiCopyParams{{host}, {other}});
                TORCH_CHECK(host.eq(53 + device).all().item<bool>());
#if CUDART_VERSION >= 12080
                host.zero_();
                TORCH_CHECK(execBatchedMemoryCopy({{{host.data_ptr(), other.data_ptr(), host.nbytes()}}, device}));
                TORCH_CHECK(host.eq(53 + device).all().item<bool>());
#endif
                host.zero_();
                StagedMemoryCopyParams staged;
                staged.host_base    = host.data_ptr();
                staged.host_bytes   = host.nbytes();
                staged.tiles        = {{other.data_ptr(), 0, host.nbytes()}};
                staged.device_index = device;
                staged.direction    = StagedMemoryCopyDirection::D2H;
                TORCH_CHECK(execStagedMemoryCopy(staged));
                TORCH_CHECK(host.eq(53 + device).all().item<bool>());
            }
        } catch (...) {
            failure = std::current_exception();
        }
    });
    worker.join();
    ASSERT_NO_THROW({
        if (failure) {
            std::rethrow_exception(failure);
        }
    });
}

}  // namespace
}  // namespace rtp_llm
