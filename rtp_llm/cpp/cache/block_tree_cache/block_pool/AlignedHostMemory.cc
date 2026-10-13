#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/AlignedHostMemory.h"

#include <exception>

#include "rtp_llm/cpp/utils/AssertUtils.h"

#if USING_CUDA
#include <cuda_runtime.h>
#endif

namespace rtp_llm {

AlignedHostMemory::AlignedHostMemory(size_t usable_bytes, size_t alignment, const std::string& allocation_name) {
#if USING_CUDA
    // These long-lived pools must not be rounded up to power-of-two sizes by
    // PyTorch's caching host allocator, which can exceed the configured budget.
    void*      allocation = nullptr;
    const auto status     = cudaHostAlloc(&allocation, usable_bytes + alignment, cudaHostAllocDefault);
    if (status != cudaSuccess) {
        RTP_LLM_FAIL("allocate pinned host memory failed, allocation=%s usable_bytes=%zu error=%s",
                     allocation_name.c_str(),
                     usable_bytes,
                     cudaGetErrorString(status));
    }
    pinned_allocation_.reset(allocation, [](void* ptr) { cudaFreeHost(ptr); });
    const auto raw_base = reinterpret_cast<uintptr_t>(allocation);
#else
    try {
        backing_ = torch::empty({static_cast<int64_t>(usable_bytes + alignment)},
                                torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
    } catch (const std::exception& e) {
        RTP_LLM_FAIL("allocate pinned host memory failed, allocation=%s usable_bytes=%zu error=%s",
                     allocation_name.c_str(),
                     usable_bytes,
                     e.what());
    }
    RTP_LLM_CHECK_WITH_INFO(
        backing_.is_pinned(), "host allocation [%s] must use pinned CPU memory", allocation_name.c_str());

#if USING_ROCM
    // Host/staging blocks may later be copied to GPU caches consumed by Gluon
    // PA. Keep unused lanes finite, matching device-pool initialization.
    backing_.zero_();
#endif

    const auto raw_base = reinterpret_cast<uintptr_t>(backing_.data_ptr<uint8_t>());
#endif
    data_ = reinterpret_cast<uint8_t*>((raw_base + alignment - 1) / alignment * alignment);
}

uint8_t* AlignedHostMemory::data() const {
    return data_;
}

}  // namespace rtp_llm
