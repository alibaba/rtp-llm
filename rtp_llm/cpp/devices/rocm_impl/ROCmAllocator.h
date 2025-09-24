#pragma once

#include "rtp_llm/cpp/core/allocator.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#include <hip/hip_runtime.h>

namespace rtp_llm {

template<AllocatorType AType, MemoryType MType, hipError_t (*Alloc)(void**, size_t), hipError_t (*Free)(void*)>
class ROCmAllocator: public TypedAllocator<AType> {
public:
    ROCmAllocator() {}
    ~ROCmAllocator() {
        RTP_LLM_LOG_INFO("rocm allocator destroyed"); /* TODO(rocm): Free all memory? */
    }

    MemoryType memoryType() const {
        return MType;
    }

    void* malloc(size_t size) {
        void* ptr = nullptr;
        ROCM_CHECK(Alloc(&ptr, size));
        return ptr;
    };

    void* mallocSync(size_t size) {
        void* ptr = nullptr;
        ROCM_CHECK(Alloc(&ptr, size));
        return ptr;
    };

    void free(void** ptr) {
        ROCM_CHECK(Free(*ptr));
        *ptr = nullptr;
    };

    // not expected to be called
    void* reMalloc(void* ptr, size_t size) {
        /* TODO(rocm): Missing */
        RTP_LLM_LOG_ERROR("rocm allocator doesn't support remalloc");
        abort();
    };
};

template<>
class Allocator<AllocatorType::ROCM>: public ROCmAllocator<AllocatorType::ROCM, MEMORY_GPU, hipMalloc, hipFree> {};

template<>
class Allocator<AllocatorType::ROCM_HOST>:
    public ROCmAllocator<AllocatorType::ROCM_HOST, MEMORY_CPU_PINNED, hipMallocHost, hipHostFree> {};

}  // namespace rtp_llm
