#pragma once

#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/rocm/hip_utils.h"
#include <hip/hip_runtime.h>

namespace fastertransformer {


template<AllocatorType AType, MemoryType MType,
         hipError_t (*Alloc)(void**, size_t), hipError_t (*Free)(void*)>
class ROCmAllocator: public TypedAllocator<AType> {
public:
    ROCmAllocator() {}
    ~ROCmAllocator() { FT_LOG_INFO("rocm allocator destroyed"); /* TODO(rocm): Free all memory? */ }

    MemoryType memoryType() const {
        return MType;
    }

    void* malloc(size_t size) {
        void* ptr = nullptr;
        (void)Alloc(&ptr, size);
        return ptr;
    };

    void free(void** ptr) {
        (void)Free(*ptr);
        *ptr = nullptr;
    };

    // not expected to be called
    void* reMalloc(void* ptr, size_t size) {
        /* TODO(rocm): Missing */
        FT_LOG_ERROR("rocm allocator doesn't support remalloc");
        abort();
    };
};

template<>
class Allocator<AllocatorType::ROCM>:
    public ROCmAllocator<AllocatorType::ROCM, MEMORY_GPU, hipMalloc, hipFree> {};

template<>
class Allocator<AllocatorType::ROCM_HOST>:
    public ROCmAllocator<AllocatorType::ROCM_HOST, MEMORY_CPU_PINNED, hipMallocHost, hipHostFree> {};

}  // namespace fastertransformer
