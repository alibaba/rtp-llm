#pragma once

#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/rocm/hip_utils.h"
#include <hip/hip_runtime.h>

namespace fastertransformer {

template<AllocatorType AType, MemoryType MType, hipError_t (*Alloc)(void**, size_t), hipError_t (*Free)(void*)>
class AllocatorT: public TypedAllocator<AType> {
public:
    AllocatorT() {}
    ~AllocatorT() {
        FT_LOG_INFO("rocm allocator destroyed"); /* TODO(rocm): Free all memory? */
    }

    MemoryType memoryType() const {
        return MType;
    }

    void* malloc(size_t size) {
        void* ptr = nullptr;
        HIP_CHECK(Alloc(&ptr, size));
        return ptr;
    };

    void free(void** ptr) {
        HIP_CHECK(Free(*ptr));
        *ptr = nullptr;
    };

    // not expected to be called
    void* reMalloc(void* ptr, size_t size) {
        FT_LOG_ERROR("rocm allocator doesn't support reMalloc");
        fflush(stdout);
        abort();
    };
};

template<>
class Allocator<AllocatorType::ROCM>: public AllocatorT<AllocatorType::ROCM, MEMORY_GPU, hipMalloc, hipFree> {};

static inline hipError_t hipHostMalloc(void** ptr, size_t size) {
    return ::hipHostMalloc(ptr, size, 0);
}

template<>
class Allocator<AllocatorType::ROCM_HOST>:
    public AllocatorT<AllocatorType::ROCM_HOST, MEMORY_CPU_PINNED, hipHostMalloc, hipHostFree> {};

}  // namespace fastertransformer
