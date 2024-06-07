#pragma once
#include "src/fastertransformer/core/allocator.h"

namespace fastertransformer {

template<>
class Allocator<AllocatorType::ROCM> : public TypedAllocator<AllocatorType::ROCM> {
public:
    Allocator() {}
    ~Allocator() {}

    MemoryType memoryType() const {
        return MEMORY_CPU;
    }

    void* malloc(size_t size) {
        return nullptr;
    };

    void  free(void** ptr) {
        *ptr = nullptr;
    };

    // not expected to be called
    void* reMalloc(void* ptr, size_t size) {
        abort();
    };
};


} // namespace fastertransformer

