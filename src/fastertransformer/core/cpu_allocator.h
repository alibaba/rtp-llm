#pragma once
#include "src/fastertransformer/core/allocator.h"
#include <cassert>

namespace fastertransformer {

template<>
class Allocator<AllocatorType::CPU> : public TypedAllocator<AllocatorType::CPU> {
public:
    Allocator() {}
    ~Allocator() {}

    MemoryType memoryType() const {
        return MEMORY_CPU;
    }

    void* malloc(size_t size);
    void  free(void** ptr);
    void* reMalloc(void* ptr, size_t size);
};


} // namespace fastertransformer

