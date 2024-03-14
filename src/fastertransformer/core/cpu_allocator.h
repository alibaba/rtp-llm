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

    void* malloc(size_t size, const bool is_set_zero = false);
    void  free(void** ptr) const;
    void* reMalloc(void* ptr, size_t size, const bool is_set_zero = false);
    void  memSet(void* ptr, const int val, const size_t size) const;

};


} // namespace fastertransformer

