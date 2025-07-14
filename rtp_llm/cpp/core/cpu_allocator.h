#pragma once
#include "rtp_llm/cpp/core/allocator.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

template<>
class Allocator<AllocatorType::CPU>: public TypedAllocator<AllocatorType::CPU> {
public:
    Allocator() {}
    ~Allocator() {}

    MemoryType memoryType() const {
        return MEMORY_CPU;
    }

    void* malloc(size_t size);
    void* mallocSync(size_t size);
    void  free(void** ptr);
    void* reMalloc(void* ptr, size_t size);
};

}  // namespace rtp_llm
