#include "src/fastertransformer/core/cpu_allocator.h"

namespace fastertransformer {

void* Allocator<AllocatorType::CPU>::malloc(size_t size) {
    return std::malloc(size);
}

void  Allocator<AllocatorType::CPU>::free(void** ptr) {
    std::free(*ptr);
    *ptr = nullptr;
}

// these two methods are not expected to be called
void* Allocator<AllocatorType::CPU>::reMalloc(void* ptr, size_t size) {
    assert(false);
}


}  // namespace fastertransformer
