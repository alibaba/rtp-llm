#include "src/fastertransformer/core/cpu_allocator.h"

namespace fastertransformer {

void* Allocator<AllocatorType::CPU>::malloc(size_t size, const bool is_set_zero) {
    assert(!is_set_zero);
    return std::malloc(size);
}

void  Allocator<AllocatorType::CPU>::free(void** ptr) const {
    std::free(*ptr);
    *ptr = nullptr;
}

// these two methods are not expected to be called
void* Allocator<AllocatorType::CPU>::reMalloc(void* ptr, size_t size, const bool is_set_zero) {
    assert(false);
}

void  Allocator<AllocatorType::CPU>::memSet(void* ptr, const int val, const size_t size) const {
    assert(false);
}

}  // namespace fastertransformer
