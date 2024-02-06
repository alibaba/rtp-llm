#pragma once

#include <unordered_map>
#include <vector>
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/utils/logger.h"

namespace fastertransformer {


void* IAllocator::reMalloc(void* ptr, size_t size, const bool is_set_zero, bool is_host) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size              = ((size + 31) / 32) * 32;  // make the buffer align with 32 bytes
    void* void_ptr    = (void*)ptr;
    void* ptr_address = void_ptr;
    if (isExist(ptr_address)) {
        ReallocType realloc_type = isReMalloc(ptr_address, size);
        if (realloc_type == ReallocType::INCREASE) {
            FT_LOG_DEBUG("ReMalloc the buffer %p since it is too small.", void_ptr);
            free((void**)(&void_ptr), is_host);
            return malloc(size, is_set_zero, is_host);
        } else if (realloc_type == ReallocType::DECREASE) {
            FT_LOG_DEBUG("ReMalloc the buffer %p to release unused memory to memory pools.", void_ptr);
            free((void**)(&void_ptr), is_host);
            return malloc(size, is_set_zero, is_host);
        } else {
            FT_LOG_DEBUG("Reuse original buffer %p with size %d and do nothing for reMalloc.", void_ptr, size);
            if (is_set_zero) {
                memSet(void_ptr, 0, size);
            }
            return void_ptr;
        }
    } else {
        FT_LOG_DEBUG("Cannot find buffer %p, mallocing new one.", void_ptr);
        return malloc(size, is_set_zero, is_host);
    }
}

}  // namespace fastertransformer
