/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * Memory Allocator
 **/

#pragma once

#include <unordered_map>
#include <vector>

#include "src/fastertransformer/utils/logger.h"

#if defined(CUDART_VERSION) && CUDART_VERSION < 11020
#define CUDA_MEMORY_POOL_DISABLED
#endif

namespace fastertransformer {

enum class AllocatorType {
    CUDA,
    TF,
    TH
};

enum class ReallocType {
    INCREASE,
    REUSE,
    DECREASE,
};

class IAllocator {
public:
    virtual ~IAllocator(){};

    virtual void*        malloc(size_t size, const bool is_set_zero = true, bool is_host = false) = 0;
    virtual void         free(void** ptr, bool is_host = false) const                             = 0;
    virtual void         memSet(void* ptr, const int val, const size_t size)                      = 0;

    template<typename T>
    void* reMalloc(T* ptr, size_t size, const bool is_set_zero = false, bool is_host = false) {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        size              = ((size + 31) / 32) * 32;  // make the buffer align with 32 bytes
        void* void_ptr    = (void*)ptr;
        void* ptr_address = getAddress(void_ptr);
        if (isExist(ptr_address)) {
            ReallocType realloc_type = isReMalloc(ptr_address, size);
            if (realloc_type == ReallocType::INCREASE) {
                FT_LOG_DEBUG("ReMalloc the buffer %p since it is too small.", void_ptr);
                free((void**)(&void_ptr), is_host);
                return malloc(size, is_set_zero, is_host);
            }
#if !defined(CUDA_MEMORY_POOL_DISABLED)
            else if (realloc_type == ReallocType::DECREASE) {
                FT_LOG_DEBUG("ReMalloc the buffer %p to release unused memory to memory pools.", void_ptr);
                free((void**)(&void_ptr), is_host);
                return malloc(size, is_set_zero, is_host);
            }
#endif
            else {
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

protected:
    virtual bool        isExist(void* address) const                 = 0;
    virtual ReallocType isReMalloc(void* address, size_t size) const = 0;

    void* getAddress(void* ptr) const {
        return ptr;
    }
};

template<AllocatorType AllocType_>
class Allocator;


}  // namespace fastertransformer
