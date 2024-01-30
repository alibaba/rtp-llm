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
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/cuda/cuda_utils.h"
#include <unordered_map>
#include <vector>

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

    virtual void* malloc(size_t size, const bool is_set_zero = true, bool is_host = false) = 0;
    virtual void  free(void** ptr, bool is_host = false) const                             = 0;
    virtual void  memSet(void* ptr, const int val, const size_t size)                      = 0;

    void* reMalloc(void* ptr, size_t size, const bool is_set_zero = false, bool is_host = false);

protected:
    virtual bool        isExist(void* address) const                 = 0;
    virtual ReallocType isReMalloc(void* address, size_t size) const = 0;

    void* getAddress(void* ptr) const {
        return ptr;
    }
};

template<AllocatorType AllocType_>
class Allocator;


class ICudaAllocator: public IAllocator {
public:
    ICudaAllocator() {}

    void setStream(cudaStream_t stream) {
        stream_ = stream;
    }

    cudaStream_t returnStream() {
        return stream_;
    };

protected:
    cudaStream_t                       stream_ = 0;  // initialize as default stream
};

template<>
class Allocator<AllocatorType::CUDA>: public ICudaAllocator {
private:
    const int                          device_id_;
    std::unordered_map<void*, size_t>* pointer_mapping_;

    bool isExist(void* address) const {
        return pointer_mapping_->count(address) > 0;
    }
    ReallocType isReMalloc(void* address, size_t size) const {
        FT_CHECK(isExist(address));
        if (pointer_mapping_->at(address) < size) {
            return ReallocType::INCREASE;
        } else if (pointer_mapping_->at(address) == size) {
            return ReallocType::REUSE;
        } else {
            return ReallocType::DECREASE;
        }
    }

public:
    Allocator(int device_id);

    virtual ~Allocator();

    void* malloc(size_t size, const bool is_set_zero = true, bool is_host = false);

    void free(void** ptr, bool is_host = false) const;

    void memSet(void* ptr, const int val, const size_t size) {
        check_cuda_error(cudaMemsetAsync(ptr, val, size, stream_));
    }
};



}  // namespace fastertransformer
