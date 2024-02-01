#pragma once
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/cuda/cuda_utils.h"

#if defined(CUDART_VERSION) && CUDART_VERSION < 11020
#define CUDA_MEMORY_POOL_DISABLED
#endif

namespace fastertransformer{

class ICudaAllocator: virtual public IAllocator {
public:
    ICudaAllocator() {}

    MemoryType memoryType() const override {
        return MEMORY_GPU;
    }

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
class Allocator<AllocatorType::CUDA>: public ICudaAllocator, public TypedAllocator<AllocatorType::CUDA> {
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

} // namespace fastertransformer

