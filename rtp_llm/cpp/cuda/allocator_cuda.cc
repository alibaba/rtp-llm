#include "rtp_llm/cpp/cuda/allocator_cuda.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <mutex>

namespace rtp_llm {

void* ICudaAllocator::reMalloc(void* ptr, size_t size) {
    size              = ((size + 127) / 128) * 128;  // make the buffer align with 128 bytes
    void* void_ptr    = (void*)ptr;
    void* ptr_address = void_ptr;
    if (isExist(ptr_address)) {
        ReallocType realloc_type = isReMalloc(ptr_address, size);
        if (realloc_type == ReallocType::INCREASE) {
            RTP_LLM_LOG_DEBUG("ReMalloc the buffer %p since it is too small.", void_ptr);
            free((void**)(&void_ptr));
            return malloc(size);
        } else if (realloc_type == ReallocType::DECREASE) {
            RTP_LLM_LOG_DEBUG("ReMalloc the buffer %p to release unused memory to memory pools.", void_ptr);
            free((void**)(&void_ptr));
            return malloc(size);
        } else {
            RTP_LLM_LOG_DEBUG("Reuse original buffer %p with size %d and do nothing for reMalloc.", void_ptr, size);
            return void_ptr;
        }
    } else {
        RTP_LLM_LOG_DEBUG("Cannot find buffer %p, mallocing new one.", void_ptr);
        return malloc(size);
    }
}

PurePointerCudaAllocator::PurePointerCudaAllocator(int device_id):
    ICudaAllocator(device_id), pointer_mapping_(new std::unordered_map<void*, size_t>) {}

PurePointerCudaAllocator::~PurePointerCudaAllocator() {}

void PurePointerCudaAllocator::destroy() {
    while (!pointer_mapping_->empty()) {
        auto it  = pointer_mapping_->begin();
        auto ptr = it->first;
        free(&ptr);
    }
}

bool PurePointerCudaAllocator::isExist(void* address) const {
    return pointer_mapping_->count(address) > 0;
}

ReallocType PurePointerCudaAllocator::isReMalloc(void* address, size_t size) const {
    RTP_LLM_CHECK(isExist(address));
    if (pointer_mapping_->at(address) < size) {
        return ReallocType::INCREASE;
    } else if (pointer_mapping_->at(address) == size) {
        return ReallocType::REUSE;
    } else {
        return ReallocType::DECREASE;
    }
}

void* PurePointerCudaAllocator::malloc(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void*                       ptr = doMalloc(size);
    std::lock_guard<std::mutex> lock(lock_);
    pointer_mapping_->insert({ptr, size});
    return ptr;
}

void* PurePointerCudaAllocator::mallocSync(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void*                       ptr = doMallocSync(size);
    std::lock_guard<std::mutex> lock(lock_);
    pointer_mapping_->insert({ptr, size});
    return ptr;
}

void PurePointerCudaAllocator::free(void** ptr) {
    void* address = *ptr;
    if (address) {
        std::lock_guard<std::mutex> lock(lock_);
        RTP_LLM_CHECK_WITH_INFO(
            pointer_mapping_->count(address), "pointer_mapping_ does not have information of ptr at %p", address);
        doFree(address);
        *ptr = nullptr;
        pointer_mapping_->erase(address);
    }
    return;
}

Allocator<AllocatorType::CUDA>::Allocator(int device_id): PurePointerCudaAllocator(device_id) {}

Allocator<AllocatorType::CUDA>::~Allocator() {
    destroy();
}

void* Allocator<AllocatorType::CUDA>::doMalloc(size_t size) {
    void* ptr = nullptr;
    check_cuda_value(cudaMalloc(&ptr, (size_t)(ceil(size / 128.)) * 128));
    return ptr;
}

void* Allocator<AllocatorType::CUDA>::doMallocSync(size_t size) {
    void* ptr = nullptr;
    check_cuda_value(cudaMalloc(&ptr, (size_t)(ceil(size / 128.)) * 128));
    return ptr;
}

void Allocator<AllocatorType::CUDA>::doFree(void* address) {
    // tmp sync to avoid memory free before kernel run. cudaFree will not perform any implicit synchronization when the
    // pointer was allocated with cudaMallocAsync or cudaMallocFromPoolAsync
    cudaStreamSynchronize(stream_);
    check_cuda_value(cudaFree(address));
    return;
}

Allocator<AllocatorType::CUDA_HOST>::Allocator(int device_id): PurePointerCudaAllocator(device_id) {}

Allocator<AllocatorType::CUDA_HOST>::~Allocator() {
    destroy();
}

void* Allocator<AllocatorType::CUDA_HOST>::doMalloc(size_t size) {
    void* ptr = nullptr;
    check_cuda_value(cudaMallocHost(&ptr, (size_t)(ceil(size / 128.)) * 128));
    return ptr;
}

void* Allocator<AllocatorType::CUDA_HOST>::doMallocSync(size_t size) {
    return doMalloc(size);
}

void Allocator<AllocatorType::CUDA_HOST>::doFree(void* address) {
    if (address) {
        check_cuda_value(cudaFreeHost(address));
    }
    return;
}

}  // namespace rtp_llm
