#include "src/fastertransformer/cuda/allocator_cuda.h"

namespace fastertransformer {

void* ICudaAllocator::reMalloc(void* ptr, size_t size, const bool is_set_zero) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size              = ((size + 31) / 32) * 32;  // make the buffer align with 32 bytes
    void* void_ptr    = (void*)ptr;
    void* ptr_address = void_ptr;
    if (isExist(ptr_address)) {
        ReallocType realloc_type = isReMalloc(ptr_address, size);
        if (realloc_type == ReallocType::INCREASE) {
            FT_LOG_DEBUG("ReMalloc the buffer %p since it is too small.", void_ptr);
            free((void**)(&void_ptr));
            return malloc(size, is_set_zero);
        } else if (realloc_type == ReallocType::DECREASE) {
            FT_LOG_DEBUG("ReMalloc the buffer %p to release unused memory to memory pools.", void_ptr);
            free((void**)(&void_ptr));
            return malloc(size, is_set_zero);
        } else {
            FT_LOG_DEBUG("Reuse original buffer %p with size %d and do nothing for reMalloc.", void_ptr, size);
            if (is_set_zero) {
                memSet(void_ptr, 0, size);
            }
            return void_ptr;
        }
    } else {
        FT_LOG_DEBUG("Cannot find buffer %p, mallocing new one.", void_ptr);
        return malloc(size, is_set_zero);
    }
}


void ICudaAllocator::memSet(void* ptr, const int val, const size_t size) const {
    check_cuda_error(cudaMemsetAsync(ptr, val, size, stream_));
}

// cuda allocator

Allocator<AllocatorType::CUDA>::Allocator(int device_id): PurePointerCudaAllocator(device_id) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    int device_count = 1;
    check_cuda_error(cudaGetDeviceCount(&device_count));
    cudaMemPool_t mempool;
    check_cuda_error(cudaDeviceGetDefaultMemPool(&mempool, device_id));
    cudaMemAccessDesc desc                  = {};
    int               peer_access_available = 0;
    for (int i = 0; i < device_count; i++) {
        if (i == device_id) {
            continue;
        }
        check_cuda_error(cudaDeviceCanAccessPeer(&peer_access_available, device_id, i));
        if (!peer_access_available) {
            FT_LOG_WARNING("Device " + std::to_string(device_id) + " peer access Device " + std::to_string(i)
                            + " is not available.");
            continue;
        }
        desc.location.type = cudaMemLocationTypeDevice;
        desc.location.id   = i;
        desc.flags         = cudaMemAccessFlagsProtReadWrite;
        check_cuda_error(cudaMemPoolSetAccess(mempool, &desc, 1));
    }
    // set memory pool threshold to avoid shrinking the pool
    uint64_t setVal = UINT64_MAX;
    check_cuda_error(cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &setVal));
}

Allocator<AllocatorType::CUDA>::~Allocator() {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    while (!pointer_mapping_->empty()) {
        free((void**)(&pointer_mapping_->begin()->first));
    }
}

void* Allocator<AllocatorType::CUDA>::malloc(size_t size, const bool is_set_zero) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (size == 0) {
        return nullptr;
    }
    void* ptr      = nullptr;
    int   o_device = 0;

    check_cuda_error(getSetDevice(device_id_, &o_device));
    check_cuda_error(cudaMallocAsync(&ptr, (size_t)(ceil(size / 32.)) * 32, stream_));
    if (is_set_zero) {
        check_cuda_error(cudaMemsetAsync(ptr, 0, (size_t)(ceil(size / 32.)) * 32, stream_));
    }
    check_cuda_error(getSetDevice(o_device));
    FT_LOG_DEBUG("malloc buffer %p with size %ld", ptr, size);

    pointer_mapping_->insert({ptr, size});

    return ptr;
}

void Allocator<AllocatorType::CUDA>::free(void** ptr) const {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    void* address = *ptr;
    if (*ptr != nullptr) {
        int o_device = 0;
        if (pointer_mapping_->count(address)) {
            FT_LOG_DEBUG("Free buffer %p", address);
            check_cuda_error(getSetDevice(device_id_, &o_device));
            check_cuda_error(cudaFreeAsync(*ptr, stream_));
            cudaStreamSynchronize(stream_);
            check_cuda_error(getSetDevice(o_device));
            pointer_mapping_->erase(address);
        } else {
            FT_LOG_WARNING("pointer_mapping_ does not have information of ptr at %p.", address);
        }
    }
    *ptr = nullptr;
    return;
}

Allocator<AllocatorType::CUDA_HOST>::Allocator(int device_id): PurePointerCudaAllocator(device_id) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

Allocator<AllocatorType::CUDA_HOST>::~Allocator() {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    while (!pointer_mapping_->empty()) {
        free((void**)(&pointer_mapping_->begin()->first));
    }
}

void* Allocator<AllocatorType::CUDA_HOST>::malloc(size_t size, const bool is_set_zero) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (size == 0) {
        return nullptr;
    }
    void* ptr      = nullptr;
    int   o_device = 0;

    check_cuda_error(getSetDevice(device_id_, &o_device));
    check_cuda_error(cudaMallocHost(&ptr, (size_t)(ceil(size / 32.)) * 32));
    if (is_set_zero) {
        check_cuda_error(cudaMemsetAsync(ptr, 0, (size_t)(ceil(size / 32.)) * 32, stream_));
    }
    check_cuda_error(getSetDevice(o_device));
    FT_LOG_DEBUG("malloc cuda host buffer %p with size %ld", ptr, size);

    pointer_mapping_->insert({ptr, size});

    return ptr;
}

void Allocator<AllocatorType::CUDA_HOST>::free(void** ptr) const {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    void* address = *ptr;
    if (*ptr != nullptr) {
        int o_device = 0;
        if (pointer_mapping_->count(address)) {
            FT_LOG_DEBUG("Free buffer %p", address);
            check_cuda_error(getSetDevice(device_id_, &o_device));
            check_cuda_error(cudaFreeHost(*ptr));
            check_cuda_error(getSetDevice(o_device));
            pointer_mapping_->erase(address);
        } else {
            FT_LOG_WARNING("pointer_mapping_ does not have information of ptr at %p.", address);
        }
    }
    *ptr = nullptr;
    return;
}

}  // namespace fastertransformer
