#pragma once

#include "src/fastertransformer/utils/allocator.h"

#ifdef TORCH_CUDA
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include "torch/extension.h"
#include <memory>
#endif

namespace fastertransformer {

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
    Allocator(int device_id): device_id_(device_id) {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        pointer_mapping_ = new std::unordered_map<void*, size_t>();
#if defined(CUDA_MEMORY_POOL_DISABLED)
        FT_LOG_WARNING(
            "Async cudaMalloc/Free is not supported before CUDA 11.2. Using Sync cudaMalloc/Free."
            "Note this may lead to hang with NCCL kernels launched in parallel; if so, try NCCL_LAUNCH_MODE=GROUP");
#else
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
#endif
    }

    virtual ~Allocator() {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        while (!pointer_mapping_->empty()) {
            free((void**)(&pointer_mapping_->begin()->first));
        }
        delete pointer_mapping_;
    }

    void* malloc(size_t size, const bool is_set_zero = true, bool is_host = false) {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        if (size == 0) {
            return nullptr;
        }
        void* ptr      = nullptr;
        int   o_device = 0;

        check_cuda_error(getSetDevice(device_id_, &o_device));
        if (is_host) {
            check_cuda_error(cudaMallocHost(&ptr, (size_t)(ceil(size / 32.)) * 32));
        } else {
#if defined(CUDA_MEMORY_POOL_DISABLED)
            check_cuda_error(cudaMalloc(&ptr, (size_t)(ceil(size / 32.)) * 32));
#else
            check_cuda_error(cudaMallocAsync(&ptr, (size_t)(ceil(size / 32.)) * 32, stream_));
#endif
        }
        if (is_set_zero) {
            check_cuda_error(cudaMemsetAsync(ptr, 0, (size_t)(ceil(size / 32.)) * 32, stream_));
        }
        check_cuda_error(getSetDevice(o_device));
        FT_LOG_DEBUG("malloc buffer %p with size %ld", ptr, size);

        pointer_mapping_->insert({getAddress(ptr), size});

        return ptr;
    }

    void free(void** ptr, bool is_host = false) const {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        void* address = getAddress(*ptr);
        if (*ptr != nullptr) {
            int o_device = 0;
            if (pointer_mapping_->count(address)) {
                FT_LOG_DEBUG("Free buffer %p", address);
                check_cuda_error(getSetDevice(device_id_, &o_device));
                if (is_host) {
                    check_cuda_error(cudaFreeHost(*ptr));
                } else {
#if defined(CUDA_MEMORY_POOL_DISABLED)
                    check_cuda_error(cudaFree(*ptr));
#else
                    check_cuda_error(cudaFreeAsync(*ptr, stream_));
                    cudaStreamSynchronize(stream_);
#endif
                }
                check_cuda_error(getSetDevice(o_device));
                pointer_mapping_->erase(address);
            } else {
                FT_LOG_WARNING("pointer_mapping_ does not have information of ptr at %p.", address);
            }
        }
        *ptr = nullptr;
        return;
    }

    void memSet(void* ptr, const int val, const size_t size) {
        check_cuda_error(cudaMemsetAsync(ptr, val, size, stream_));
    }
};

#ifdef TORCH_CUDA
template<>
class Allocator<AllocatorType::TH>: public ICudaAllocator {
    std::unordered_map<void*, torch::Tensor>* pointer_mapping_;

    bool isExist(void* address) const {
        return pointer_mapping_->count(address) > 0;
    }
    ReallocType isReMalloc(void* address, size_t size) const {
        FT_CHECK(isExist(address));
        size_t current_buffer_size = 1;
        for (int i = 0; i < pointer_mapping_->at(address).dim(); i++) {
            current_buffer_size *= pointer_mapping_->at(address).size(i);
        }
        FT_LOG_DEBUG(
            "current_buffer_size: %d, original buffer: %p, new buffer: %d", current_buffer_size, address, size);
        if (current_buffer_size < size) {
            return ReallocType::INCREASE;
        } else if (current_buffer_size == size) {
            return ReallocType::REUSE;
        } else {
            return ReallocType::DECREASE;
        }
    }

public:
    Allocator() {
        pointer_mapping_ = new std::unordered_map<void*, torch::Tensor>();
    }

    void* malloc(size_t size, const bool is_set_zero = true, bool is_host = false) {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        int64_t       buf_size = static_cast<int64_t>(ceil(size / 32.)) * 32;
        torch::Tensor buf;
        if (is_host) {
            buf = torch::empty({buf_size}, torch::dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
        } else {
            buf = torch::empty({buf_size}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
        }
        void* ptr = buf.data_ptr();
        if (is_set_zero) {
            cudaMemsetAsync(ptr, 0, buf_size, stream_);
        }
        FT_LOG_DEBUG("malloc buffer %p with size %ld", ptr, buf_size);
        pointer_mapping_->insert({getAddress(ptr), buf});
        return ptr;
    }

    void free(void** ptr, bool is_host = false) const {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        void* address = getAddress(*ptr);
        pointer_mapping_->erase(address);
        *ptr = nullptr;
        return;
    }

    virtual ~Allocator() {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        while (!pointer_mapping_->empty()) {
            void* ptr = pointer_mapping_->begin()->second.data_ptr();
            free((void**)(&ptr));
        }
        pointer_mapping_->clear();
        delete pointer_mapping_;
    }

    void memSet(void* ptr, const int val, const size_t size) {
        check_cuda_error(cudaMemsetAsync(ptr, val, size, stream_));
    }
};
#endif

}  // namespace fastertransformer
