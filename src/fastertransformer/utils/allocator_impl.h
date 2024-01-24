#pragma once

#include "src/fastertransformer/utils/allocator.h"

#ifdef GOOGLE_CUDA_
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#endif

#ifdef TORCH_CUDA
#include "torch/extension.h"
#include <memory>
#endif

namespace fastertransformer {

#ifdef GOOGLE_CUDA_
using namespace tensorflow;
template<>
class Allocator<AllocatorType::TF>: public IAllocator {
    OpKernelContext*                               context_;
    std::unordered_map<void*, tensorflow::Tensor>* pointer_mapping_;
    cudaStream_t                                   stream_;

    bool isExist(void* address) const {
        return pointer_mapping_->count(address) > 0;
    }
    ReallocType isReMalloc(void* address, size_t size) const {
        FT_CHECK(isExist(address));
        size_t current_buffer_size = 1;
        for (int i = 0; i < pointer_mapping_->at(address).dims(); i++) {
            current_buffer_size *= pointer_mapping_->at(address).dim_size(i);
        }
        FT_LOG_DEBUG("current_buffer_size: %d, new buffer: %d", current_buffer_size, size);
        if (current_buffer_size < size) {
            return ReallocType::INCREASE;
        } else if (current_buffer_size == size) {
            return ReallocType::REUSE;
        } else {
            return ReallocType::DECREASE;
        }
    }

public:
    Allocator(OpKernelContext* context, cudaStream_t stream): context_(context), stream_(stream) {
        pointer_mapping_ = new std::unordered_map<void*, tensorflow::Tensor>();
    }

    void setStream(cudaStream_t stream) {
        stream_ = stream;
    }

    cudaStream_t returnStream() {
        return stream_;
    };

    void* malloc(size_t size, const bool is_set_zero = true, bool is_host = false) {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        tensorflow::Tensor buf;
        long long int      buf_size = ((long long int)ceil(size / 32.) * 32);
        tensorflow::Status status;
        if (is_host) {
            tensorflow::AllocatorAttributes pinned_allocator;
            pinned_allocator.set_on_host(true);
            pinned_allocator.set_gpu_compatible(true);
            status = context_->allocate_temp(DT_UINT8, TensorShape{buf_size}, &buf, pinned_allocator);
        } else {
            status = context_->allocate_temp(DT_UINT8, TensorShape{buf_size}, &buf);
        }

        if (status != tensorflow::Status::OK()) {
            throw std::runtime_error("TF error: context->allocate_temp failed");
        }

        auto  flat = buf.flat<uint8>();
        void* ptr  = (void*)flat.data();
        if (is_set_zero) {
            cudaMemsetAsync(ptr, 0, buf_size, stream_);
        }
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
        while (!pointer_mapping_->empty()) {
            void* ptr = pointer_mapping_->begin()->second.flat<uint8>().data();
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

#ifdef TORCH_CUDA
template<>
class Allocator<AllocatorType::TH>: public IAllocator {
    std::unordered_map<void*, torch::Tensor>* pointer_mapping_;
    cudaStream_t                              stream_ = 0;

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

    void setStream(cudaStream_t stream) {
        stream_ = stream;
    }

    cudaStream_t returnStream() {
        return stream_;
    };

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
