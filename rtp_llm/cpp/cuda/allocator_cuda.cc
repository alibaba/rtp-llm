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

Allocator<AllocatorType::CUDA>::Allocator(int device_id): PurePointerCudaAllocator(device_id), IVirtualMemAllocator() {
    // get GPU Compute Capability
    check_cuda_value(cudaGetDevice(&device_id));

    cudaDeviceProp props;
    check_cuda_value(cudaGetDeviceProperties(&props, device_id));

    // enable vmem allocation only on sm80+ (A100, etc.)
    auto sm_code = props.major * 10 + props.minor;
    _enable_virtual_mem_allocation =
        (sm_code == 80 || sm_code == 87 || sm_code == 90 || sm_code == 100 || sm_code == 120);

    vmem_allocations_ = std::make_unique<std::unordered_map<CUdeviceptr, VmemBlock>>();
}

Allocator<AllocatorType::CUDA>::~Allocator() {
    destroy();
}

void* Allocator<AllocatorType::CUDA>::mallocPhysical(size_t size) {
    RTP_LLM_LOG_DEBUG("malloc physical memory with size %lu\n", size);

    // malloc memory via super class
    auto address = malloc(size);
    if (!address) {
        return nullptr;
    }

    if (_enable_virtual_mem_allocation) {
        // change control block attribute
        CUdeviceptr                 dptr = reinterpret_cast<CUdeviceptr>(address);
        std::lock_guard<std::mutex> lock(lock_);
        auto                        it = vmem_allocations_->find(dptr);
        if (it == vmem_allocations_->end()) {
            RTP_LLM_LOG_ERROR("Unexpected allocation, pointer mapping missing.");
            return address;
        }
        auto& block  = it->second;
        block.pin    = true;
        block.mapped = true;
    }

    return address;
}

void* Allocator<AllocatorType::CUDA>::doMalloc(size_t size) {
    RTP_LLM_LOG_DEBUG("Malloc virtual memory with size %lu\n", size);
    std::lock_guard<std::mutex> lock(lock_);

    // if sm80+, enable virtual memory control
    if (_enable_virtual_mem_allocation) {

        size_t              granularity = 0;
        CUmemAllocationProp prop{};
        prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id   = device_id_;

        check_cuda_value(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

        if (granularity > 0) {
            const size_t padded_size = (size + granularity - 1) / granularity * granularity;

            RTP_LLM_LOG_DEBUG("Malloc virtual memory with padded size %lu\n", padded_size);

            // 1. 先保留虚拟地址
            CUdeviceptr dptr = 0;
            check_cuda_value(cuMemAddressReserve(&dptr, padded_size, 0, 0, 0));

            // 2. 创建物理显存
            CUmemGenericAllocationHandle handle{};
            check_cuda_value(cuMemCreate(&handle, padded_size, &prop, 0));

            // 3. 映射
            check_cuda_value(cuMemMap(dptr, padded_size, 0, handle, 0));

            // 4. 设置访问权限
            CUmemAccessDesc access{};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id   = device_id_;
            access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            check_cuda_value(cuMemSetAccess(dptr, padded_size, &access, 1));

            (*vmem_allocations_)[dptr] = {false, true, padded_size, handle};
            return reinterpret_cast<void*>(dptr);

        } else {
            RTP_LLM_LOG_ERROR("Get system granularity failed\n");
            return nullptr;
        }

        // for sm70, allocate physical momory instead.
    } else {
        void* ptr;

        RTP_LLM_LOG_DEBUG("Malloc physical memory with size %lu(legacy method)\n", size);
        check_cuda_value(cudaMalloc(&ptr, size));
        return ptr;
    }
}

void* Allocator<AllocatorType::CUDA>::doMallocSync(size_t size) {
    return doMalloc(size);
}

void Allocator<AllocatorType::CUDA>::doFree(void* address) {
    if (!address) {
        RTP_LLM_LOG_WARNING("Try to free an empty pointer\n");
        return;
    }

    std::lock_guard<std::mutex> lock(lock_);

    // if sm80+, enable virtual memory control
    if (_enable_virtual_mem_allocation) {

        CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(address);
        auto        it   = vmem_allocations_->find(dptr);
        if (it == vmem_allocations_->end()) {
            RTP_LLM_LOG_ERROR("Free Pointer Failed, Pointer is not managed by this alloctor %p\n", address);
            return;
        }

        RTP_LLM_LOG_DEBUG("Vmem allocator free pointer %p\n", address);
        // tmp sync to avoid memory free before kernel run. cudaFree will not perform any implicit synchronization when
        // the pointer was allocated with cudaMallocAsync or cudaMallocFromPoolAsync
        check_cuda_value(cudaStreamSynchronize(stream_));
        const auto& block = it->second;

        if (block.mapped) {
            check_cuda_value(cuMemUnmap(dptr, block.size));
            check_cuda_value(cuMemRelease(block.handle));
        }

        check_cuda_value(cuMemAddressFree(dptr, block.size));
        vmem_allocations_->erase(dptr);

        RTP_LLM_LOG_DEBUG("Vmem allocator free pointer %p successfully\n", address);

    } else {

        check_cuda_value(cudaFree(address));
        RTP_LLM_LOG_DEBUG("Vmem allocator free pointer %p successfully(legacy method)\n", address);
    }
    return;
}

void Allocator<AllocatorType::CUDA>::unmap() {
    std::lock_guard<std::mutex> lock(lock_);

    // if sm80+, enable virtual memory control
    if (_enable_virtual_mem_allocation) {
        RTP_LLM_LOG_INFO("Vmem allocator unmap all allocated buffer\n");

        for (auto& [dptr, block] : *vmem_allocations_) {

            if (block.pin || !block.mapped) {
                continue;
            }
            RTP_LLM_LOG_INFO("Vmem allocator unmap %p[%lu]\n", dptr, block.size);

            // 1. 解除映射
            check_cuda_value(cuMemUnmap(dptr, block.size));
            // 2. 释放对应的物理显存
            check_cuda_value(cuMemRelease(block.handle));
            block.mapped = false;
        }
    } else {
        throw std::runtime_error("Unmap method need more advanced nvidia card(sm80+).");
    }
}

void Allocator<AllocatorType::CUDA>::map() {
    std::lock_guard<std::mutex> lock(lock_);
    // if sm80+, enable virtual memory control
    if (_enable_virtual_mem_allocation) {
        RTP_LLM_LOG_INFO("Vmem allocator map all allocated buffer\n");

        for (auto& [dptr, block] : *vmem_allocations_) {

            if (block.pin || block.mapped) {
                continue;
            }

            RTP_LLM_LOG_INFO("Vmem allocator map %p[%lu]\n", dptr, block.size);

            size_t              padded_size = block.size;  // 沿用之前对齐后的大小
            CUmemAllocationProp prop{};
            prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id   = device_id_;

            // 1. 重新创建一块新的物理显存句柄
            CUmemGenericAllocationHandle new_handle{};
            check_cuda_value(cuMemCreate(&new_handle, padded_size, &prop, 0));

            // 2. 重新映射到新物理显存
            check_cuda_value(cuMemMap(dptr, padded_size, 0, new_handle, 0));

            // 3. 重新设置访问权限
            CUmemAccessDesc access{};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id   = device_id_;
            access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            check_cuda_value(cuMemSetAccess(dptr, padded_size, &access, 1));

            // 4. 更新 mapping 中的 handle
            block.handle = new_handle;

            block.mapped = true;
        }
    } else {
        throw std::runtime_error("Unmap method need more advanced nvidia card(sm80+).");
    }
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
