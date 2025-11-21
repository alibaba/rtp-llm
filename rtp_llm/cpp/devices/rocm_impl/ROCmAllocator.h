#pragma once

#include "rtp_llm/cpp/core/allocator.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#include <hip/hip_runtime.h>

namespace rtp_llm {

template<AllocatorType AType, MemoryType MType, hipError_t (*Alloc)(void**, size_t), hipError_t (*Free)(void*)>
class ROCmAllocator: public TypedAllocator<AType> {
public:
    ROCmAllocator() {}
    ~ROCmAllocator() {
        RTP_LLM_LOG_INFO("rocm allocator destroyed"); /* TODO(rocm): Free all memory? */
    }

    MemoryType memoryType() const {
        return MType;
    }

    void* malloc(size_t size) {
        void* ptr = nullptr;
        ROCM_CHECK(Alloc(&ptr, size));
        return ptr;
    };

    void* mallocSync(size_t size) {
        void* ptr = nullptr;
        ROCM_CHECK(Alloc(&ptr, size));
        return ptr;
    };

    void free(void** ptr) {
        ROCM_CHECK(Free(*ptr));
        *ptr = nullptr;
    };

    // not expected to be called
    void* reMalloc(void* ptr, size_t size) {
        /* TODO(rocm): Missing */
        RTP_LLM_LOG_ERROR("rocm allocator doesn't support remalloc");
        abort();
    };
};

template<>
class Allocator<AllocatorType::ROCM>:
    public ROCmAllocator<AllocatorType::ROCM, MEMORY_GPU, hipMalloc, hipFree>,
    public IVirtualMemAllocator {

public:
    Allocator(): IVirtualMemAllocator() {
        ROCM_CHECK(hipGetDevice(&_device_id));
        ROCM_CHECK(hipDeviceGetAttribute(
            &_enable_virtual_mem_allocation, hipDeviceAttributeVirtualMemoryManagementSupported, _device_id));
        vmem_allocations_ = std::make_unique<std::unordered_map<void*, VmemBlock>>();
    }

    void* mallocPhysical(size_t size) {
        RTP_LLM_LOG_DEBUG("malloc physical memory with size %lu\n", size);
        std::lock_guard<std::mutex> lock(lock_);

        void* ptr = malloc(size);
        if (_enable_virtual_mem_allocation && ptr != nullptr) {
            auto it = vmem_allocations_->find(ptr);
            if (it == vmem_allocations_->end()) {
                RTP_LLM_LOG_ERROR("Unexpected allocation, can not find %lu vmem record.", ptr);
                return ptr;
            }
            auto& block = it->second;
            block.pin   = true;
        }
    }

    void* malloc(size_t size) {
        RTP_LLM_LOG_DEBUG("malloc memory with size %lu\n", size);

        if (!_enable_virtual_mem_allocation) {
            return ROCmAllocator<AllocatorType::ROCM, MEMORY_GPU, hipMalloc, hipFree>::malloc(size);
        } else {

            // https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/virtual_memory.html

            hipMemGenericAllocationHandle_t allocHandle;
            hipMemAllocationProp            prop = {};

            prop.type          = hipMemAllocationTypePinned;
            prop.location.type = hipMemLocationTypeDevice;
            prop.location.id   = _device_id;

            size_t granularity = 0;

            ROCM_CHECK(hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));

            if (granularity <= 0) {
                throw std::runtime_error("unsupported system granularity.");
            }

            // Allocate physical memory
            size_t padded_size = (size + granularity - 1) / granularity * granularity;
            ROCM_CHECK(hipMemCreate(&allocHandle, padded_size, &prop, 0));

            RTP_LLM_LOG_DEBUG("malloc memory with size %lu, pad it to %lu\n", size, padded_size);

            // Reserve a virtual memory address range
            void* virtualPointer = nullptr;
            ROCM_CHECK(hipMemAddressReserve(&virtualPointer, padded_size, granularity, nullptr, 0));

            // Map the physical memory to the virtual address range
            ROCM_CHECK(hipMemMap(virtualPointer, padded_size, 0, allocHandle, 0));

            // Set memory access permission for pointer
            hipMemAccessDesc accessDesc = {};
            accessDesc.location.type    = hipMemLocationTypeDevice;
            accessDesc.location.id      = _device_id;
            accessDesc.flags            = hipMemAccessFlagsProtReadWrite;

            ROCM_CHECK(hipMemSetAccess(virtualPointer, padded_size, &accessDesc, 1));

            // record vmem allocation info
            (*vmem_allocations_)[virtualPointer] = {false, true, padded_size, allocHandle};

            return virtualPointer;
        }
    };

    void* mallocSync(size_t size) {
        return malloc(size);
    };

    void free(void** ptr) {
        if (!_enable_virtual_mem_allocation) {
            return ROCmAllocator<AllocatorType::ROCM, MEMORY_GPU, hipMalloc, hipFree>::free(ptr);
        } else {
            if (!(*ptr)) {
                RTP_LLM_LOG_WARNING("Try to free an empty pointer\n");
                return;
            }

            std::lock_guard<std::mutex> lock(lock_);
            auto                        it = vmem_allocations_->find(*ptr);
            if (it == vmem_allocations_->end()) {
                RTP_LLM_LOG_ERROR("Free Pointer Failed, Pointer is not managed by this alloctor %p\n", *ptr);
                return;
            }

            RTP_LLM_LOG_DEBUG("Vmem allocator free pointer %p\n", *ptr);
            // tmp sync to avoid memory free before kernel run. cudaFree will not perform any implicit synchronization
            // when the pointer was allocated with cudaMallocAsync or cudaMallocFromPoolAsync
            const auto& block = it->second;

            if (block.mapped) {
                check_cuda_value(hipMemUnmap(*ptr, block.size));
                check_cuda_value(hipMemRelease(block.handle));
                RTP_LLM_LOG_DEBUG("Double free ptr %p\n", *ptr);
            }

            check_cuda_value(hipMemAddressFree(*ptr, block.size));

            RTP_LLM_LOG_DEBUG("Vmem allocator free pointer %p successfully\n", *ptr);
            return;
        }
    };

    void unmap() {
        std::lock_guard<std::mutex> lock(lock_);

        if (_enable_virtual_mem_allocation) {
            RTP_LLM_LOG_INFO("Vmem allocator unmap all allocated buffer\n");

            for (auto& [ptr, block] : *vmem_allocations_) {

                if (block.pin || !block.mapped) {
                    continue;
                }
                RTP_LLM_LOG_INFO("Vmem allocator unmap %p[%lu]\n", ptr, block.size);

                // 1. 解除映射
                ROCM_CHECK(hipMemUnmap(ptr, block.size));

                // 2. 释放对应的物理显存
                ROCM_CHECK(hipMemRelease(block.handle));
                block.mapped = false;
            }
        } else {
            throw std::runtime_error("Unmap method need more advanced computing device.");
        }
    }

    void map() {
        std::lock_guard<std::mutex> lock(lock_);

        if (_enable_virtual_mem_allocation) {
            RTP_LLM_LOG_INFO("Vmem allocator map all allocated buffer\n");

            for (auto& [ptr, block] : *vmem_allocations_) {

                if (block.pin || block.mapped) {
                    continue;
                }

                RTP_LLM_LOG_INFO("Vmem allocator map %p[%lu]\n", ptr, block.size);

                size_t padded_size = block.size;

                hipMemAllocationProp prop = {};
                prop.type                 = hipMemAllocationTypePinned;
                prop.location.type        = hipMemLocationTypeDevice;
                prop.location.id          = _device_id;

                // Allocate physical memory
                hipMemGenericAllocationHandle_t allocHandle;
                ROCM_CHECK(hipMemCreate(&allocHandle, padded_size, &prop, 0));

                // Map the physical memory to the virtual address range
                ROCM_CHECK(hipMemMap(ptr, padded_size, 0, allocHandle, 0));

                // Set memory access permission for pointer
                hipMemAccessDesc accessDesc = {};
                accessDesc.location.type    = hipMemLocationTypeDevice;
                accessDesc.location.id      = _device_id;
                accessDesc.flags            = hipMemAccessFlagsProtReadWrite;

                ROCM_CHECK(hipMemSetAccess(ptr, padded_size, &accessDesc, 1));

                // update control block info
                block.handle = allocHandle;
                block.mapped = true;
            }
        } else {
            throw std::runtime_error("Unmap method need more advanced computing device.");
        }
    }

private:
    struct VmemBlock {
        bool                            pin;     // Whether the memory block is pinned (resident)
        bool                            mapped;  // Whether the memory has been mapped to virtual memory
        size_t                          size;    // Size of the block
        hipMemGenericAllocationHandle_t handle;  // Physical memory handle (ROCM memory handle)
    };
    std::unique_ptr<std::unordered_map<void*, VmemBlock>> vmem_allocations_;
    std::mutex                                            lock_;
    int32_t                                               _enable_virtual_mem_allocation;
    int32_t                                               _device_id;
};

template<>
class Allocator<AllocatorType::ROCM_HOST>:
    public ROCmAllocator<AllocatorType::ROCM_HOST, MEMORY_CPU_PINNED, hipMallocHost, hipHostFree> {};

}  // namespace rtp_llm
