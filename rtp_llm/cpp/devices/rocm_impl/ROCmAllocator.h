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
    /**
     * @brief Construct a new ROCM allocator.
     *
     * Initializes device id, queries whether virtual memory management is
     * supported on the current device, and allocates the internal bookkeeping
     * structure used to track virtual memory allocations.
     */
    Allocator() {
        ROCM_CHECK(hipGetDevice(&_device_id));
        ROCM_CHECK(hipDeviceGetAttribute(
            &_enable_virtual_mem_allocation, hipDeviceAttributeVirtualMemoryManagementSupported, _device_id));

        _enable_virtual_mem_allocation = false;
        vmem_allocations_              = std::make_unique<std::unordered_map<void*, VmemBlock>>();
    }

    /**
     * @brief Allocate resident memory and mark it as pinned
     *        in the virtual memory managment. prevent it from being
     *        unmapped by unmap().
     *
     * @param size Number of bytes to allocate.
     * @return Pointer to the allocated memory (virtual or physical).
     */
    void* mallocResidentMemory(size_t size) {
        void* p = malloc(size);
        if (!_enable_virtual_mem_allocation)
            return p;

        auto it = vmem_allocations_->find(p);
        if (it == vmem_allocations_->end()) {
            RTP_LLM_LOG_ERROR("mallocPinnedVirtualMemory: missing vmem block for %p", p);
            return p;
        }

        it->second.pin = true;
        return p;
    }

    /**
     * @brief Allocate device memory, optionally using ROCm virtual memory APIs.
     *
     * When virtual memory management is supported, this function first reserves
     * a contiguous range of virtual address space of at least @p size bytes
     * (rounded up to the device granularity), then allocates a single physical
     * memory object of that size and maps it to the reserved virtual address
     * range. The returned pointer is a virtual address that can be used like a
     * normal device pointer by client code.
     *
     * When virtual memory management is not supported, this function falls
     * back to a standard hipMalloc-based allocation and returns a raw device
     * pointer.
     *
     * @param size Number of bytes to allocate.
     * @return Pointer to the allocated memory (virtual address if VMM is used).
     */
    void* malloc(size_t size) {
        RTP_LLM_LOG_INFO("Rocm virtual memory malloc %lu bytes, device %d", size, _device_id);

        if (!_enable_virtual_mem_allocation) {
            return ROCmAllocator::malloc(size);
        }

        std::lock_guard<std::mutex> g(lock_);

        hipMemAllocationProp prop{};
        prop.type          = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id   = _device_id;

        size_t granularity = 0;
        ROCM_CHECK(hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));

        if (granularity == 0)
            throw std::runtime_error("Unsupported granularity");

        size_t padded = ((size + granularity - 1) / granularity) * granularity;

        void* vptr = nullptr;
        ROCM_CHECK(hipMemAddressReserve(&vptr, padded, granularity, nullptr, 0));

        VmemBlock block(vptr, padded);
        allocatePhysicalMemory(block);

        (*vmem_allocations_)[vptr] = std::move(block);

        RTP_LLM_LOG_INFO("Rocm virtual memory malloc successfully %p, device %d", vptr, _device_id);
        return vptr;
    }

    /**
     * @brief Allocate device memory synchronously.
     *
     * This is a thin wrapper around malloc(size) that preserves the synchronous
     * allocation semantics expected by the allocator interface. When virtual
     * memory management is enabled, the same virtual-memory-based allocation
     * path is used. Otherwise, it falls back to standard hipMalloc.
     *
     * @param size Number of bytes to allocate.
     * @return Pointer to the allocated memory.
     */
    void* mallocSync(size_t size) {
        return malloc(size);
    }

    /**
     * @brief Free memory previously allocated by this allocator.
     *
     * If virtual memory management is not enabled, this function delegates to
     * the base ROCmAllocator implementation and releases the physical device
     * memory via hipFree.
     *
     * If virtual memory management is enabled, this function looks up the
     * corresponding VmemBlock, unmaps the associated virtual address range,
     * releases the underlying physical memory handle, frees the reserved
     * virtual address space, and removes the block from the internal tracking
     * map. The caller's pointer is set to nullptr on success.
     *
     * @param ptr Address of the pointer to the memory to be freed. The pointer
     *            will be set to nullptr if the memory is successfully released.
     */
    void free(void** ptr) {
        if (!ptr || !*ptr)
            return;
        RTP_LLM_LOG_INFO("Rocm memory pointer:%p", *ptr);

        if (!_enable_virtual_mem_allocation) {
            ROCmAllocator::free(ptr);
            return;
        }

        std::lock_guard<std::mutex> g(lock_);

        auto it = vmem_allocations_->find(*ptr);
        if (it == vmem_allocations_->end()) {
            RTP_LLM_LOG_ERROR("free: pointer %p not allocated by vmem allocator", *ptr);
            return;
        }

        VmemBlock& block = it->second;

        if (block.mapped) {
            ROCM_CHECK(hipMemUnmap(block.vptr, block.size));
            ROCM_CHECK(hipMemRelease(block.physical_mem_handle));
            block.mapped = false;
        }

        ROCM_CHECK(hipMemAddressFree(block.vptr, block.size));
        vmem_allocations_->erase(it);

        *ptr = nullptr;
    }

    /**
     * @brief Unmap physical memory from all non-pinned virtual allocations.
     *
     * This function iterates over all tracked virtual memory blocks and, for
     * each block that is currently mapped and not marked as pinned, unmaps
     * the virtual address range from its physical allocation and releases the
     * corresponding physical memory handle. The virtual address space remains
     * reserved and the virtual pointer returned to client code stays valid,
     * but it no longer has backing physical memory until map() is called again.
     *
     * This operation is only available when virtual memory management support
     * is enabled on the current device. If not supported, a std::runtime_error
     * is thrown.
     */
    void unmap() {
        if (!_enable_virtual_mem_allocation)
            throw std::runtime_error("unmap requires VMM support");

        std::lock_guard<std::mutex> g(lock_);
        RTP_LLM_LOG_INFO("Rocm virtual memory unmap, there are %d pointer(s)", vmem_allocations_->size());

        for (auto& kv : *vmem_allocations_) {
            VmemBlock& b = kv.second;
            if (b.mapped && !b.pin) {
                ROCM_CHECK(hipMemUnmap(b.vptr, b.size));
                ROCM_CHECK(hipMemRelease(b.physical_mem_handle));
                b.mapped = false;
            }
        }
    }

    /**
     * @brief Remap physical memory for all unmapped virtual allocations.
     *
     * This function iterates over all tracked virtual memory blocks and, for
     * each block that is currently unmapped, allocates a new physical memory
     * object and maps it back to the previously reserved virtual address range.
     * After this call, all virtual pointers that were previously unmapped by
     * unmap() (and not marked as pinned at that time) will again refer to
     * valid, newly allocated physical memory.
     *
     * This operation is only available when virtual memory management support
     * is enabled on the current device. If not supported, a std::runtime_error
     * is thrown.
     */
    void map() {
        if (!_enable_virtual_mem_allocation)
            throw std::runtime_error("map requires VMM support");

        std::lock_guard<std::mutex> g(lock_);

        RTP_LLM_LOG_INFO("Rocm virtual memory map, there are %d pointer(s)", vmem_allocations_->size());
        for (auto& kv : *vmem_allocations_) {
            VmemBlock& b = kv.second;
            if (!b.mapped) {
                allocatePhysicalMemory(b);
            }
        }
    }

private:
    struct VmemBlock {
        void*                           vptr;
        bool                            pin;
        bool                            mapped;
        size_t                          size;
        hipMemGenericAllocationHandle_t physical_mem_handle;

        VmemBlock(): vptr(nullptr), pin(false), mapped(false), size(0) {}

        VmemBlock(void* v, size_t s): vptr(v), pin(false), mapped(false), size(s) {}
    };

    /**
     * @brief Allocate a single physical memory object and map it to a given
     *        virtual address range.
     *
     * This helper creates a HIP generic allocation of size @p block.size on
     * the current device, maps it to the virtual address stored in
     * @p block.vptr, sets appropriate access permissions for the device, and
     * updates the VmemBlock to reflect that it is now mapped and associated
     * with the given physical memory handle.
     *
     * @param block VmemBlock describing the virtual address range to be backed
     *              by newly allocated physical memory.
     */
    void allocatePhysicalMemory(VmemBlock& block) {
        hipMemAllocationProp prop{};
        prop.type          = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id   = _device_id;

        hipMemGenericAllocationHandle_t h;
        RTP_LLM_LOG_INFO("Rocm virtual allocate, size = %lu bytes", block.size);

        ROCM_CHECK(hipMemCreate(&h, block.size, &prop, 0));
        ROCM_CHECK(hipMemMap(block.vptr, block.size, 0, h, 0));

        hipMemAccessDesc d{};
        d.location.type = hipMemLocationTypeDevice;
        d.location.id   = _device_id;
        d.flags         = hipMemAccessFlagsProtReadWrite;

        ROCM_CHECK(hipMemSetAccess(block.vptr, block.size, &d, 1));
        block.physical_mem_handle = h;
        block.mapped              = true;
    }

private:
    static const size_t MAX_PHYSICAL_CHUNK = (1ULL << 28);

    std::unique_ptr<std::unordered_map<void*, VmemBlock>> vmem_allocations_;
    std::mutex                                            lock_;
    int                                                   _enable_virtual_mem_allocation = 0;
    int                                                   _device_id                     = 0;
};

template<>
class Allocator<AllocatorType::ROCM_HOST>:
    public ROCmAllocator<AllocatorType::ROCM_HOST, MEMORY_CPU_PINNED, hipMallocHost, hipHostFree> {};

}  // namespace rtp_llm
