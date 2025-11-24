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
    Allocator() {
        ROCM_CHECK(hipGetDevice(&_device_id));
        ROCM_CHECK(hipDeviceGetAttribute(
            &_enable_virtual_mem_allocation, hipDeviceAttributeVirtualMemoryManagementSupported, _device_id));

        vmem_allocations_ = std::make_unique<std::unordered_map<void*, VmemBlock>>();
    }

    // -------------------------------------
    // mallocPinnedVirtualMemory
    // -------------------------------------
    void* mallocPhysical(size_t size) {
        std::lock_guard<std::mutex> g(lock_);

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

    // -------------------------------------
    // malloc (virtual memory version)
    // -------------------------------------
    void* malloc(size_t size) {
        if (!_enable_virtual_mem_allocation)
            return ROCmAllocator::malloc(size);

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

        setAccess(vptr, padded);

        (*vmem_allocations_)[vptr] = std::move(block);
        return vptr;
    }

    void* mallocSync(size_t size) {
        return malloc(size);
    }

    // -------------------------------------
    // free
    // -------------------------------------
    void free(void** ptr) {
        if (!ptr || !*ptr)
            return;

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
            releaseHandles(block);
            block.mapped = false;
        }

        ROCM_CHECK(hipMemAddressFree(block.vptr, block.size));

        vmem_allocations_->erase(it);

        *ptr = nullptr;
    }

    // -------------------------------------
    // unmap: unmap but keep virtual address reserved
    // -------------------------------------
    void unmap() {
        if (!_enable_virtual_mem_allocation)
            throw std::runtime_error("unmap requires VMM support");

        std::lock_guard<std::mutex> g(lock_);

        for (auto& kv : *vmem_allocations_) {
            VmemBlock& b = kv.second;
            if (b.pin || !b.mapped)
                continue;

            ROCM_CHECK(hipMemUnmap(b.vptr, b.size));
            releaseHandles(b);
            b.mapped = false;
        }
    }

    // -------------------------------------
    // map: map back previously unmapped blocks
    // -------------------------------------
    void map() {
        if (!_enable_virtual_mem_allocation)
            throw std::runtime_error("map requires VMM support");

        std::lock_guard<std::mutex> g(lock_);

        for (auto& kv : *vmem_allocations_) {
            VmemBlock& b = kv.second;
            if (b.pin || b.mapped)
                continue;

            allocatePhysicalMemory(b);
            setAccess(b.vptr, b.size);
        }
    }

private:
    struct VmemBlock {
        void*                                        vptr;
        bool                                         pin;
        bool                                         mapped;
        size_t                                       size;
        std::vector<hipMemGenericAllocationHandle_t> handles;

        VmemBlock(): vptr(nullptr), pin(false), mapped(false), size(0) {}

        VmemBlock(void* v, size_t s): vptr(v), pin(false), mapped(false), size(s) {}
    };

    void allocatePhysicalMemory(VmemBlock& block) {
        hipMemAllocationProp prop{};
        prop.type          = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id   = _device_id;

        size_t allocated = 0;
        block.handles.clear();

        while (allocated < block.size) {
            size_t remain = block.size - allocated;
            size_t csize  = (remain > MAX_PHYSICAL_CHUNK ? MAX_PHYSICAL_CHUNK : remain);

            hipMemGenericAllocationHandle_t h;
            ROCM_CHECK(hipMemCreate(&h, csize, &prop, 0));
            ROCM_CHECK(hipMemMap(block.vptr, csize, allocated, h, 0));

            block.handles.push_back(h);
            allocated += csize;
        }

        block.mapped = true;
    }

    void releaseHandles(VmemBlock& block) {
        for (auto h : block.handles)
            ROCM_CHECK(hipMemRelease(h));
        block.handles.clear();
    }

    void setAccess(void* vptr, size_t size) {
        hipMemAccessDesc d{};
        d.location.type = hipMemLocationTypeDevice;
        d.location.id   = _device_id;
        d.flags         = hipMemAccessFlagsProtReadWrite;

        ROCM_CHECK(hipMemSetAccess(vptr, size, &d, 1));
    }

private:
    static const size_t MAX_PHYSICAL_CHUNK = (1ULL << 30);

    std::unique_ptr<std::unordered_map<void*, VmemBlock>> vmem_allocations_;
    std::mutex                                            lock_;
    int                                                   _enable_virtual_mem_allocation = 0;
    int                                                   _device_id                     = 0;
};

template<>
class Allocator<AllocatorType::ROCM_HOST>:
    public ROCmAllocator<AllocatorType::ROCM_HOST, MEMORY_CPU_PINNED, hipMallocHost, hipHostFree> {};

}  // namespace rtp_llm
