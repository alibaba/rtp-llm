#pragma once
#include "rtp_llm/cpp/core/allocator.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include <cuda.h>
#include <mutex>
#include <unordered_set>

namespace rtp_llm {

class ICudaAllocator: virtual public IAllocator {
public:
    ICudaAllocator(int device_id): device_id_(device_id) {}
    virtual ~ICudaAllocator() {};

    MemoryType memoryType() const override {
        return MEMORY_GPU;
    }

    void setStream(cudaStream_t stream) {
        stream_ = stream;
    }

    cudaStream_t returnStream() {
        return stream_;
    };

    void* reMalloc(void* ptr, size_t size) override;

protected:
    virtual bool        isExist(void* address) const                 = 0;
    virtual ReallocType isReMalloc(void* address, size_t size) const = 0;

protected:
    cudaStream_t stream_ = 0;  // initialize as default stream
    const int    device_id_;
};

class PurePointerCudaAllocator: public ICudaAllocator {
public:
    PurePointerCudaAllocator(int device_id);
    ~PurePointerCudaAllocator();

public:
    void* malloc(size_t size) override;
    void* mallocSync(size_t size) override;
    void  free(void** ptr) override;

protected:
    virtual bool        isExist(void* address) const;
    virtual ReallocType isReMalloc(void* address, size_t size) const;

    virtual void* doMalloc(size_t size)     = 0;
    virtual void* doMallocSync(size_t size) = 0;
    virtual void  doFree(void* ptr)         = 0;

    void destroy();

private:
    std::unique_ptr<std::unordered_map<void*, size_t>> pointer_mapping_;
    std::mutex                                         lock_;
};

template<>
class Allocator<AllocatorType::CUDA>:
    public PurePointerCudaAllocator,
    public IVirtualMemAllocator,
    public TypedAllocator<AllocatorType::CUDA> {
public:
    Allocator(int device_id);
    ~Allocator();

    void* doMalloc(size_t size) override;
    void* doMallocSync(size_t size) override;
    void  doFree(void* ptr) override;

    /**
     * @brief Remaps all non-pinned virtual address ranges to freshly allocated
     *        physical memory.
     *
     * This function is intended to be called after `unmap()` has been used to
     * release underlying physical allocations (for instance, when temporarily
     * returning memory to the OS or across suspend/resume cycles).  For every
     * virtual block that is **not** marked as pinned, it:
     *   1. Creates a new physical allocation of identical size.
     *   2. Maps that allocation to the previously reserved virtual address.
     *   3. Re-establishes read/write access for the current device.
     *   4. Updates internal bookkeeping with the new handle.
     *
     * Pinned blocks are skipped entirely, preserving their original physical
     * backing.
     *
     * Thread-safe: protected by an internal mutex.
     *
     * @note Must be preceded by a matching `unmap()` call; otherwise the
     *       virtual addresses remain in an unmapped state.
     */
    void map() override;
    /**
     * @brief Releases the physical backing of all non-pinned device allocations
     *        while preserving their virtual address reservations.
     *
     * For each block managed by this allocator:
     *   - If the block is **pinned**, it is left untouched.
     *   - Otherwise:
     *       1. The virtual-to-physical mapping is removed (`cuMemUnmap`).
     *       2. The physical allocation handle is destroyed (`cuMemRelease`).
     *
     * Virtual address ranges remain reserved and can later be re-populated with
     * new physical memory by calling `map()`.
     *
     * Thread-safe: protected by an internal mutex.
     *
     * @warning After this call, accessing device pointers associated with
     *          non-pinned blocks results in undefined behavior until `map()`
     *          is invoked.
     */
    void  unmap() override;
    void* mallocPhysical(size_t size) override;

private:
    struct VmemBlock {
        bool                         pin;     // Whether the memory block is pinned (resident)
        bool                         mapped;  // Whether the memory has been mapped to virtual memory
        size_t                       size;    // Size of the block
        CUmemGenericAllocationHandle handle;  // Physical memory handle (CUDA memory handle)
    };

    // tag for legacy device <= sm70
    bool _enable_virtual_mem_allocation;

    // Mapping from virtual device pointer (CUdeviceptr) to the memory control block
    std::unique_ptr<std::unordered_map<CUdeviceptr, VmemBlock>> vmem_allocations_;
    std::mutex                                                  lock_;
};

template<>
class Allocator<AllocatorType::CUDA_HOST>:
    public PurePointerCudaAllocator,
    public TypedAllocator<AllocatorType::CUDA_HOST> {
public:
    Allocator(int device_id);
    ~Allocator();

    MemoryType memoryType() const override {
        return MEMORY_CPU_PINNED;
    }

    void* doMalloc(size_t size) override;
    void* doMallocSync(size_t size) override;
    void  doFree(void* ptr) override;
};

}  // namespace rtp_llm
