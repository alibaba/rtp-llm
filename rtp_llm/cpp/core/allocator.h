#pragma once
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/core/Types.h"
#include <unordered_map>
#include <vector>

namespace rtp_llm {

enum class AllocatorType {
    CPU,
    CUDA,
    CUDA_HOST,
    TH,
    ROCM,
    ROCM_HOST,
};

enum class ReallocType {
    INCREASE,
    REUSE,
    DECREASE,
};

class IAllocator {
public:
    IAllocator() {};
    virtual ~IAllocator() {};

    virtual AllocatorType type() const       = 0;
    virtual MemoryType    memoryType() const = 0;

    virtual void* malloc(size_t size)     = 0;
    virtual void* mallocSync(size_t size) = 0;
    virtual void  free(void** ptr)        = 0;
    // TODO: remove reMalloc
    virtual void* reMalloc(void* ptr, size_t size) = 0;

    // mallocPrivate is used for cuda graph,
    // the allocated memories are freezed
    // and can not be allocated for non-private allocation.
    virtual void* mallocPrivate(size_t size);
};

/**
@brief Interface for allocators that allocate virtual address ranges and
   optionally bind / unbind physical memory to them.
This interface is intended for virtual memory managers that need to:
reserve large contiguous virtual address regions up-front,
attach / detach physical backing (BAR, VRAM, etc.) on demand,
support over-commitment or memory oversubscription scenarios.
*/
class IVirtualMemAllocator: virtual public IAllocator {
public:
    /// @brief Maps physical memory to the virtual address ranges owned by this
    ///        allocator (idempotent).
    virtual void map() = 0;
    /// @brief Unmaps physical memory from the virtual address ranges owned by
    ///        this allocator, without releasing the virtual addresses.
    virtual void unmap() = 0;

    /// @brief Allocates a block of memory that does not undergo virtual memory mapping.
    ///        The map and unmap methods will not be able to release memory allocated via mallocResidentMemory.
    virtual void* mallocResidentMemory(size_t size) = 0;
};

template<AllocatorType AllocType_>
class TypedAllocator: virtual public IAllocator {
public:
    AllocatorType type() const override {
        return AllocType_;
    }
};

template<AllocatorType AllocType_>
class Allocator: public TypedAllocator<AllocType_> {};

}  // namespace rtp_llm
