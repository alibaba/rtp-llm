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
