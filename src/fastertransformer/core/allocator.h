#pragma once
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/core/Types.h"
#include <unordered_map>
#include <vector>


namespace fastertransformer {

enum class AllocatorType {
    CPU,
    CUDA,
    CUDA_HOST,
    TH,
    ROCM,
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

    virtual AllocatorType type() const = 0;
    virtual MemoryType    memoryType() const = 0;

    virtual void* malloc(size_t size)              = 0;
    virtual void  free(void** ptr)                 = 0;
    virtual void* reMalloc(void* ptr, size_t size) = 0;
};

template<AllocatorType AllocType_>
class TypedAllocator : virtual public IAllocator {
public:
    AllocatorType type() const override {
        return AllocType_;
    }
};

template<AllocatorType AllocType_>
class Allocator : public TypedAllocator<AllocType_> {};

}  // namespace fastertransformer
