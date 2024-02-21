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

    virtual void* malloc(size_t size, const bool is_set_zero = false) = 0;
    virtual void  free(void** ptr) const                              = 0;
    virtual void* reMalloc(void* ptr, size_t size, const bool is_set_zero = false) = 0;

protected:
    virtual void  memSet(void* ptr, const int val, const size_t size) const                = 0;
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
