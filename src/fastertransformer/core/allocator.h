#pragma once
#include "src/fastertransformer/utils/logger.h"
#include <unordered_map>
#include <vector>


namespace fastertransformer {

enum class AllocatorType {
    CPU,
    CUDA,
    TH,
};

enum class ReallocType {
    INCREASE,
    REUSE,
    DECREASE,
};

class IAllocator {
public:
    virtual ~IAllocator(){};

    virtual void* malloc(size_t size, const bool is_set_zero = true, bool is_host = false) = 0;
    virtual void  free(void** ptr, bool is_host = false) const                             = 0;
    virtual void  memSet(void* ptr, const int val, const size_t size)                      = 0;

    void* reMalloc(void* ptr, size_t size, const bool is_set_zero = false, bool is_host = false);

protected:
    virtual bool        isExist(void* address) const                 = 0;
    virtual ReallocType isReMalloc(void* address, size_t size) const = 0;

    void* getAddress(void* ptr) const {
        return ptr;
    }
};

template<AllocatorType AllocType_>
class Allocator;

}  // namespace fastertransformer
