#pragma once

#include <memory>
#include <new>
#include <stddef.h>

#include "autil/mem_pool/ChunkAllocatorBase.h"

namespace autil {
namespace mem_pool {

class SimpleAllocator : public ChunkAllocatorBase {
public:
    SimpleAllocator();
    ~SimpleAllocator();

public:
    void *doAllocate(size_t numBytes) override { return static_cast<void *>(new (std::nothrow) char[numBytes]); }
    void doDeallocate(void *const addr, size_t numBytes) override { delete[](char *) addr; }
};

typedef std::shared_ptr<SimpleAllocator> SimpleAllocatorPtr;

} // namespace mem_pool
} // namespace autil
