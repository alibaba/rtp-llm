#pragma once

#include "autil/mem_pool/ChunkAllocatorBase.h"

namespace autil {
namespace mem_pool {

class SubPoolAllocator : public autil::mem_pool::ChunkAllocatorBase {
public:
    SubPoolAllocator(Pool *pool) : _pool(pool) {}
    void *doAllocate(size_t numBytes) final { return _pool->allocate(numBytes); }
    void doDeallocate(void *const addr, size_t numBytes) final { return _pool->deallocate(addr, numBytes); }
    Pool *_pool;
};

} // namespace mem_pool
} // namespace autil
