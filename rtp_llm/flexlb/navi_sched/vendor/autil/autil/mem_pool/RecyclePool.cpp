#include "autil/mem_pool/RecyclePool.h"

#include <iosfwd>

#include "autil/TimeUtility.h"

using namespace std;
using namespace autil;

namespace autil {
namespace mem_pool {
AUTIL_LOG_SETUP(autil, RecyclePool);

RecyclePool::RecyclePool(ChunkAllocatorBase *allocator, size_t chunkSize, size_t alignSize)
    : ChunkPoolBase(allocator, chunkSize, alignSize), _freeSize(0) {}

RecyclePool::RecyclePool(size_t chunkSize, size_t alignSize) : ChunkPoolBase(chunkSize, alignSize), _freeSize(0) {}

RecyclePool::~RecyclePool() {}

} // namespace mem_pool
} // namespace autil
