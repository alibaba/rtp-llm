#include "autil/mem_pool/Pool.h"

#include <stdint.h>

#include "autil/mem_pool/ChunkAllocatorBase.h"
#include "autil/mem_pool/SimpleAllocatePolicy.h"

namespace autil {
namespace mem_pool {

AUTIL_LOG_SETUP(autil, Pool);

MemoryChunk Pool::DUMMY_CHUNK;

Pool::Pool(AllocatePolicy *allocatePolicy, size_t alignSize)
    : _alignSize(alignSize), _memChunk(&(Pool::DUMMY_CHUNK)), _allocSize(0), _allocPolicy(allocatePolicy) {}

Pool::Pool(ChunkAllocatorBase *allocator, size_t chunkSize, size_t alignSize)
    : Pool(new SimpleAllocatePolicy(allocator, chunkSize), alignSize) {}

Pool::Pool(size_t chunkSize, size_t alignSize)
    : Pool(new SimpleAllocatePolicy(chunkSize,
                                    [this](size_t oldTotalBytes, size_t newTotalBytes) {
                                        if (!_quotaController || oldTotalBytes == newTotalBytes) {
                                            return;
                                        }
                                        if (oldTotalBytes < newTotalBytes) {
                                            _quotaController->allocateNoFailure(newTotalBytes - oldTotalBytes);
                                        } else {
                                            _quotaController->deallocate(oldTotalBytes - newTotalBytes);
                                        }
                                    }),
           alignSize) {}

Pool::~Pool() {
    _memChunk = NULL;
    delete _allocPolicy;
    _allocPolicy = NULL;
}

bool Pool::isInPool(const void *ptr) const { return _allocPolicy->isInChunk(ptr); }

void *Pool::allocate(size_t numBytes) {
    ScopedSpinLock lock(_mutex);
    return allocateUnsafe(numBytes);
}

void *Pool::allocate(size_t numBytes, size_t alignment) {
    ScopedSpinLock lock(_mutex);
    return allocateUnsafe(numBytes, alignment);
}

void *Pool::allocateUnsafe(size_t numBytes) {
    size_t allocSize = alignBytes(numBytes, _alignSize);
    void *ptr = _memChunk->allocate(allocSize);
    if (!ptr) {
        MemoryChunk *chunk = _allocPolicy->allocate(allocSize);
        if (!chunk) {
            AUTIL_LOG(ERROR,
                      "Allocate too large memory chunk: %ld, "
                      "max available chunk size: %ld",
                      (int64_t)numBytes,
                      (int64_t)_allocPolicy->getAvailableChunkSize());
            return NULL;
        }
        _memChunk = chunk;
        ptr = _memChunk->allocate(allocSize);
    }
    _allocSize += allocSize;
    SanitizerUtil::UnpoisonMemoryRegion(ptr, allocSize);
    return ptr;
}

void *Pool::allocateUnsafe(size_t numBytes, size_t alignment) {
    size_t allocSize = alignBytes(numBytes, _alignSize);
    void *ptr = _memChunk->allocate(allocSize, alignment);
    if (!ptr) {
        MemoryChunk *chunk = _allocPolicy->allocate(allocSize);
        if (!chunk) {
            AUTIL_LOG(ERROR,
                      "Allocate memory chunk of size %ld failed, "
                      "available chunk size: %ld",
                      (int64_t)numBytes,
                      (int64_t)_allocPolicy->getAvailableChunkSize());
            return NULL;
        }
        _memChunk = chunk;
        ptr = _memChunk->allocate(allocSize, alignment);
    }
    _allocSize += allocSize;
    SanitizerUtil::UnpoisonMemoryRegion(ptr, allocSize);
    return ptr;
}

void Pool::clear() {
    ScopedSpinLock lock(_mutex);
    _allocPolicy->clear();
}

void Pool::bindQuotaController(std::shared_ptr<mem::QuotaController> quotaController) {
    if (quotaController == _quotaController) {
        return;
    }
    size_t totalBytes = _allocPolicy->getTotalBytes();
    std::swap(_quotaController, quotaController);
    if (quotaController) {
        quotaController->deallocate(totalBytes);
    }
    if (_quotaController) {
        _quotaController->allocateNoFailure(totalBytes);
    }
}

std::shared_ptr<mem::QuotaController> Pool::getQuotaController() const { return _quotaController; }

namespace {
class SharedPoolWrapper final : public Pool {
public:
    explicit SharedPoolWrapper(Pool *impl) : _impl(impl) {}

    ~SharedPoolWrapper() = default;

public:
    void *allocate(size_t numBytes) override { return _impl->allocate(numBytes); }

    void *allocate(size_t numBytes, size_t alignment) override { return _impl->allocate(numBytes, alignment); }

    void deallocate(void *ptr, size_t size) override { _impl->deallocate(ptr, size); }

    void release() override { _impl->release(); }

    size_t reset() override { return _impl->reset(); }

    size_t getUsedBytes() const override { return _impl->getUsedBytes(); }

    size_t getTotalBytes() const override { return _impl->getTotalBytes(); }

    bool isInPool(const void *ptr) const override { return _impl->isInPool(ptr); }

private:
    Pool *_impl = nullptr;
};
} // namespace

std::shared_ptr<Pool> Pool::maybeShared(Pool *pool) {
    if (pool == nullptr) {
        return {};
    }
#ifdef MATCHDOC_USING_FAST_PATH
    return std::shared_ptr<Pool>(pool, [](Pool *) {});
#endif
    auto sp = pool->weak_from_this().lock();
    if (sp) {
        return sp;
    }
    return std::make_shared<SharedPoolWrapper>(pool);
}

// ChunkPoolBase
ChunkPoolBase::ChunkPoolBase(AllocatePolicy *allocatePolicy, size_t alignSize) : Pool(allocatePolicy, alignSize) {}

ChunkPoolBase::ChunkPoolBase(ChunkAllocatorBase *allocator, size_t chunkSize, size_t alignSize)
    : Pool(allocator, chunkSize, alignSize) {}

ChunkPoolBase::ChunkPoolBase(size_t chunkSize, size_t alignSize) : Pool(chunkSize, alignSize) {}

ChunkPool::ChunkPool(AllocatePolicy *allocatePolicy, size_t alignSize) : ChunkPoolBase(allocatePolicy, alignSize) {}

ChunkPool::ChunkPool(ChunkAllocatorBase *allocator, size_t chunkSize, size_t alignSize)
    : ChunkPoolBase(allocator, chunkSize, alignSize) {}

ChunkPool::ChunkPool(size_t chunkSize, size_t alignSize) : ChunkPoolBase(chunkSize, alignSize) {}

// UnsafePool
UnsafePool::UnsafePool(ChunkAllocatorBase *allocator, size_t chunkSize, size_t alignSize)
    : ChunkPoolBase(allocator, chunkSize, alignSize) {}

UnsafePool::UnsafePool(size_t chunkSize, size_t alignSize) : ChunkPoolBase(chunkSize, alignSize) {}

UnsafePool::UnsafePool(AllocatePolicy *allocatePolicy, size_t alignSize) : ChunkPoolBase(allocatePolicy, alignSize) {}

} // namespace mem_pool
} // namespace autil
