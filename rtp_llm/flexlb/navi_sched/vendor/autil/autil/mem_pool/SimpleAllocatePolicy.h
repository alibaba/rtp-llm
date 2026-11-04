#pragma once

#include <functional>
#include <new>
#include <stddef.h>

#include "autil/Log.h"
#include "autil/SanitizerUtil.h"
#include "autil/mem_pool/AllocatePolicy.h"
#include "autil/mem_pool/ChunkAllocatorBase.h"
#include "autil/mem_pool/MemoryChunk.h"

namespace autil {
namespace mem_pool {

class SimpleAllocatePolicy : public AllocatePolicy {
public:
    typedef std::function<void(size_t, size_t)> TotalBytesChangedListenerFunc;

public:
    SimpleAllocatePolicy(ChunkAllocatorBase *allocator, size_t chunkSize, bool ownAllocator = false);
    SimpleAllocatePolicy(size_t chunkSize, TotalBytesChangedListenerFunc totalBytesChangedListener = {});
    ~SimpleAllocatePolicy();

public:
    /**
     * alloc a chunk
     * @param nBytes number of bytes to allocate
     * @return allocated chunk
     */
    MemoryChunk *allocate(size_t nBytes) override;

    /**
     * release allocated chunks
     * @return total size of chunks
     */
    size_t release() override;

    /**
     * Reset memory to init state
     */
    size_t reset() override;

    /**
     * clear trunk data
     */
    void clear() override;

    /**
     * Get used chunk size
     */
    size_t getUsedBytes() const override;

    /**
     * Get total chunk size
     */
    size_t getTotalBytes() const override;

    /**
     * Return chunk size
     */
    size_t getChunkSize() const override;

    /**
     * Return available chunk size, it is always
     * equal to GetChunkSize() - sizeof(ChainedMemoryChunk)
     */
    size_t getAvailableChunkSize() const override;

    /*
     * check whether pointer in pool.
     */
    bool isInChunk(const void *ptr) const override;

protected:
    ChunkAllocatorBase *_allocator;
    bool _ownAllocator;

    ChainedMemoryChunk *_chunkHeader;
    ChainedMemoryChunk *_currentChunk;
    size_t _chunkSize;
    size_t _usedBytes;
    size_t _totalBytes;

private:
    TotalBytesChangedListenerFunc _totalBytesChangedListener;

private:
    AUTIL_LOG_DECLARE();
};

/////////////////////////////////////////////////////////////////////////
//

inline MemoryChunk *SimpleAllocatePolicy::allocate(size_t nBytes) {
    size_t nAllocSize = nBytes + sizeof(ChainedMemoryChunk);
    if (nAllocSize <= _chunkSize) {
        nAllocSize = _chunkSize;
    }

    ChainedMemoryChunk *nextChunk = _currentChunk ? _currentChunk->next() : _chunkHeader;
    if (nextChunk && nAllocSize <= _chunkSize) {
        _currentChunk = nextChunk;
    } else {
        const auto allocatedChunk = _allocator->allocate(nAllocSize);
        if (!allocatedChunk) {
            return nullptr;
        }
        nextChunk = new (allocatedChunk) ChainedMemoryChunk(nAllocSize);
        SanitizerUtil::PoisonMemoryRegion(((char *)nextChunk) + sizeof(ChainedMemoryChunk),
                                          nAllocSize - sizeof(ChainedMemoryChunk));
        if (!nextChunk) {
            return NULL;
        }
        size_t oldTotalBytes = _totalBytes;
        _totalBytes += nAllocSize;
        if (_totalBytesChangedListener) {
            _totalBytesChangedListener(oldTotalBytes, _totalBytes);
        }
        if (!_chunkHeader) {
            _currentChunk = _chunkHeader = nextChunk;
        } else {
            if (_currentChunk) {
                nextChunk->next() = _currentChunk->next();
                if (_currentChunk->next()) {
                    _currentChunk->next()->prev() = nextChunk;
                }
                _currentChunk->next() = nextChunk;
                nextChunk->prev() = _currentChunk;
            } else {
                nextChunk->next() = _chunkHeader;
                _chunkHeader->prev() = nextChunk;
                _chunkHeader = nextChunk;
            }
            _currentChunk = nextChunk;
        }
    }
    _usedBytes += _currentChunk->getTotalBytes();
    return _currentChunk;
}

inline size_t SimpleAllocatePolicy::release() {
    ChainedMemoryChunk *pChunk = _chunkHeader;
    ChainedMemoryChunk *pChunk2 = NULL;
    while (pChunk) {
        pChunk2 = pChunk;
        pChunk = pChunk2->next();
        SanitizerUtil::UnpoisonMemoryRegion((void *)pChunk2, pChunk2->getTotalBytes());
        _allocator->deallocate((void *)pChunk2, pChunk2->getTotalBytes());
    }
    _chunkHeader = _currentChunk = NULL;
    size_t oldTotalBytes = _totalBytes;
    _usedBytes = _totalBytes = 0;
    if (_totalBytesChangedListener) {
        _totalBytesChangedListener(oldTotalBytes, _totalBytes);
    }
    return oldTotalBytes;
}

inline size_t SimpleAllocatePolicy::reset() {
    size_t oldTotalBytes = _totalBytes;
    if (_currentChunk == nullptr) {
        // skip useless reset to avoid cache-miss
        return oldTotalBytes;
    }
    for (ChainedMemoryChunk *chunk = _chunkHeader; chunk;) {
        if (chunk->getTotalBytes() <= _chunkSize) {
            chunk->reset();
            chunk = chunk->next();
        } else {
            ChainedMemoryChunk *prevChunk = chunk->prev();
            ChainedMemoryChunk *nextChunk = chunk->next();
            _totalBytes -= chunk->getTotalBytes();
            SanitizerUtil::UnpoisonMemoryRegion((void *)chunk, chunk->getTotalBytes());
            _allocator->deallocate((void *)chunk, chunk->getTotalBytes());
            chunk = nextChunk;
            if (prevChunk) {
                prevChunk->next() = nextChunk;
            } else {
                _chunkHeader = nextChunk;
            }
            if (nextChunk) {
                nextChunk->prev() = prevChunk;
            }
        }
    }
    _usedBytes = 0;
    _currentChunk = NULL;
    if (oldTotalBytes != _totalBytes && _totalBytesChangedListener) {
        _totalBytesChangedListener(oldTotalBytes, _totalBytes);
    }
    return oldTotalBytes;
}

inline void SimpleAllocatePolicy::clear() {
    ChainedMemoryChunk *pChunk = _chunkHeader;
    while (pChunk) {
        SanitizerUtil::UnpoisonMemoryRegion((void *)pChunk, pChunk->getTotalBytes());
        pChunk->clear();
        pChunk = pChunk->next();
    }
}

inline size_t SimpleAllocatePolicy::getUsedBytes() const { return _usedBytes; }

inline size_t SimpleAllocatePolicy::getTotalBytes() const { return _totalBytes; }

inline size_t SimpleAllocatePolicy::getChunkSize() const { return _chunkSize; }

inline size_t SimpleAllocatePolicy::getAvailableChunkSize() const { return _chunkSize - sizeof(ChainedMemoryChunk); }

inline bool SimpleAllocatePolicy::isInChunk(const void *ptr) const {
    ChainedMemoryChunk *chunk = _chunkHeader;
    while (chunk) {
        if (chunk->isInChunk(ptr)) {
            return true;
        }
        chunk = chunk->next();
    }
    return false;
}

} // namespace mem_pool
} // namespace autil
