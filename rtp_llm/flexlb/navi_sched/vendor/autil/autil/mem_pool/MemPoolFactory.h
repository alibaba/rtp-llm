#pragma once

#include <memory>
#include <stack>
#include <stddef.h>

#include "autil/Lock.h"
#include "autil/mem_pool/Pool.h"

namespace autil {
namespace mem_pool {

template <class T>
class MemPoolFactory {
public:
    static constexpr size_t DEFAULT_CHUNK_SIZE = 5 * 1024 * 1024; // 5M
    static constexpr size_t MAX_ALLOCATE_SIZE = 10 * 1024 * 1024; // 10M
    static constexpr size_t MAX_RESERVE_POOL_SIZE = 400;

public:
    MemPoolFactory(size_t chunkSize = DEFAULT_CHUNK_SIZE,
                   size_t maxAllocSize = MAX_ALLOCATE_SIZE,
                   size_t maxAllocPoolNum = MAX_RESERVE_POOL_SIZE)
        : _chunkSize(chunkSize), _maxAllocSize(maxAllocSize), _maxAllocPoolNum(maxAllocPoolNum) {}

    ~MemPoolFactory();

public:
    Pool *getPool();
    void recycle(Pool *pool);
    void getPoolVec(std::vector<Pool *> &poolVec, size_t expectedCount);
    void recyclePoolVec(std::vector<Pool *> &poolVec);

    size_t getPoolStackSize() const {
        autil::ScopedLock lock(_mutex);
        return _poolStack.size();
    }

    size_t getChunkSize() const {
        autil::ScopedLock lock(_mutex);
        return _chunkSize;
    }

    size_t getMaxAllocSize() const {
        autil::ScopedLock lock(_mutex);
        return _maxAllocSize;
    }

    size_t getMaxAllocPoolNum() const {
        autil::ScopedLock lock(_mutex);
        return _maxAllocPoolNum;
    }

    void updateChunkSize(size_t chunkSize);
    void updateMaxAllocSize(size_t maxAllocSize);
    void updateMaxAllocPoolNum(size_t maxAllocPoolNum);

private:
    mutable autil::ThreadMutex _mutex;
    std::stack<Pool *> _poolStack;

    size_t _chunkSize;
    size_t _maxAllocSize;
    size_t _maxAllocPoolNum;
};

template <class T>
MemPoolFactory<T>::~MemPoolFactory() {
    Pool *pool = NULL;
    autil::ScopedLock lock(_mutex);
    while (_poolStack.size() > 0) {
        pool = _poolStack.top();
        _poolStack.pop();
        if (pool) {
            pool->reset();
            delete pool;
        }
    }
}

template <class T>
inline Pool *MemPoolFactory<T>::getPool() {
    {
        Pool *pool = NULL;
        autil::ScopedLock lock(_mutex);
        while (_poolStack.size() > 0) {
            pool = _poolStack.top();
            _poolStack.pop();
            if (pool) {
                return pool;
            }
        }
    }
    return new (std::nothrow) T(_chunkSize);
}

template <class T>
inline void MemPoolFactory<T>::recycle(Pool *pool) {
    if (!pool)
        return;
    if (pool->getTotalBytes() > _maxAllocSize) {
        pool->release();
    } else {
        pool->reset();
    }

    {
        autil::ScopedLock lock(_mutex);
        if (_poolStack.size() < _maxAllocPoolNum) {
            _poolStack.push(pool);
            return;
        }
        // delete redundant pool when _maxAllocPoolNum is changed
        while (_poolStack.size() > _maxAllocPoolNum) {
            auto curPool = _poolStack.top();
            _poolStack.pop();
            delete curPool;
        }
    }
    // delete pool should not in mutex life cycle
    delete pool;
}

template <class T>
inline void MemPoolFactory<T>::getPoolVec(std::vector<Pool *> &poolVec, size_t expectedCount) {
    size_t count = 0;
    {
        autil::ScopedLock lock(_mutex);
        while (_poolStack.size() > 0 && count < expectedCount) {
            auto pool = _poolStack.top();
            _poolStack.pop();
            if (pool) {
                count++;
                poolVec.push_back(pool);
            }
        }
    }
    while (count < expectedCount) {
        poolVec.push_back(new (std::nothrow) T(_chunkSize));
        count++;
    }
}

template <class T>
inline void MemPoolFactory<T>::recyclePoolVec(std::vector<Pool *> &poolVec) {
    if (poolVec.empty())
        return;
    for (auto &pool : poolVec) {
        if (!pool)
            continue;
        if (pool->getTotalBytes() > _maxAllocSize) {
            pool->release();
        } else {
            pool->reset();
        }
    }

    size_t beginDelCursor = poolVec.size();
    {
        autil::ScopedLock lock(_mutex);
        for (size_t i = 0; i < poolVec.size(); i++) {
            if (poolVec[i] == nullptr) {
                continue;
            }
            if (_poolStack.size() < _maxAllocPoolNum) {
                _poolStack.push(poolVec[i]);
                continue;
            }
            beginDelCursor = i;
            break;
        }
    }
    // delete pool should not in mutex life cycle
    for (size_t i = beginDelCursor; i < poolVec.size(); i++) {
        if (poolVec[i] != nullptr) {
            delete poolVec[i];
        }
    }
    poolVec.clear();
}

template <class T>
inline void MemPoolFactory<T>::updateChunkSize(size_t chunkSize) {
    autil::ScopedLock lock(_mutex);
    _chunkSize = chunkSize;
}

template <class T>
inline void MemPoolFactory<T>::updateMaxAllocSize(size_t maxAllocSize) {
    autil::ScopedLock lock(_mutex);
    _maxAllocSize = maxAllocSize;
}

template <class T>
inline void MemPoolFactory<T>::updateMaxAllocPoolNum(size_t maxAllocPoolNum) {
    std::vector<Pool *> poolsToDelete;
    {
        autil::ScopedLock lock(_mutex);
        _maxAllocPoolNum = maxAllocPoolNum;
        while (_poolStack.size() > _maxAllocPoolNum) {
            poolsToDelete.push_back(_poolStack.top());
            _poolStack.pop();
        }
    }
    for (auto *pool : poolsToDelete) {
        delete pool;
    }
}

} // namespace mem_pool
} // namespace autil
