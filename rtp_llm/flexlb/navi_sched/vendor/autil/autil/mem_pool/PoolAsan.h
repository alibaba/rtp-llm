#pragma once

#include <vector>

#include "autil/mem/QuotaController.h"
#include "autil/mem_pool/Pool.h"

namespace autil {
namespace mem_pool {

class PoolAsan : public Pool {
public:
    PoolAsan() {}
    ~PoolAsan() { reset(); }

public:
    PoolAsan(const PoolAsan &) = delete;
    PoolAsan &operator=(const PoolAsan &) = delete;

public:
    void *allocate(size_t numBytes) override {
        ScopedSpinLock lock(_mutex);
        auto ptr = malloc(numBytes);
        _chunks.emplace_back(ptr, numBytes);
        _allocSize += numBytes;
        if (_qc) {
            _qc->allocateNoFailure(numBytes);
        }
        return ptr;
    }
    void *allocate(size_t numBytes, size_t alignment) override { return allocate(numBytes); }
    void *allocateUnsafe(size_t numBytes) override { return allocate(numBytes); }
    void *allocateUnsafe(size_t numBytes, size_t alignment) override { return allocate(numBytes, alignment); }
    void deallocate(void *ptr, size_t size) override { return; }
    void release() override { reset(); }
    size_t reset() override {
        ScopedSpinLock lock(_mutex);
        for (auto chunk : _chunks) {
            free(chunk.first);
        }
        _chunks.clear();
        if (_qc) {
            _qc->deallocate(_allocSize);
        }
        _allocSize = 0;
        return 0;
    }

public:
    bool isInPool(const void *ptr) const override {
        ScopedSpinLock lock(_mutex);
        for (auto chunk : _chunks) {
            if ((size_t)chunk.first <= (size_t)ptr && (size_t)ptr < (size_t)chunk.first + chunk.second) {
                return true;
            }
        }
        return false;
    }
    size_t getUsedBytes() const override {
        ScopedSpinLock lock(_mutex);
        return _allocSize;
    }
    size_t getTotalBytes() const override {
        ScopedSpinLock lock(_mutex);
        return _allocSize;
    }
    void bindQuotaController(std::shared_ptr<mem::QuotaController> qc) override {
        if (qc == _qc) {
            return;
        }
        size_t totalBytes = getTotalBytes();
        std::swap(_qc, qc);
        if (qc) {
            qc->deallocate(totalBytes);
        }
        if (_qc) {
            _qc->allocateNoFailure(totalBytes);
        }
    }

private:
    std::vector<std::pair<void *, size_t>> _chunks;
    size_t _allocSize = 0;
    mutable autil::SpinLock _mutex;
    std::shared_ptr<mem::QuotaController> _qc;
};

} // namespace mem_pool
} // namespace autil
