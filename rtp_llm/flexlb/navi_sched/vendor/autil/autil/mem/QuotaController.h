#pragma once

#include <memory>
#include <thread>
#include <vector>

#include "autil/Lock.h"
#include "autil/Log.h"
#include "autil/ObjectTracer.h"

namespace autil {
namespace mem {

class QuotaController;

struct QuotaControllerConfig {
    size_t shardsPower = 2;          // 4 shards
    size_t totalBytesLimit = 0;      // no limit
    size_t reserveBytes = 16 * 1024; // 16KB
    std::shared_ptr<QuotaController> parent;
    bool sessional = false; // sessional use overflow flag
};

class QuotaController : public ObjectTracer<QuotaController> {
public:
    QuotaController(QuotaControllerConfig config);
    ~QuotaController();
    QuotaController(const QuotaController &) = delete;
    QuotaController &operator=(const QuotaController &) = delete;

private:
    struct ShardType {
        mutable autil::ThreadCond lock;
        size_t availableBytes = 0;
    };

public:
    size_t getTotalBytes() const { return _totalBytes.load(std::memory_order_relaxed); }

    size_t getTotalBytesLimit() const { return _config.totalBytesLimit; }

    size_t isOverflow() const {
        if (_config.sessional) {
            return _overflow;
        }
        return _config.totalBytesLimit > 0 && getTotalBytes() > _config.totalBytesLimit;
    }

    const QuotaControllerConfig &getConfig() const { return _config; }

    void updateQuotaLimit(size_t totalBytesLimit) { _config.totalBytesLimit = totalBytesLimit; }

public:
    __attribute__((warn_unused_result)) bool allocate(size_t bytes) { // RECOMMENDED
        AUTIL_LOG(TRACE1, "[%p] allocate with bytes[%lu]", this, bytes);
        if (bytes < _config.reserveBytes) {
            return allocateWithShards(bytes);
        } else {
            return allocateDirectly(bytes);
        }
    }

    void allocateNoFailure(size_t bytes) { // for legacy allocate usage, NOT RECOMMENDED
        AUTIL_LOG(TRACE1, "[%p] allocate with bytes[%lu]", this, bytes);
        if (bytes < _config.reserveBytes) {
            allocateNoFailureWithShards(bytes);
        } else {
            allocateNoFailureDirectly(bytes);
        }
    }

    void deallocate(size_t bytes) {
        AUTIL_LOG(TRACE1, "[%p] deallocate with bytes[%lu]", this, bytes);
        if (bytes < _config.reserveBytes) {
            deallocateWithShards(bytes);
        } else {
            deallocateDirectly(bytes);
        }
    }

private:
    bool allocateWithShards(size_t bytes) {
        AUTIL_LOG(TRACE3, "[%p] allocate with bytes[%lu] idx[%lu]", this, bytes, shardIndex());
        assert(bytes < _config.reserveBytes);
        auto &shard = _shards[shardIndex()];
        autil::ScopedLock _(shard.lock);
        // 这里不检查sessional && overflow, 是给小内存申请提供一定的免费额度
        if (bytes <= shard.availableBytes) {
            shard.availableBytes -= bytes;
            return true;
        }
        if (!allocateDirectly(_config.reserveBytes)) {
            maybeMarkOverflow();
            return false;
        }
        shard.availableBytes += (_config.reserveBytes - bytes);
        return true;
    }

    bool allocateDirectly(size_t bytes) {
        AUTIL_LOG(TRACE3, "[%p] allocate with bytes[%lu]", this, bytes);
        if (_config.sessional && _overflow) {
            return false;
        }

        if (_config.parent) {
            if (!_config.parent->allocate(bytes)) {
                maybeMarkOverflow();
                return false;
            }
        }
        size_t totalBytes = (_totalBytes += bytes);
        if (_config.totalBytesLimit > 0) {
            if (totalBytes > _config.totalBytesLimit) {
                _totalBytes -= bytes;
                if (_config.parent) {
                    _config.parent->deallocate(bytes);
                }
                maybeMarkOverflow();
                return false;
            }
        }
        return true;
    }

    void allocateNoFailureWithShards(size_t bytes) {
        AUTIL_LOG(TRACE3, "[%p] allocate with bytes[%lu] idx[%lu]", this, bytes, shardIndex());
        assert(bytes < _config.reserveBytes);
        auto &shard = _shards[shardIndex()];
        autil::ScopedLock _(shard.lock);
        if (bytes <= shard.availableBytes) {
            shard.availableBytes -= bytes;
            return;
        }
        allocateNoFailureDirectly(_config.reserveBytes);
        shard.availableBytes += (_config.reserveBytes - bytes);
    }

    void allocateNoFailureDirectly(size_t bytes) {
        AUTIL_LOG(TRACE3, "[%p] allocate with bytes[%lu]", this, bytes);
        if (_config.parent) {
            _config.parent->allocateNoFailure(bytes);
        }
        _totalBytes += bytes;
    }

    void deallocateWithShards(size_t bytes) {
        AUTIL_LOG(TRACE3, "[%p] deallocate with bytes[%lu] idx[%lu]", this, bytes, shardIndex());
        assert(bytes < _config.reserveBytes);
        auto &shard = _shards[shardIndex()];
        size_t giveBackBytes = 0;
        {
            autil::ScopedLock _(shard.lock);
            shard.availableBytes += bytes;
            if (shard.availableBytes > 2 * _config.reserveBytes) { // ATTENTION: compromise for `allocate deallocate
                                                                   // mixture` vs `always deallocate`
                giveBackBytes = shard.availableBytes - _config.reserveBytes;
                shard.availableBytes -= giveBackBytes;
            }
        }
        deallocateDirectly(giveBackBytes);
    }

    void deallocateDirectly(size_t bytes) {
        AUTIL_LOG(TRACE3, "[%p] deallocate with bytes[%lu]", this, bytes);
        assert(_totalBytes >= bytes);
        _totalBytes -= bytes;
        if (_config.parent) {
            _config.parent->deallocate(bytes);
        }
    }

    size_t shardIndex() const {
        thread_local const size_t hash = std::hash<std::thread::id>{}(std::this_thread::get_id());
        return hash & _shardsMask;
    }

    void maybeMarkOverflow() {
        if (_config.sessional) {
            _overflow = true;
        }
    }

private:
    size_t innerGetShardsBytes() const;

private:
    QuotaControllerConfig _config;
    std::atomic<size_t> _totalBytes = 0; // totalBytes = allocatedBytes + shardsAvailableBytes;
    std::vector<ShardType> _shards;
    size_t _shardsMask = 0;
    bool _overflow = false;

private:
    AUTIL_LOG_DECLARE();
};

} // namespace mem
} // namespace autil
