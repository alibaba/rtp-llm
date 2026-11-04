#include "autil/mem/QuotaController.h"

namespace autil {
namespace mem {

AUTIL_LOG_SETUP(autil::mem, QuotaController);

QuotaController::QuotaController(QuotaControllerConfig config)
    : _config(std::move(config)), _shards(1 << _config.shardsPower), _shardsMask(_shards.size() - 1) {}

QuotaController::~QuotaController() {
    if (size_t shardsBytes = innerGetShardsBytes(), totalBytes = _totalBytes; shardsBytes != totalBytes) {
        AUTIL_INTERVAL_LOG(100,
                           WARN,
                           "[%p] Resource leak detected: expected [%lu] bytes, but got [%lu] bytes",
                           this,
                           totalBytes,
                           shardsBytes);
    }
    if (_config.parent) { // return reserved bytes
        _config.parent->deallocate(_totalBytes);
    }
}

size_t QuotaController::innerGetShardsBytes() const {
    size_t totalAvailableBytes = 0;
    for (const auto &shard : _shards) {
        autil::ScopedLock _(shard.lock);
        totalAvailableBytes += shard.availableBytes;
    }
    return totalAvailableBytes;
}

} // namespace mem
} // namespace autil
