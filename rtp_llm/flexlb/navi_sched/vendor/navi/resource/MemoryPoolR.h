#pragma once

#include "navi/engine/Resource.h"
#include "navi/resource/MemoryPoolRBase.h"

namespace kmonitor {
class MetricsReporter;
class MetricsTags;
} // namespace kmonitor

namespace autil::mem {
class QuotaController;
}

namespace navi {

class MemoryPoolR : public ExternalResource, public MemoryPoolRBase {
public:
    MemoryPoolR();
    ~MemoryPoolR();
private:
    MemoryPoolR(const MemoryPoolR &);
    MemoryPoolR &operator=(const MemoryPoolR &);
public:
    static const std::string RESOURCE_ID;
public:
    bool Init(std::shared_ptr<kmonitor::MetricsReporter> metricsReporter);
    void workLoop();
    const std::shared_ptr<autil::mem::QuotaController> &getQuotaController() const { return _quotaController; }
public:
    std::shared_ptr<autil::mem_pool::Pool> getPoolPtr();
    std::shared_ptr<autil::mem_pool::Pool> getPoolPtrWithQC();
    autil::mem_pool::Pool *getPool() const;
    void putPool(autil::mem_pool::Pool *pool) const;
    kmonitor::MetricsReporter *getGraphMemoryPoolReporter() const { return _graphMemoryPoolReporter.get(); }
    bool reachReleaseThreshold(autil::mem_pool::Pool *pool) const {
        return _useAsanPool || (_poolReleaseThreshold != 0 && pool->getTotalBytes() > _poolReleaseThreshold);
    }
private:
    void initConfigParams();
    void initMetricsReporter(kmonitor::MetricsReporter &baseReporter);
    bool reachPoolCacheSizeLimit() const;
    void reportQuotaControllerMetric();
    void poolCacheSizeAutoScale();

private:
    static constexpr size_t DEFAULT_POOL_CHUNK_SIZE = 1;        // 1MB
    static constexpr size_t DEFAULT_POOL_RELEASE_THRESHOLD = 1; // 1MB
    static constexpr size_t DEFAULT_POOL_RELEASE_RATE = 3ul;    // rate = 3 / 13
    static constexpr size_t DEFAULT_POOL_CACHE_AUTOSCALE_KEEP_COUNT = 500ul;
private:
    DECLARE_LOGGER();
    bool _useAsanPool;
    size_t _poolChunkSize;
    size_t _poolReleaseThreshold;
    size_t _poolCacheAutoScaleKeepCount;
    size_t _poolAutoScaleReleaseRate; // range [0..10]
private:
    std::shared_ptr<autil::mem::QuotaController> _quotaController;
private:
    bool _enableReporter = true;
    std::shared_ptr<kmonitor::MetricsReporter> _commonReporter;
    std::shared_ptr<kmonitor::MetricsReporter> _graphMemoryPoolReporter;
    std::shared_ptr<kmonitor::MetricsReporter> _getPoolOpByNewReporter;
    std::shared_ptr<kmonitor::MetricsReporter> _putPoolOpDeletedByThresholdReporter;
    std::shared_ptr<kmonitor::MetricsReporter> _putPoolOpDeletedByPoolCacheSizeLimitReporter;
private:
    mutable std::atomic<size_t> _poolCacheSizeLB;
    mutable std::atomic<size_t> _poolCacheSizeUB;
    mutable std::atomic<size_t> _poolCacheSizeLimit;
    size_t _lastScaledTime = 0;
};

NAVI_TYPEDEF_PTR(MemoryPoolR);

} // namespace navi
