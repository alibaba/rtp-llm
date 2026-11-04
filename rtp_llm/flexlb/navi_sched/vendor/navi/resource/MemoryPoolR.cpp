#include "navi/resource/MemoryPoolR.h"

#include "autil/EnvUtil.h"
#include "kmonitor/client/MetricMacro.h"
#include "kmonitor/client/MetricsReporter.h"
#include "navi/builder/ResourceDefBuilder.h"
#include "navi/engine/NaviRegisterMacro.h"
#include "navi/engine/CollectFlags.h"

using namespace kmonitor;
using namespace std;

namespace navi {

static const std::string POOL_MODE_ASAN = "naviPoolModeAsan";
static const std::string POOL_CACHE_AUTOSCALE_KEEP_COUNT = "naviPoolAutoScaleKeepCount";
static const std::string POOL_TRUNK_SIZE = "naviPoolTrunkSize";
static const std::string POOL_RECYCLE_SIZE_LIMIT = "naviPoolRecycleSizeLimit";
static const std::string POOL_RELEASE_RATE = "naviPoolReleaseRate";
static const std::string POOL_CACHE_SIZE_METRIC = "poolCacheSize";
static const std::string QUOTA_CONTROLLER_TOTAL_MB = "quotaControllerTotalMB";
static const std::string QUOTA_CONTROLLER_TOTAL_LIMIT_MB = "quotaControllerTotalLimitMB";
static const std::string POOL_CACHE_SIZE_LIMIT_METRIC = "poolCacheSizeLimit";
static const std::string ENABLE_POOL_REPORTER = "enablePoolReporter";
static constexpr size_t POOL_AUTO_SCALE_SECONDS = 180;
class GetPoolOpMetrics : public MetricsGroup
{
public:
    bool init(kmonitor::MetricsGroupManager *manager) override {
        REGISTER_QPS_MUTABLE_METRIC(_qps, "Qps");
        return true;
    }
    void report(const kmonitor::MetricsTags *tags, autil::mem_pool::Pool *pool) {
        REPORT_MUTABLE_QPS(_qps);
    }
private:
    MutableMetric *_qps = nullptr;
};

class PutPoolOpMetrics : public MetricsGroup
{
public:
    bool init(kmonitor::MetricsGroupManager *manager) override {
        REGISTER_QPS_MUTABLE_METRIC(_qps, "Qps");
        REGISTER_GAUGE_MUTABLE_METRIC(_poolTotalBytes, "PoolTotalBytes");
        return true;
    }
    void report(const kmonitor::MetricsTags *tags, autil::mem_pool::Pool *pool) {
        REPORT_MUTABLE_QPS(_qps);
        REPORT_MUTABLE_METRIC(_poolTotalBytes, pool->getTotalBytes());
    }
private:
    MutableMetric *_qps = nullptr;
    MutableMetric *_poolTotalBytes = nullptr;
};

const std::string MemoryPoolR::RESOURCE_ID = "navi.buildin.resource.mem_pool";

MemoryPoolR::MemoryPoolR()
    : ExternalResource(RESOURCE_ID, RS_EXTERNAL)
    , _useAsanPool(autil::EnvUtil::getEnv(POOL_MODE_ASAN, false))
    , _poolChunkSize(autil::EnvUtil::getEnv(POOL_TRUNK_SIZE, DEFAULT_POOL_CHUNK_SIZE) * 127 * 512)
    , _poolReleaseThreshold(autil::EnvUtil::getEnv(POOL_RECYCLE_SIZE_LIMIT, DEFAULT_POOL_RELEASE_THRESHOLD) * 1024 *
                            1024)
    , _poolCacheAutoScaleKeepCount(
          autil::EnvUtil::getEnv(POOL_CACHE_AUTOSCALE_KEEP_COUNT, DEFAULT_POOL_CACHE_AUTOSCALE_KEEP_COUNT))
    , _poolAutoScaleReleaseRate(autil::EnvUtil::getEnv(POOL_RELEASE_RATE, DEFAULT_POOL_RELEASE_RATE))
    , _quotaController(std::make_shared<autil::mem::QuotaController>(autil::mem::QuotaControllerConfig{
          .shardsPower = 4,                 // 16shards
          .totalBytesLimit = autil::EnvUtil::getEnv<size_t>(QUOTA_CONTROLLER_TOTAL_LIMIT_MB, 0 /*no limit*/) * 1024 * 1024,
          .reserveBytes = 16 * 1024 * 1024, // 16MB
          .sessional = false                // global
      }))
    , _enableReporter(autil::EnvUtil::getEnv(ENABLE_POOL_REPORTER, true))
    , _poolCacheSizeLB(std::numeric_limits<size_t>::max())
    , _poolCacheSizeUB(std::numeric_limits<size_t>::min())
    , _poolCacheSizeLimit(0ul){}


MemoryPoolR::~MemoryPoolR() {
}

void MemoryPoolR::initConfigParams() {
    if (_poolReleaseThreshold != 0 && _poolReleaseThreshold < _poolChunkSize) {
        NAVI_KERNEL_LOG(WARN,
                        "original poolReleaseThreshold[%lu] < poolChunkSize[%lu]"
                        ", value changed",
                        _poolReleaseThreshold,
                        _poolChunkSize);
        _poolReleaseThreshold = _poolChunkSize;
    }
    if (_poolAutoScaleReleaseRate > 10ul) {
        NAVI_KERNEL_LOG(WARN, "original poolAutoScaleReleaseRate[%lu] > 10, force to[10]", _poolAutoScaleReleaseRate);
        _poolAutoScaleReleaseRate = 10ul;
    }
    NAVI_KERNEL_LOG(INFO,
                    "memory pool resource config finished: "
                    "poolChunkSize[%lu] "
                    "poolReleaseThreshold[%lu] poolAutoScaleReleaseRate[%lu] keepCount[%lu] useAsanPool[%d]",
                    _poolChunkSize,
                    _poolReleaseThreshold,
                    _poolAutoScaleReleaseRate,
                    _poolCacheAutoScaleKeepCount,
                    _useAsanPool);
    _poolCacheSizeLimit = 0ul;
}

bool MemoryPoolR::Init(kmonitor::MetricsReporterPtr metricsReporter) {
    initLogger(_logger, "NaviMemScale");
    initConfigParams();
    if (metricsReporter && _enableReporter) {
        initMetricsReporter(*metricsReporter);
    } else {
        NAVI_KERNEL_LOG(INFO, "no metrics reporter provided, skip init metrics reporter");
    }
    return true;
}

std::shared_ptr<autil::mem_pool::Pool> MemoryPoolR::getPoolPtr() {
    auto thisPtr = std::dynamic_pointer_cast<MemoryPoolR>(shared_from_this());
    auto pool = getPool();
    return std::shared_ptr<autil::mem_pool::Pool>(pool, [thisPtr](autil::mem_pool::Pool *pool) mutable {
        thisPtr->putPool(pool);
        thisPtr.reset();
    });
}

std::shared_ptr<autil::mem_pool::Pool> MemoryPoolR::getPoolPtrWithQC() {

    auto thisPtr = std::dynamic_pointer_cast<MemoryPoolR>(shared_from_this());
    auto pool = getPool();
    pool->bindQuotaController(_quotaController);
    return std::shared_ptr<autil::mem_pool::Pool>(pool, [thisPtr](autil::mem_pool::Pool *pool) mutable {
        pool->bindQuotaController({});
        thisPtr->putPool(pool);
        thisPtr.reset();
    });
}

autil::mem_pool::Pool *MemoryPoolR::getPool() const {
    autil::mem_pool::Pool *pool;

    size_t poolCacheSize = getPoolFromCache(pool);
    if (poolCacheSize > _poolCacheSizeUB.load(std::memory_order_relaxed)) {
        _poolCacheSizeUB.store(poolCacheSize, std::memory_order_relaxed);
    }
    if (poolCacheSize < _poolCacheSizeLB.load(std::memory_order_relaxed)) {
        _poolCacheSizeLB.store(poolCacheSize, std::memory_order_relaxed);
    }

    if (!pool) {
        size_t limit = _poolCacheSizeLimit.fetch_add(1, std::memory_order_relaxed);
        REPORT_USER_MUTABLE_STATUS(_commonReporter, POOL_CACHE_SIZE_LIMIT_METRIC, limit);

        if (_useAsanPool) {
            pool = new autil::mem_pool::PoolAsan();
        } else {
            pool = new autil::mem_pool::ChunkPool(_poolChunkSize);
        }
        if (_getPoolOpByNewReporter) {
            _getPoolOpByNewReporter->report<GetPoolOpMetrics>(nullptr, pool);
        }
    }
    REPORT_USER_MUTABLE_METRIC(_commonReporter, POOL_CACHE_SIZE_METRIC, poolCacheSize);
    return pool;
}

void MemoryPoolR::putPool(autil::mem_pool::Pool *pool) const {
    auto collectMalloc = NAVI_COLLECT_MALLOC();
    if (pool && collectMalloc) {
        pool->release();
    }

    if (reachReleaseThreshold(pool)) {
        if (_putPoolOpDeletedByThresholdReporter) {
            _putPoolOpDeletedByThresholdReporter->report<PutPoolOpMetrics>(nullptr, pool);
        }
        delete pool;
    } else if (reachPoolCacheSizeLimit()) {
        if (_putPoolOpDeletedByPoolCacheSizeLimitReporter) {
            _putPoolOpDeletedByPoolCacheSizeLimitReporter->report<PutPoolOpMetrics>(
                    nullptr, pool);
        }
        delete pool;
    } else {
        size_t poolCacheSize = putPoolToCache(pool);
        REPORT_USER_MUTABLE_METRIC(_commonReporter, POOL_CACHE_SIZE_METRIC, poolCacheSize);
    }
}

void MemoryPoolR::initMetricsReporter(kmonitor::MetricsReporter &baseReporter) {
    _commonReporter = baseReporter.getSubReporter("memory_pool", {});
    _graphMemoryPoolReporter = _commonReporter->getSubReporter("graph", {});
    _getPoolOpByNewReporter = _commonReporter->getSubReporter(
            "GetPool", {"type", "ByNew"});
    _putPoolOpDeletedByThresholdReporter = _commonReporter->getSubReporter(
            "PutPool", {"type", "DeletedByThreshold"});
    _putPoolOpDeletedByPoolCacheSizeLimitReporter = _commonReporter->getSubReporter(
            "PutPool", {"type", "DeletedByPoolCacheSizeLimit"});
    NAVI_KERNEL_LOG(INFO, "finish init metrics reporter");
}

bool MemoryPoolR::reachPoolCacheSizeLimit() const {
    size_t poolCacheSizeLimit = _poolCacheSizeLimit.load(std::memory_order_relaxed);
    return _poolQueue.Size() + 1 > poolCacheSizeLimit;
}

void MemoryPoolR::workLoop() {
    reportQuotaControllerMetric();
    if (int64_t cur = autil::TimeUtility::currentTimeInSeconds(); cur >= _lastScaledTime + POOL_AUTO_SCALE_SECONDS) {
        poolCacheSizeAutoScale();
        _lastScaledTime = cur;
    }
}

void MemoryPoolR::reportQuotaControllerMetric() {
    REPORT_USER_MUTABLE_METRIC(
        _commonReporter, QUOTA_CONTROLLER_TOTAL_MB, _quotaController->getTotalBytes() / 1024 / 1024);
}

void MemoryPoolR::poolCacheSizeAutoScale() {
    size_t poolCacheSizeLB = _poolCacheSizeLB.load();
    size_t poolCacheSizeUB = _poolCacheSizeUB.load();
    _poolCacheSizeLB.store(std::numeric_limits<size_t>::max());
    _poolCacheSizeUB.store(std::numeric_limits<size_t>::min());

    size_t poolCacheSizeLimit = 0;
    if (poolCacheSizeUB >= poolCacheSizeLB) {
        // to be released pools: poolCacheSizeLB, but add a smooth factor
        poolCacheSizeLimit =
            poolCacheSizeUB - poolCacheSizeLB * _poolAutoScaleReleaseRate / (10 + _poolAutoScaleReleaseRate);
    }
    poolCacheSizeLimit = std::max(poolCacheSizeLimit, _poolCacheAutoScaleKeepCount);
    _poolCacheSizeLimit.store(poolCacheSizeLimit);

    REPORT_USER_MUTABLE_STATUS(_commonReporter,
                               POOL_CACHE_SIZE_LIMIT_METRIC, poolCacheSizeLimit);
    NAVI_LOG(DEBUG, "pool cache size limit updated, lb [%lu] ub [%lu] limit [%lu]",
             poolCacheSizeLB, poolCacheSizeUB, poolCacheSizeLimit);
}




REGISTER_BUILDIN_RESOURCE(MemoryPoolR);

}
