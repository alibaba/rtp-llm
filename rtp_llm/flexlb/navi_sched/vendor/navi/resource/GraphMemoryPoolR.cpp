#include "navi/resource/GraphMemoryPoolR.h"

#include "autil/EnvUtil.h"
#include "autil/mem/QuotaController.h"
#include "kmonitor/client/MetricMacro.h"
#include "navi/builder/ResourceDefBuilder.h"
#include "navi/engine/CollectFlags.h"
#include "navi/engine/NaviRegisterMacro.h"
#include "navi/resource/MemoryPoolR.h"

using namespace kmonitor;

namespace navi {

class GraphPoolOpMetrics : public MetricsGroup
{
public:
    bool init(kmonitor::MetricsGroupManager *manager) override {
        REGISTER_GAUGE_MUTABLE_METRIC(_totalUsedCount, "GraphMemoryPool.TotalUsedCount");
        REGISTER_GAUGE_MUTABLE_METRIC(_maxUsedCount, "GraphMemoryPool.MaxUsedCount");
        return true;
    }
    void report(const kmonitor::MetricsTags *tags, GraphMemoryPoolInfo *info) {
        REPORT_MUTABLE_METRIC(_totalUsedCount, info->totalUsedCount);
        REPORT_MUTABLE_METRIC(_maxUsedCount, info->maxUsedCount);
    }
private:
    MutableMetric *_totalUsedCount = nullptr;
    MutableMetric *_maxUsedCount = nullptr;
};
static const std::string GRAPH_POOL_RECYCLE_SIZE_LIMIT = "naviGraphPoolRecycleSizeLimit";

const std::string GraphMemoryPoolR::RESOURCE_ID = "navi.buildin.resource.graph_mem_pool";

GraphMemoryPoolR::GraphMemoryPoolR()
    : _graphPoolReleaseThreshold(getGraphPoolReleaseThreshold())
{
}

GraphMemoryPoolR::GraphMemoryPoolR(navi::MemoryPoolR &memoryPoolR)
    : _memoryPoolR(&memoryPoolR)
    , _graphPoolReleaseThreshold(getGraphPoolReleaseThreshold())
{
}

GraphMemoryPoolR::~GraphMemoryPoolR() {

    if (_graphMemoryPoolReporter) {
        _graphMemoryPoolReporter->report<GraphPoolOpMetrics, GraphMemoryPoolInfo>(nullptr, &_info);
    }
    while (true) {
        autil::mem_pool::Pool *pool = nullptr;
        getPoolFromCache(pool);
        if (pool) {
            _memoryPoolR->putPool(pool);
        } else {
            break;
        }
    }
}

size_t GraphMemoryPoolR::getGraphPoolReleaseThreshold() const {
    static size_t graphPoolReleaseThreshold = autil::EnvUtil::getEnv(GRAPH_POOL_RECYCLE_SIZE_LIMIT, DEFAULT_GRAPH_POOL_RELEASE_THRESHOLD);
    return graphPoolReleaseThreshold;
}

void GraphMemoryPoolR::def(ResourceDefBuilder &builder) const {
    builder.name(RESOURCE_ID, RS_GRAPH);
}

bool GraphMemoryPoolR::config(ResourceConfigContext &ctx) {
    return true;
}

ErrorCode GraphMemoryPoolR::init(ResourceInitContext &ctx) {
    static const size_t defaultQueryMemoryLimit =
        autil::EnvUtil::getEnv("query_mem_limit_bytes", 8ul * 1024 * 1024 * 1024); // default 8GB
    _graphMemoryPoolReporter = _memoryPoolR->getGraphMemoryPoolReporter();
    _quotaController = std::make_shared<autil::mem::QuotaController>(autil::mem::QuotaControllerConfig{
        .shardsPower = 2, // 4shards
        .totalBytesLimit = defaultQueryMemoryLimit,
        .reserveBytes = 4 * 1024 * 1024, // 4MB
        .parent = _memoryPoolR->getQuotaController(),
        .sessional = true
    });
    return EC_NONE;
}

std::shared_ptr<autil::mem_pool::Pool> GraphMemoryPoolR::getPool() {
    _info.currentUsedCount++;
    _info.totalUsedCount++;
    if (_info.maxUsedCount < _info.currentUsedCount) {
        _info.maxUsedCount = _info.currentUsedCount;
    }

    auto thisPtr = std::dynamic_pointer_cast<GraphMemoryPoolR>(shared_from_this());
    auto pool = getPoolInner();
    pool->bindQuotaController(_quotaController);
    return std::shared_ptr<autil::mem_pool::Pool>(pool, [thisPtr](autil::mem_pool::Pool *pool) mutable {
        thisPtr->putPool(pool);
        thisPtr.reset();
    });
}

void GraphMemoryPoolR::putPool(autil::mem_pool::Pool *pool) {
    auto collectMalloc = NAVI_COLLECT_MALLOC();
    if (pool && collectMalloc) {
        pool->release();
    }

    _info.currentUsedCount--;
    pool->bindQuotaController({});
    if (unlikely(_memoryPoolR->reachReleaseThreshold(pool) || size() > _graphPoolReleaseThreshold)) {
        _memoryPoolR->putPool(pool);
    } else {
        // cache && reuse within current query
        putPoolToCache(pool);
    }
}

autil::mem_pool::Pool *GraphMemoryPoolR::getPoolInner() {
    autil::mem_pool::Pool *pool;
    getPoolFromCache(pool);
    if (!pool) {
        pool = _memoryPoolR->getPool();
    }
    return pool;
}

REGISTER_BUILDIN_RESOURCE(GraphMemoryPoolR);

}
