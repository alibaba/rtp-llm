#pragma once

#include "autil/ObjectTracer.h"
#include "kmonitor/client/MetricsReporter.h"
#include "navi/engine/Resource.h"
#include "navi/resource/MemoryPoolR.h"

namespace autil::mem {
class QuotaController;
}

namespace navi {

class GraphMemoryPoolInfo {
public:
    GraphMemoryPoolInfo()
        : currentUsedCount(0)
        , totalUsedCount(0)
        , maxUsedCount(0)
    {
    }
public:
    std::atomic<size_t> currentUsedCount;
    std::atomic<size_t> totalUsedCount;
    size_t maxUsedCount;
};

class GraphMemoryPoolR : public Resource, public MemoryPoolRBase, public autil::ObjectTracer<GraphMemoryPoolR> {
public:
    GraphMemoryPoolR();
    GraphMemoryPoolR(navi::MemoryPoolR &memoryPoolR); // only for elasticpp bridge
    ~GraphMemoryPoolR();
    GraphMemoryPoolR(const GraphMemoryPoolR &) = delete;
    GraphMemoryPoolR &operator=(const GraphMemoryPoolR &) = delete;
public:
    void def(ResourceDefBuilder &builder) const override;
    bool config(ResourceConfigContext &ctx) override;
    ErrorCode init(ResourceInitContext &ctx) override;
public:
    static const std::string RESOURCE_ID;
public:
    std::shared_ptr<autil::mem_pool::Pool> getPool();
    const std::shared_ptr<autil::mem::QuotaController> &getQuotaController() const { return _quotaController; }
private:
    static constexpr size_t DEFAULT_GRAPH_POOL_RELEASE_THRESHOLD = 4; // size
private:
    void putPool(autil::mem_pool::Pool *pool);
    autil::mem_pool::Pool *getPoolInner();
    size_t getGraphPoolReleaseThreshold() const;
private:
    RESOURCE_DEPEND_DECLARE();
private:
    RESOURCE_DEPEND_ON(MemoryPoolR, _memoryPoolR);
    kmonitor::MetricsReporter *_graphMemoryPoolReporter = nullptr;
    std::shared_ptr<autil::mem::QuotaController> _quotaController;
    size_t _graphPoolReleaseThreshold;
    GraphMemoryPoolInfo _info;
};

NAVI_TYPEDEF_PTR(GraphMemoryPoolR);

}
