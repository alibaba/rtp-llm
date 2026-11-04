#pragma once

#include <unordered_map>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/WorkerNodeInfo.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/SubscribeClustersConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric/MetricReporterManager.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/subscribe/SubscribeServiceFactory.h"
#include "autil/Lock.h"

namespace multi_call {

class SubscribeServiceManager
{

public:
    SubscribeServiceManager();
    virtual ~SubscribeServiceManager();

private:
    SubscribeServiceManager(const SubscribeServiceManager &);
    SubscribeServiceManager &operator=(const SubscribeServiceManager &);

public:
    virtual bool empty() {
        return _subServiceVec.empty();
    };
    bool clusterInfoNeedUpdate();
    virtual bool getClusterInfoMap(TopoNodeVec &topoNodeVec, HeartbeatSpecMap &heartbeatSpecs);
    bool addSubscribe(const SubscribeClustersConfig &gigSubConf);
    bool deleteSubscribe(const SubscribeClustersConfig &gigSubConf);
    virtual bool addSubscribeService(const SubscribeConfig &config);
    virtual bool updateSubscribeService(const SubscribeConfig &config);

public:
    void setMetricReporterManager(const MetricReporterManagerPtr &metricReporterManager) {
        _metricReporterManager = metricReporterManager;
    }

private:
    using UpdateSubScribeServiceMap =
        std::map<SubscribeType,
                 std::map<std::string, std::pair<SubscribeServicePtr, SubscribeServicePtr>>>;
    bool tryCreateNewSubscribeService(const VipConfig &config,
                                      const std::vector<SubscribeServicePtr> &subServiceVec,
                                      UpdateSubScribeServiceMap &updateServiceMap) const;
    void tryGetClusterInfoMapFromUpdateService(TopoNodeVec &topoNodeVec,
                                               HeartbeatSpecMap &heartbeatSpecMap,
                                               std::vector<SubscribeServicePtr> &updateServiceVec);

private:
    MetricReporterManagerPtr _metricReporterManager;
    autil::ReadWriteLock _subLock;
    std::vector<SubscribeServicePtr> _subServiceVec;

    // 待更新的 subscribe service. 结构: {service type: {service key: <old service, new service>}}
    UpdateSubScribeServiceMap _updateServiceMap;
    autil::ReadWriteLock _updateServiceLock;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(SubscribeServiceManager);

} // namespace multi_call
