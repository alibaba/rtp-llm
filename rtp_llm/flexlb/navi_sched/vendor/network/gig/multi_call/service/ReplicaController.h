#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceProvider.h"

namespace multi_call {

struct SnapshotBizInfo;

class ReplicaController
{
public:
    ReplicaController(const SearchServiceProviderVector *serviceVector,
                      int64_t cooldownDuration = COOLDOWN_AFTER_RESET_US);
    ~ReplicaController();

private:
    ReplicaController(const ReplicaController &);
    ReplicaController &operator=(const ReplicaController &);

public:
    void init(size_t bestChainQueueSize, bool multiVersion);
    void update(ControllerChain *candidate, const MetricLimits &metricLimits);
    ControllerChain *getBestChain() const;
    float getAvgWeight() const;
    void fillReplicaControllerInfo(ControllerFeedBack &feedBack) const;
    bool getBaseErrorRatio(float &baseErrorRatio) const;
    bool getBaseDegradeRatio(float &baseDegradeRatio) const;
    bool getBaseAvgLatency(float &baseAvgLatency) const;
    void fillControllerInfo(SnapshotBizInfo &bizInfo) const;
    void setCooldownDuration(int64_t durationUs);
    void toString(std::string &debugStr) const;

private:
    void updateBestChain(ControllerChain *candidate, const MetricLimits &metricLimits);
    void setBestChain(ControllerChain *bestChain);
    void updateReplicaStat(int64_t currentTime);
    void updateWeightStat();
    void updateGlobalAvgLatency();

private:
    const SearchServiceProviderVector *_serviceVector;
    std::vector<ControllerChain *> _bestChainQueue;
    ControllerChain *volatile _bestChain;
    volatile bool _multiVersion;
    mutable autil::ReadWriteLock _lock;
    int64_t _lastUpateTime;
    volatile float _avgWeight;
    volatile float _maxWeight;
    volatile float _globalAvgLatency;
    Filter _bestLoadBalanceLatencyFilter;
    Filter _bestLoadBalanceDegradeRatioFilter;
    volatile int64_t _lastResetTime;
    volatile int64_t _cooldownDuration;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(ReplicaController);

} // namespace multi_call
