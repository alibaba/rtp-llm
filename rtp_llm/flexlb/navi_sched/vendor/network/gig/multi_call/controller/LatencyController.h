#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/FlowControlConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/controller/ControllerBase.h"

namespace multi_call {

class LatencyController : public ControllerBase
{
public:
    LatencyController();

private:
    LatencyController(const LatencyController &);
    LatencyController &operator=(const LatencyController &);

public:
    float controllerOutput() const override;
    float netLatency() const;

public:
    float update(ControllerFeedBack &feedBack);
    bool isLegal(float providerAvgLatency, ControllerFeedBack &feedBack) const;
    bool isLegal(const ControllerChain *bestChain, const MetricLimits &metricLimits,
                 float responseLatency = INVALID_FILTER_VALUE) const;
    float getLegalLimit(const FlowControlConfigPtr &flowControlConfig) const;
    float getLoadBalanceLegalLimit(const FlowControlConfigPtr &flowControlConfig) const;

private:
    void updateLocalLatency(MultiCallErrorCode ec, float responseLatency);
    void updateServerLatency(QueryResultStatistic &stat, float latencyUpperLimitMs);
    void updateServerLatencyMirror(QueryResultStatistic &stat, float latencyUpperLimitMs);
    void doUpdateServerValue(const AverageLatency &avgLatency, float latencyUpperLimitMs);
    bool getLoadBalanceLowLimit(const ControllerFeedBack &feedBack, float &lowLimit) const;
    bool bestControllerValid(const ControllerChain *bestChain) const;

public:
    // load balance
    float updateLoadBalance(ControllerFeedBack &feedBack);
    float controllerLoadBalanceOutput() const;
    void updateNetLatencyFromHeartbeat(int64_t netLatencyUs);

private:
    static float adjustResponseLatency(const ControllerFeedBack &feedBack);
    static float getMaxAllowLatency(float bestLatency, const MetricLimits &metricLimits);
    static float getLatencyLimit(float base, const FlowControlConfigPtr &flowControlConfig);
    static float calculateWeightPercent(float lowLimit, float thisLatency,
                                        const MetricLimits &metricLimits);

private:
    Filter _netLatencyFilter;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(LatencyController);

} // namespace multi_call

