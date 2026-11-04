#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/ControllerParam.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/QueryResultStatistic.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"

namespace multi_call {

struct MetricLimits {
    MetricLimits()
        : errorRatioLimit(MAX_PERCENT)
        , latencyUpperLimitMs(DEFAULT_LATENCY_UPPER_LIMIT_MS)
        , latencyUpperLimitPercent(DEFAULT_LATENCY_UPPER_LIMIT_PERCENT)
        , fullDegradeLatency(DEFAULT_DEGRADE_LATENCY) {
    }
    bool operator==(const MetricLimits &rhs) const {
        return errorRatioLimit == rhs.errorRatioLimit &&
               latencyUpperLimitMs == rhs.latencyUpperLimitMs &&
               latencyUpperLimitPercent == rhs.latencyUpperLimitPercent &&
               fullDegradeLatency == rhs.fullDegradeLatency;
    }
    float getLatencyUpperLimitMs() {
        return std::min(latencyUpperLimitMs + fullDegradeLatency,
                        fullDegradeLatency * latencyUpperLimitPercent);
    }
    float errorRatioLimit;
    float latencyUpperLimitMs;
    float latencyUpperLimitPercent; // this percent already add 1
    float fullDegradeLatency;
};

struct ControllerChain;

struct ControllerFeedBack {
    ControllerFeedBack(QueryResultStatistic &stat_)
        : mirrorResponse(false)
        , stat(stat_)
        , bestChain(NULL)
        , maxWeight(MAX_WEIGHT_FLOAT)
        , minWeight(MIN_WEIGHT_FLOAT)
        , bestLoadBalanceLatency(INVALID_FILTER_VALUE)
        , bestLoadBalanceDegradeRatio(INVALID_FILTER_VALUE) {
    }
    ControllerFeedBack &operator=(const ControllerFeedBack &) = delete;

private:
    // for ut
    bool operator==(const ControllerFeedBack &rhs) const {
        return mirrorResponse == rhs.mirrorResponse && &stat == &rhs.stat &&
               bestChain == rhs.bestChain && metricLimits == rhs.metricLimits &&
               maxWeight == rhs.maxWeight && minWeight == rhs.minWeight &&
               bestLoadBalanceLatency == rhs.bestLoadBalanceLatency &&
               bestLoadBalanceDegradeRatio == rhs.bestLoadBalanceDegradeRatio;
    }

public:
    bool mirrorResponse;
    QueryResultStatistic &stat;
    const ControllerChain *bestChain;
    MetricLimits metricLimits;
    float maxWeight;
    float minWeight;
    float bestLoadBalanceLatency;
    float bestLoadBalanceDegradeRatio;
};

} // namespace multi_call

