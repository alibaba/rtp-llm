#pragma once

#include <limits>
#include <unordered_map>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/GigAgent.pb.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/Filter.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/RandomGenerator.h"
#include "autil/Lock.h"
#include "autil/TimeUtility.h"

namespace multi_call {

class WarmUpStrategy;

union InProcessingQueryCount {
    InProcessingQueryCount() : value(0) {
    }
    struct {
        int32_t degradeCount;
        int32_t normalCount;
    };
    int64_t value;
};

struct Metric {
    Metric() : value(INVALID_FILTER_VALUE), loadBalanceValue(INVALID_FILTER_VALUE) {
    }
    void setMetric(float value_, float loadBalanceValue_) {
        if (value_ != INVALID_FILTER_VALUE) {
            value = value_;
        }
        if (loadBalanceValue_ != INVALID_FILTER_VALUE) {
            loadBalanceValue = loadBalanceValue_;
        }
    }
    volatile float value;
    volatile float loadBalanceValue;
};

struct MetricFilter {
    MetricFilter(uint32_t windowSize, uint32_t loadBalanceWindowSize)
        : value(windowSize)
        , loadBalanceValue(loadBalanceWindowSize) {
    }

public:
    void reset() {
        value.reset();
        loadBalanceValue.reset();
    }
    void update(float metric, float valueLimit, float loadBalanceLimit) {
        value.update(metric, valueLimit);
        loadBalanceValue.update(metric, loadBalanceLimit);
    }
    Filter value;
    Filter loadBalanceValue;
};

struct PropagationStat {
    PropagationStat(uint64_t agentId_)
        : agentId(agentId_)
        , version(-1)
        , avgWeight(INVALID_FILTER_VALUE) {
    }
    bool update(const PropagationStatDef &stat, int64_t &delay);
    uint64_t agentId;
    volatile int64_t version;
    volatile float avgWeight;
    Metric errorMetric;
    Metric degradeMetric;
    Metric latencyMetric;
};

class BizStat
{
public:
    BizStat(uint32_t windowSize, uint32_t loadBalanceWindowSize)
        : started(true)
        , beginTimeStamp(-1)
        , lastQueryTimestamp(std::numeric_limits<int64_t>::min())
        , smoothIntervalUs(INVALID_SMOOTH_WEIGHT_INTERVAL)
        , smoothWeightBeginTimeStamp(autil::TimeUtility::currentTime())
        , errorMetricFilter(windowSize, loadBalanceWindowSize)
        , degradeMetricFilter(windowSize, loadBalanceWindowSize)
        , latencyMetricFilter(windowSize, loadBalanceWindowSize)
        , normalLatencyFilter(loadBalanceWindowSize)
        , degradeLatencyFilter(loadBalanceWindowSize)
        , weightFilter(loadBalanceWindowSize / 2)
        , randomGenerator(autil::TimeUtility::currentTime()) {
    }
    ~BizStat();

public:
    void beginQuery(bool degrade);
    void endQuery(bool degrade);
    float updatePropagationStats(const PropagationStats &propagationStats);
    void fillPropagationInfos(PropagationStats &propagationStats);
    PropagationStat *getPropagationStatByIndex(uint64_t index);
    void fillThisPropagationStat(PropagationStatDef *statDef) const;
    bool warmUpFinished(WarmUpStrategy *strategy);
    void setStarted(bool started, int64_t smoothIntervalUs_);
    WeightTy getTargetWeight(int64_t currentTimeUs, WeightTy targetWeight);
    std::string toString() const;
    int64_t count() const;
    void clear();
    void fillErrorRatio(ServerRatioFilter &errorRatio) const;
    void fillDegradeRatio(ServerRatioFilter &degradeRatio) const;
    void fillAvgLatency(AverageLatency &avgLatency) const;
    float getAdjustLbDegradeRatio() const;
    float getAdjustedLoadBalanceLatency() const;

private:
    PropagationStat *getPropagationStat(uint64_t agentId);
    double getSmoothRatio(int64_t currentTimeUs) const;

private:
    static float doAdjustLatency(float alpha, float metric, const Filter &filter, int32_t count);

public:
    std::string bizName;
    volatile bool started;
    volatile int64_t beginTimeStamp;
    volatile int64_t lastQueryTimestamp;
    volatile int64_t smoothIntervalUs;
    volatile int64_t smoothWeightBeginTimeStamp;

    MetricFilter errorMetricFilter;
    MetricFilter degradeMetricFilter;
    MetricFilter latencyMetricFilter;

    InProcessingQueryCount processingCount;
    Filter normalLatencyFilter;
    Filter degradeLatencyFilter;

    Filter weightFilter;
    RandomGenerator randomGenerator;

    autil::ReadWriteLock _statLock;
    std::vector<PropagationStat *> _statArray;
    std::unordered_map<uint64_t, PropagationStat *> _statMap;
};

MULTI_CALL_TYPEDEF_PTR(BizStat);

} // namespace multi_call
