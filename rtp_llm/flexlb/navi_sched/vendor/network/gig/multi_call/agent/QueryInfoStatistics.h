#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent/BizStat.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent/WarmUpStrategy.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/ControllerParam.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric/BizMetricReporter.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/GigAgent.pb.h"
#include "autil/Log.h"

namespace multi_call {

class QueryInfoStatistics
{
public:
    QueryInfoStatistics(GigQueryInfo *queryInfo, WarmUpStrategy *warmUpStrategy,
                        BizStatPtr bizStat);
    ~QueryInfoStatistics();

public:
    float degradeLevel(float fullLevel);
    void fillResponseInfo(GigResponseInfo *responseInfo, float responseLatencyMs,
                          MultiCallErrorCode ec, WeightTy targetWeight);
    void reportBizStat(BizMetricReporter &bizReporter);

private:
    void initBizStat();
    void fillPropagationInfos(GigResponseInfo *responseInfo);
    void beginQuery(float percent);
    void endQuery();

private:
    float degradePercent() const;
    float degradePercentLatency() const;
    float degradePercentErrorRatio() const;
    void updateErrorRatio(MultiCallErrorCode ec, ServerRatioFilter &errorRatio);
    void updateDegradeRatio(MultiCallErrorCode ec, ServerRatioFilter &degradeRatio);
    void updateAverageLatency(MultiCallErrorCode ec, float responseLatencyMs,
                              AverageLatency &avgLatency);
    void updateLatency(MultiCallErrorCode ec, float responseLatencyMs);
    void updateLoadBalanceLatency(MultiCallErrorCode ec, float responseLatencyMs);
    void updateWarmUpStatus(GigResponseInfo *responseInfo);

private:
    GigQueryInfo *_queryInfo;
    BizStatPtr _bizStat;
    WarmUpStrategy *_warmUpStrategy;
    std::atomic<bool> _queryCounted;
    float _degradePercent;
    float _delayMs;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(QueryInfoStatistics);

} // namespace multi_call
