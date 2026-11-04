#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent/QueryInfo.h"

namespace multi_call {

class GigStatistic;

class QueryInfoForMock : public QueryInfo
{
public:
    QueryInfoForMock();
    ~QueryInfoForMock();

private:
    QueryInfoForMock(const QueryInfoForMock &);
    QueryInfoForMock &operator=(const QueryInfoForMock &);

public:
    static std::string createResponseInfo(float latencyMs, multi_call::MultiCallErrorCode ec,
                                          float errorRatio, float degradeRatio, float avgLatency,
                                          WeightTy targetWeight, bool warmupFinished);

private:
    GigStatistic *_statistic;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(QueryInfoForMock);

} // namespace multi_call

