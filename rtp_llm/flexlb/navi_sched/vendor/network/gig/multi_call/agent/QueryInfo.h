#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/ControllerParam.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/GigAgent.pb.h"
#include "autil/Lock.h"

namespace multi_call {

class GigStatistic;
class MetricReporterManager;
MULTI_CALL_TYPEDEF_PTR(MetricReporterManager);
class QueryInfoStatistics;
MULTI_CALL_TYPEDEF_PTR(QueryInfoStatistics);

class QueryInfo
{
public:
    QueryInfo(const std::string &queryInfoStr);
    QueryInfo(const std::string &queryInfoStr, const std::string &warmUpStrategy,
              GigStatistic *statistic,
              const std::shared_ptr<MetricReporterManager> &metricReporterManager);
    QueryInfo();
    virtual ~QueryInfo();

public:
    /* this is for server supporting multiLevel degrade strategy,
     * return value is between 0.0f - fullLevel, where larger value means
     * larger client side degrade
     */
    virtual float degradeLevel(float fullLevel) const;
    // 0.0f to 1.0f
    RequestType requestType() const;
    // return this string to client side gig, must be called
    virtual std::string
    finish(float responseLatencyMs /* ms */, MultiCallErrorCode ec,
           WeightTy targetWeight /* pass weight if needed else pass MAX_WEIGHT */);

    const std::string &getQueryInfoStr() const;
    const GigQueryInfo *getQueryInfo() const;
    const GigResponseInfo *getResponseInfo() const;
    const std::string &getBizName() const;
    bool isFinished() const;
    std::string toString() const;
    void setDecWeightFlag();

private:
    void init(const std::string &queryInfoStr);
    void fillGigMetaEnv();
    void reportQueryReceiveMetric();
    void reportAgentMetrics(float responseLatencyMs, MultiCallErrorCode ec, WeightTy targetWeight);
    bool getDecWeightFlag() const;

private:
    bool _finished;
    bool _validQueryInfo;
    google::protobuf::Arena _arena;
    std::string _queryInfoStr;
    GigQueryInfo *_queryInfo;
    GigResponseInfo *_responseInfo;
    QueryInfoStatisticsPtr _statistics;
    MetricReporterManagerPtr _metricReporterManager;
    mutable autil::ReadWriteLock _decWeightFlagLock;
    bool _decWeightFlag = false;
};

MULTI_CALL_TYPEDEF_PTR(QueryInfo);

} // namespace multi_call

