#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent/QueryInfo.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/ControllerParam.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/MultiCallConfig.h"

namespace multi_call {

class BizStat;
class GigStatistic;

MULTI_CALL_TYPEDEF_PTR(QueryInfo);

class GigAgent
{
public:
    GigAgent();
    ~GigAgent();

private:
    GigAgent(const GigAgent &);
    GigAgent &operator=(const GigAgent &);

public:
    QueryInfoPtr getQueryInfo(const std::string &queryInfo, const std::string &warmUpStrategy = "",
                              bool withBizStat = true);
    void updateWarmUpStrategy(const std::string &warmUpStrategy, int64_t timeoutInSecond,
                              int64_t queryCountLimit);
    bool init(const std::string &logPrefix = "", bool enableAgentStat = true,
              const std::string &snapshotLogDir = "");
    void start(int64_t smoothIntervalUs = INVALID_SMOOTH_WEIGHT_INTERVAL);
    void stop(int64_t smoothIntervalUs = INVALID_SMOOTH_WEIGHT_INTERVAL);
    bool isStopped() const;
    void start(const std::string &bizName, PartIdTy partId = INVALID_PART_ID,
               int64_t smoothIntervalUs = INVALID_SMOOTH_WEIGHT_INTERVAL);
    void stop(const std::string &bizName, PartIdTy partId = INVALID_PART_ID,
              int64_t smoothIntervalUs = INVALID_SMOOTH_WEIGHT_INTERVAL);
    bool isStopped(const std::string &bizName, PartIdTy partId = INVALID_PART_ID) const;

    bool longTimeNoQuery(int64_t noQueryTimeInSecond) const;
    bool longTimeNoQuery(const std::string &bizName, int64_t noQueryTimeInSecond) const;
    std::string getLogPrefix() const;
    std::string getSnapshotLogDir() const;

private:
    std::shared_ptr<BizStat> getBizStat(const std::string &bizName, PartIdTy partId);
    friend class HeartbeatServerManager;

private:
    GigStatistic *_statistic;
    std::shared_ptr<MetricReporterManager> _metricReporterManager;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(GigAgent);

} // namespace multi_call
