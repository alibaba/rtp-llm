#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/GigAgent.pb.h"
#include "autil/StringUtil.h"

namespace multi_call {

class GigStatistic;
class BizStat;
class WarmUpStrategy;

struct QueryInfoImpl {
    QueryInfoImpl(const std::string &queryInfoStr_, const std::string &warmUpStrategyName_,
                  GigStatistic *statistic);
    ~QueryInfoImpl();
    std::string toString() const;
    void beginQuery(float percent);
    void endQuery();
    void fillPropagationInfos();

private:
    void initBizStat(GigStatistic *statistic);

public:
    google::protobuf::Arena arena;
    bool finished;
    bool validQueryInfo;
    std::string queryInfoStr;
    std::string warmUpStrategyName;
    std::shared_ptr<BizStat> bizStat;
    WarmUpStrategy *warmUpStrategy;
    GigQueryInfo *queryInfo;
    GigResponseInfo *responseInfo;
    bool queryCounted;
    float degradePercent;
    float delayMs;
};

} // namespace multi_call

