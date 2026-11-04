#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent/QueryInfoForMock.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent/GigStatistic.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric/MetricReporterManager.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/GigAgent.pb.h"

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, QueryInfoForMock);

static GigStatistic statistic;

QueryInfoForMock::QueryInfoForMock()
    : QueryInfo("", "", &statistic, MetricReporterManagerPtr(new MetricReporterManager())) {
}

QueryInfoForMock::~QueryInfoForMock() {
}

std::string QueryInfoForMock::createResponseInfo(float latencyMs, multi_call::MultiCallErrorCode ec,
                                                 float errorRatio, float degradeRatio,
                                                 float avgLatency, WeightTy targetWeight,
                                                 bool warmupFinished) {
    GigResponseInfo responseInfo;
    responseInfo.set_gig_response_checksum(GIG_RESPONSE_CHECKSUM);
    responseInfo.set_latency_ms(latencyMs);
    responseInfo.set_ec((int)ec);

    auto error = responseInfo.mutable_error_ratio();
    error->set_filter_ready(true);
    error->set_ratio(errorRatio);

    auto degrade = responseInfo.mutable_degrade_ratio();
    degrade->set_filter_ready(true);
    degrade->set_ratio(degradeRatio);

    auto latency = responseInfo.mutable_avg_latency();
    latency->set_filter_ready(true);
    latency->set_latency(avgLatency);

    responseInfo.set_target_weight(targetWeight);
    responseInfo.mutable_warm_up_status()->set_finished(warmupFinished);

    string ret;
    if (!responseInfo.SerializeToString(&ret)) {
        AUTIL_LOG(ERROR, "gig serialize message failed");
        return string();
    }
    return ret;
}

} // namespace multi_call
