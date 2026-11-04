#include "rtp_llm/flexlb/navi_sched/service/NaviSchedClientSummary.h"

namespace navi_sched {

void NaviSchedClientSummary::Jsonize(JsonWrapper &json)
{
    NaviSchedRoleSummary::Jsonize(json);
    json.Jsonize("master_biz_name", masterBizName);
    json.Jsonize("request_received_count", requestReceivedCount);
    json.Jsonize("invalid_request_count", invalidRequestCount);
    json.Jsonize("stopped_request_count", stoppedRequestCount);
    json.Jsonize("schedule_count", scheduleCount);
    json.Jsonize("schedule_success_count", scheduleSuccessCount);
    json.Jsonize("schedule_error_count", scheduleErrorCount);
    json.Jsonize("master_rate_limited_count", masterRateLimitedCount);
    json.Jsonize("master_timeout_count", masterTimeoutCount);
    json.Jsonize("master_transport_error_count", masterTransportErrorCount);
    json.Jsonize("master_routing_error_count", masterRoutingErrorCount);
    json.Jsonize("master_no_decision_count", masterNoDecisionCount);
    json.Jsonize("master_other_error_count", masterOtherErrorCount);
}

std::string NaviSchedClientSummary::format() const
{
    return NaviSchedRoleSummary::format() +
           "\n\nClient\n"
           "  master_biz_name: " +
           masterBizName +
           "\n"
           "  request_received_count: " +
           std::to_string(requestReceivedCount) +
           "\n"
           "  invalid_request_count: " +
           std::to_string(invalidRequestCount) +
           "\n"
           "  stopped_request_count: " +
           std::to_string(stoppedRequestCount) +
           "\n"
           "  schedule_count: " +
           std::to_string(scheduleCount) +
           "\n"
           "  schedule_success_count: " +
           std::to_string(scheduleSuccessCount) +
           "\n"
           "  schedule_error_count: " +
           std::to_string(scheduleErrorCount) +
           "\n"
           "  master_rate_limited_count: " +
           std::to_string(masterRateLimitedCount) +
           "\n"
           "  master_timeout_count: " +
           std::to_string(masterTimeoutCount) +
           "\n"
           "  master_transport_error_count: " +
           std::to_string(masterTransportErrorCount) +
           "\n"
           "  master_routing_error_count: " +
           std::to_string(masterRoutingErrorCount) +
           "\n"
           "  master_no_decision_count: " +
           std::to_string(masterNoDecisionCount) +
           "\n"
           "  master_other_error_count: " +
           std::to_string(masterOtherErrorCount);
}

} // namespace navi_sched
