#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/MetricUtil.h"
#include "kmonitor/client/core/MutableMetric.h"

using namespace kmonitor;

namespace multi_call {

class HeartbeatMetricReporter
{
public:
    explicit HeartbeatMetricReporter(kmonitor::KMonitor *kMonitor);
    ~HeartbeatMetricReporter();

    HeartbeatMetricReporter(const HeartbeatMetricReporter &) = delete;
    HeartbeatMetricReporter &operator=(const HeartbeatMetricReporter &) = delete;

    DECLARE_METRIC(HeartbeatCostTime);
    DECLARE_METRIC(HeartbeatRequestLength);
    DECLARE_METRIC(HeartbeatRequestCount);

    DECLARE_MUTABLE_METRIC(HeartbeatQps);
    DECLARE_MUTABLE_METRIC(HeartbeatResponseLength);
    DECLARE_MUTABLE_METRIC(HeartbeatResponseLatency);

public:
    kmonitor::MetricsTagsPtr getMetricsTags(const std::string &statusCode);

private:
    kmonitor::KMonitor *_kMonitor;

private:
    AUTIL_LOG_DECLARE();
};
MULTI_CALL_TYPEDEF_PTR(HeartbeatMetricReporter);

} // namespace multi_call

