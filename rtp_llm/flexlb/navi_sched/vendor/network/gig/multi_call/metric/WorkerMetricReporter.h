#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/MetaEnv.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/MetricUtil.h"

using namespace kmonitor;

namespace multi_call {

class WorkerMetricReporter
{
public:
    WorkerMetricReporter(kmonitor::KMonitor *kMonitor, const MetaEnv &metaEnv);
    ~WorkerMetricReporter();

private:
    WorkerMetricReporter(const WorkerMetricReporter &);
    WorkerMetricReporter &operator=(const WorkerMetricReporter &);

    DECLARE_METRIC(QueueSize);
    DECLARE_METRIC(SubUpdateSuccQps);
    DECLARE_METRIC(SubUpdateFailedQps);
    DECLARE_METRIC(CreateSnapshotSuccQps);
    DECLARE_METRIC(CreateSnapshotFailQps);
    DECLARE_METRIC(UpdateSnapshotQps);

private:
    kmonitor::KMonitor *_kMonitor;
    kmonitor::MetricsTagsPtr _defaultTags;

private:
    AUTIL_LOG_DECLARE();
};
MULTI_CALL_TYPEDEF_PTR(WorkerMetricReporter);
} // namespace multi_call

