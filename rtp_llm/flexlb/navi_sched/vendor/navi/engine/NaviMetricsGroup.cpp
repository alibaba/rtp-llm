#include "navi/engine/NaviMetricsGroup.h"

#include "kmonitor/client/MetricMacro.h"
#include "kmonitor/client/core/MutableMetric.h"
#include "navi/engine/TaskQueue.h"

namespace navi {

bool RunGraphMetrics::init(kmonitor::MetricsGroupManager *manager) {
    REGISTER_LATENCY_MUTABLE_METRIC(_latency, "user.tf.run_sql.Latency");
    REGISTER_LATENCY_MUTABLE_METRIC(_naviRunLatency, "user.tf.run_sql.naviRunLatency");
    REGISTER_LATENCY_MUTABLE_METRIC(_sessionScheduleWaitLatency, "user.tf.run_sql.sessionScheduleWaitLatency");
    REGISTER_LATENCY_MUTABLE_METRIC(_totalQueueLatency, "user.tf.run_sql.totalQueueLatency");
    REGISTER_LATENCY_MUTABLE_METRIC(_totalComputeLatency, "user.tf.run_sql.totalComputeLatency");
    REGISTER_LATENCY_MUTABLE_METRIC(_newSessionLatency, "user.tf.run_sql.newSessionLatency");
    REGISTER_QPS_MUTABLE_METRIC(_naviRunSuccessQps, "user.tf.run_sql.naviRunSuccesssQps");
    REGISTER_QPS_MUTABLE_METRIC(_naviRunFailedQps, "user.tf.run_sql.naviRunFailedQps");
    REGISTER_QPS_MUTABLE_METRIC(_naviRunQps, "user.tf.run_sql.naviRunQps");
    REGISTER_QPS_MUTABLE_METRIC(_naviRpcLackQps, "user.ops.ExchangeKernel.LackQps");
    return true;
}

void RunGraphMetrics::report(const kmonitor::MetricsTags *tags,
                             RunGraphMetricsCollector *collector) {
    if (collector->newSessionLatency) {
        REPORT_MUTABLE_METRIC(_newSessionLatency, collector->newSessionLatency); // us
    }
    if (collector->fillResultTime) {
        REPORT_MUTABLE_METRIC(_latency, (collector->fillResultTime - collector->sessionBeginTime) / 1000);
        REPORT_MUTABLE_METRIC(_naviRunLatency,
                              (collector->fillResultTime - collector->initScheduleTime) / 1000);
    }
    if (collector->initScheduleTime) {
        REPORT_MUTABLE_METRIC(_sessionScheduleWaitLatency,
                              (collector->initScheduleTime - collector->sessionBeginTime) / 1000);
    }
    REPORT_MUTABLE_METRIC(_totalQueueLatency, collector->totalQueueLatency);
    REPORT_MUTABLE_METRIC(_totalComputeLatency, collector->totalComputeLatency);
    if (collector->hasError) {
        REPORT_MUTABLE_QPS(_naviRunFailedQps);
    } else {
        REPORT_MUTABLE_QPS(_naviRunSuccessQps);
    }
    if (collector->hasLack) {
        REPORT_MUTABLE_QPS(_naviRpcLackQps);
    }
    REPORT_MUTABLE_QPS(_naviRunQps);
}

bool NaviStatMetrics::init(kmonitor::MetricsGroupManager *manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(_serviceStatus, "serviceStatus");
    REGISTER_GAUGE_MUTABLE_METRIC(_updatingStatus, "updatingStatus");
    return true;
}

void NaviStatMetrics::report(const kmonitor::MetricsTags *tags,
                                     const NaviStat *stat) {
    REPORT_MUTABLE_METRIC(_serviceStatus, stat->serviceStatus);
    REPORT_MUTABLE_METRIC(_updatingStatus, stat->updatingStatus);
}

bool NaviThreadPoolStatMetrics::init(kmonitor::MetricsGroupManager *manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(_activeThreadCount, "run_sql.NaviActiveThreadCount");
    REGISTER_GAUGE_MUTABLE_METRIC(_activeThreadQueueCount, "run_sql.NaviActiveThreadQueueSize");
    REGISTER_GAUGE_MUTABLE_METRIC(_idleThreadQueueCount, "run_sql.NaviIdleThreadQueueSize");
    return true;
}

void NaviThreadPoolStatMetrics::report(const kmonitor::MetricsTags *tags, const NaviThreadPoolStat *stat) {
    REPORT_MUTABLE_METRIC(_activeThreadCount, stat->activeThreadCount);
    REPORT_MUTABLE_METRIC(_activeThreadQueueCount, stat->activeThreadQueueSize);
    REPORT_MUTABLE_METRIC(_idleThreadQueueCount, stat->idleThreadQueueSize);
}

bool TaskQueueStatMetrics::init(kmonitor::MetricsGroupManager *manager) {
    if (!_threadPoolMetrics.init(manager)) {
        return false;
    }
    REGISTER_GAUGE_MUTABLE_METRIC(_processingCount, "run_sql.NaviRunCount");
    REGISTER_GAUGE_MUTABLE_METRIC(_processingCountRatio, "run_sql.NaviRunCountRatio");
    REGISTER_GAUGE_MUTABLE_METRIC(_queueCount, "run_sql.NaviQueueCount");
    REGISTER_GAUGE_MUTABLE_METRIC(_queueCountRatio, "run_sql.NaviQueueCountRatio");
    return true;
}

void TaskQueueStatMetrics::report(const kmonitor::MetricsTags *tags,
                                  const TaskQueueStat *stat) {
    _threadPoolMetrics.report(tags, &stat->threadPoolStat);
    REPORT_MUTABLE_METRIC(_processingCount, stat->processingCount);
    REPORT_MUTABLE_METRIC(_processingCountRatio, stat->processingCountRatio);
    REPORT_MUTABLE_METRIC(_queueCount, stat->queueCount);
    REPORT_MUTABLE_METRIC(_queueCountRatio, stat->queueCountRatio);
}
}
