#pragma once

#include <cstdint>
#include "kmonitor/client/MetricsReporter.h"

namespace kmonitor {
class MutableMetric;
class MetricsTags;
class MetricsGroupManager;
}

namespace navi {

struct NaviStat {
    int serviceStatus = 0;
    int updatingStatus = 0;
};

struct RunGraphMetricsCollector {
    int64_t newSessionLatency = 0;
    int64_t totalQueueLatency = 0;
    int64_t totalComputeLatency = 0;
    double sessionBeginTime = 0;
    double initScheduleTime = 0;
    double fillResultTime = 0;
    bool hasError = false;
    bool hasLack = false;
};

class RunGraphMetrics : public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager *manager) override;
    void report(const kmonitor::MetricsTags *tags, RunGraphMetricsCollector *collector);
private:
    kmonitor::MutableMetric *_latency = nullptr;
    kmonitor::MutableMetric *_sessionScheduleWaitLatency = nullptr;
    kmonitor::MutableMetric *_naviRunLatency = nullptr;
    kmonitor::MutableMetric *_totalQueueLatency = nullptr;
    kmonitor::MutableMetric *_totalComputeLatency = nullptr;
    kmonitor::MutableMetric *_newSessionLatency = nullptr;
    kmonitor::MutableMetric *_naviRunSuccessQps = nullptr;
    kmonitor::MutableMetric *_naviRunFailedQps = nullptr;
    kmonitor::MutableMetric *_naviRunQps = nullptr;
    kmonitor::MutableMetric *_naviRpcLackQps = nullptr;
};

class NaviStatMetrics : public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager *manager) override;
    void report(const kmonitor::MetricsTags *tags, const NaviStat *stat);
private:
    kmonitor::MutableMetric *_serviceStatus = nullptr;
    kmonitor::MutableMetric *_updatingStatus = nullptr;
};

struct NaviThreadPoolStat;

class NaviThreadPoolStatMetrics : public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager *manager) override;
    void report(const kmonitor::MetricsTags *tags, const NaviThreadPoolStat *stat);
private:
    kmonitor::MutableMetric *_activeThreadCount = nullptr;
    kmonitor::MutableMetric *_activeThreadQueueCount = nullptr;
    kmonitor::MutableMetric *_idleThreadQueueCount = nullptr;
};

struct TaskQueueStat;

class TaskQueueStatMetrics : public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager *manager) override;
    void report(const kmonitor::MetricsTags *tags, const TaskQueueStat *stat);
private:
    NaviThreadPoolStatMetrics _threadPoolMetrics;
    kmonitor::MutableMetric *_processingCount = nullptr;
    kmonitor::MutableMetric *_processingCountRatio = nullptr;
    kmonitor::MutableMetric *_queueCount = nullptr;
    kmonitor::MutableMetric *_queueCountRatio = nullptr;
};


}
