#include "kmonitor_adapter/Monitor.h"

#include <sstream>

#include "kmonitor/client/KMonitor.h"
#include "kmonitor/client/KMonitorFactory.h"
#include "kmonitor/client/KMonitorWorker.h"
#include "kmonitor/client/common/Common.h"
#include "kmonitor/client/core/MetricsConfig.h"
#include "kmonitor/client/core/MetricsTags.h"

using namespace std;
using namespace kmonitor;

namespace kmonitor_adapter {
AUTIL_DECLARE_AND_SETUP_LOGGER(monitor, Monitor);

Monitor::Monitor(string serviceName, autil::RecursiveThreadMutex *metricMutex, kmonitor::KMonitor *kmonitor)
    : _metricMutex(metricMutex), _kmonitor(kmonitor) {
    _kmonitor->SetServiceName(serviceName);
}

Monitor::~Monitor() {}

MetricPtr Monitor::registerRawMetric(const string &name, MetricLevel level, const KVVec &tags) {
    return registerMetric(name, RAW, level, tags);
}

MetricPtr Monitor::registerQPSMetric(const string &name, MetricLevel level, const KVVec &tags) {
    return registerMetric(name, QPS, level, tags);
}

MetricPtr Monitor::registerCounterMetric(const string &name, MetricLevel level, const KVVec &tags) {
    return registerMetric(name, COUNTER, level, tags);
}

MetricPtr Monitor::registerStatusMetric(const string &name, MetricLevel level, const KVVec &tags) {
    return registerMetric(name, kmonitor::MetricType::STATUS, level, tags);
}

MetricPtr Monitor::registerGaugeMetric(const string &name, MetricLevel level, const KVVec &tags, bool more_stats) {
    return registerMetric(name, GAUGE, level, tags);
}

MetricPtr Monitor::registerGaugePercentileMetric(const string &name, MetricLevel level, const KVVec &tags) {
    return registerMetric(name, SUMMARY, level, tags);
}

MetricPtr Monitor::registerLatencyMetric(const string &name, MetricLevel level, const KVVec &tags) {
    return registerMetric(name, SUMMARY, level, tags);
}

MetricPtr Monitor::registerMetric(const string &name, MetricType metricType, MetricLevel level, const KVVec &tags) {
    // autil::ScopedLock lock(*_metricMutex);
    MetricPtr metric(new Metric(_kmonitor->RegisterMetric(name, metricType, level)));
    metric->addTags(tags);
    return metric;
}

void Monitor::registerBuildInMetrics(MetricsTags &tags) { _kmonitor->registerBuildInMetrics(&tags); }

} // namespace kmonitor_adapter
