#include "kmonitor_adapter/QpsRecorder.h"

#include <cassert>
#include <unistd.h>

#include "autil/Log.h"
#include "kmonitor_adapter/MonitorFactory.h"

namespace kmonitor_adapter {

QpsRecorder::QpsRecorder(const std::string &name, MetricPtr metric)
    : _name(name), _metric(std::move(metric)), _mergedSum(0), _last(0) {
    registerRecorder();
}

QpsRecorder::QpsRecorder(Monitor *monitor,
                         const std::string &metricName,
                         kmonitor::MetricLevel level,
                         const Metric::KVVec &tags)
    : _name(metricName), _mergedSum(0), _last(0) {
    assert(monitor);
    _metric = monitor->registerQPSMetric(metricName, level, tags);
    assert(_metric);
    registerRecorder();
}

QpsRecorder::~QpsRecorder() { unregisterRecoder(); }

} // namespace kmonitor_adapter
