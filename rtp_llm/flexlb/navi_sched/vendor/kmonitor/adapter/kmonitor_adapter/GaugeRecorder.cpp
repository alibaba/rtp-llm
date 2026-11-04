#include "kmonitor_adapter/GaugeRecorder.h"

#include <cassert>
#include <unistd.h>

#include "autil/Log.h"
#include "kmonitor_adapter/MonitorFactory.h"

namespace kmonitor_adapter {

GaugeRecorder::GaugeRecorder(const std::string &name, MetricPtr metric)
    : _name(name), _metric(std::move(metric)), _mergedSum(0), _mergedNum(0) {
    registerRecorder();
}

GaugeRecorder::GaugeRecorder(Monitor *monitor,
                             const std::string &metricName,
                             kmonitor::MetricLevel level,
                             const Metric::KVVec &tags)
    : _name(metricName), _mergedSum(0), _mergedNum(0) {
    assert(monitor);
    _metric = monitor->registerGaugeMetric(metricName, level, tags);
    assert(_metric);
    registerRecorder();
}

GaugeRecorder::~GaugeRecorder() { unregisterRecoder(); }

} // namespace kmonitor_adapter
