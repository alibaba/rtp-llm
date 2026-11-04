#include "kmonitor_adapter/CounterRecorder.h"

#include <cassert>
#include <unistd.h>

#include "autil/Log.h"
#include "kmonitor_adapter/MonitorFactory.h"

namespace kmonitor_adapter {

CounterRecorder::CounterRecorder(const std::string &name, MetricPtr metric)
    : _name(name), _metric(std::move(metric)), _mergedSum(0) {
    registerRecorder();
}

CounterRecorder::CounterRecorder(Monitor *monitor,
                                 const std::string &metricName,
                                 kmonitor::MetricLevel level,
                                 const Metric::KVVec &tags)
    : _name(metricName), _mergedSum(0) {
    if (monitor) {
        _metric = monitor->registerGaugeMetric(metricName, level, tags);
        assert(_metric);
        registerRecorder();
    }
}

CounterRecorder::~CounterRecorder() { unregisterRecoder(); }

} // namespace kmonitor_adapter
