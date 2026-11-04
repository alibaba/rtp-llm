#pragma once

#include <atomic>

#include "kmonitor_adapter/GaugeRecorder.h"
#include "kmonitor_adapter/MonitorFactory.h"
#include "kmonitor_adapter/ThreadData.h"
#include "kmonitor_adapter/Time.h"

namespace kmonitor_adapter {

class LatencyRecorder : public GaugeRecorder {
public:
    LatencyRecorder(const std::string &name, MetricPtr metric) : GaugeRecorder(name, std::move(metric)) {}
    LatencyRecorder(Monitor *monitor,
                    const std::string &metricName,
                    kmonitor::MetricLevel level,
                    const Metric::KVVec &tags = Metric::KVVec())
        : GaugeRecorder() {
        assert(monitor);
        GaugeRecorder::_name = metricName;
        GaugeRecorder::_metric = monitor->registerGaugePercentileMetric(metricName, level, tags);
        assert(GaugeRecorder::_metric);
        registerRecorder();
    }

    uint64_t processValue(uint64_t val) override { return time::getMicroSeconds(val); }
};

} // namespace kmonitor_adapter
