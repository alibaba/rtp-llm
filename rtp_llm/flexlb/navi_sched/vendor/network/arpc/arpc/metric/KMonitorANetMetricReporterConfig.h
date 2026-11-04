#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/metric/ANetMetricReporterConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/metric/MetricReporterConfig.h"
#include "kmonitor/client/KMonitor.h"

namespace arpc {

class KMonitorANetMetricReporterConfig : public autil::legacy::Jsonizable
{
public:
    KMonitorANetMetricReporterConfig() { arpcConfig.type = "anet"; }
    ~KMonitorANetMetricReporterConfig() = default;

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override
    {
        json.Jsonize("arpc", arpcConfig);
        json.Jsonize("anet", anetConfig);
        json.Jsonize("metric_level", metricLevel);
    }

public:
    MetricReporterConfig arpcConfig {false};
    anet::ANetMetricReporterConfig anetConfig {false};
    kmonitor::MetricLevel metricLevel {kmonitor::NORMAL};
};

} // namespace arpc
