#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/metric/KMonitorANetMetricReporter.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/metric/KMonitorANetMetricReporterConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/metric/KMonitorServerMetricReporter.h"
#include "autil/Log.h"

namespace arpc {

class KMonitorANetServerMetricReporter : public ServerMetricReporter {
public:
    KMonitorANetServerMetricReporter(const KMonitorANetMetricReporterConfig &config);
    ~KMonitorANetServerMetricReporter() override = default;

public:
    bool init(anet::Transport *transport);
public:
    void doReportMetric(ServerRPCStats *stats) override;
    void stop() override;

private:
    KMonitorANetMetricReporterConfig _config;
    std::shared_ptr<KMonitorServerMetricReporter> _arpcReporter;
    std::shared_ptr<anet::KMonitorANetMetricReporter> _anetReporter;

private:
    AUTIL_LOG_DECLARE();
};

} // namespace arpc
