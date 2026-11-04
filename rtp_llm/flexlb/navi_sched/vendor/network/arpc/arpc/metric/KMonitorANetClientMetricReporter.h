#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/metric/KMonitorANetMetricReporter.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/metric/KMonitorANetMetricReporterConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/metric/KMonitorClientMetricReporter.h"
#include "autil/Log.h"

namespace arpc {

class KMonitorANetClientMetricReporter : public ClientMetricReporter {
public:
    KMonitorANetClientMetricReporter(const KMonitorANetMetricReporterConfig &config);
    ~KMonitorANetClientMetricReporter() override = default;

public:
    bool init(anet::Transport *transport);

public:
    void doReportMetric(ClientRPCStats *stats) override;
    void stop() override;
private:
    KMonitorANetMetricReporterConfig _config;
    std::shared_ptr<KMonitorClientMetricReporter> _arpcReporter;
    std::shared_ptr<anet::KMonitorANetMetricReporter> _anetReporter;

private:
    AUTIL_LOG_DECLARE();
};

} // namespace arpc
