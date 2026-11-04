#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/metric/ServerMetricReporter.h"
#include "kmonitor/client/KMonitor.h"
#include "kmonitor/client/core/MutableMetric.h"

namespace arpc {

class KMonitorServerMetricReporter : public ServerMetricReporter {
public:
    KMonitorServerMetricReporter(const MetricReporterConfig &config, kmonitor::MetricLevel level);
    virtual ~KMonitorServerMetricReporter() = default;

public:
    bool init();
    void doReportMetric(ServerRPCStats *stats) override;

private:
    kmonitor::MetricLevel _level;
    kmonitor::MetricsTags _defaultTags;

    std::unique_ptr<kmonitor::MutableMetric> _serverQps;
    std::unique_ptr<kmonitor::MutableMetric> _errorQps;
    std::unique_ptr<kmonitor::MutableMetric> _serverTotalCost;
    std::unique_ptr<kmonitor::MutableMetric> _serverRequestUnpackCost;
    std::unique_ptr<kmonitor::MutableMetric> _serverRequestPendingCost;
    std::unique_ptr<kmonitor::MutableMetric> _serverRequestCallCost;
    std::unique_ptr<kmonitor::MutableMetric> _serverResponsePackCost;
    std::unique_ptr<kmonitor::MutableMetric> _serverResponseSendCost;
    std::unique_ptr<kmonitor::MutableMetric> _serverResponseIovSize;
    std::unique_ptr<kmonitor::MutableMetric> _serverThreadItemCount;

private:
    AUTIL_LOG_DECLARE();
};

} // namespace arpc
