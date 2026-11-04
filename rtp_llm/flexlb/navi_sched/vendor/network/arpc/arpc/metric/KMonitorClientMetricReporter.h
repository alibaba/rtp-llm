#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/metric/ClientMetricReporter.h"
#include "kmonitor/client/KMonitor.h"
#include "kmonitor/client/core/MutableMetric.h"

namespace arpc {

class KMonitorClientMetricReporter : public ClientMetricReporter
{
public:
    KMonitorClientMetricReporter(const MetricReporterConfig& config, kmonitor::MetricLevel level);
    virtual ~KMonitorClientMetricReporter() = default;

public:
    bool init();
    void doReportMetric(ClientRPCStats* stats) override;

private:
    kmonitor::MetricLevel _level;
    kmonitor::MetricsTags _defaultTags;

    std::unique_ptr<kmonitor::MutableMetric> _clientQps;
    std::unique_ptr<kmonitor::MutableMetric> _errorQps;

    std::unique_ptr<kmonitor::MutableMetric> _clientTotalCost;
    std::unique_ptr<kmonitor::MutableMetric> _clientRequestPendingCost;
    std::unique_ptr<kmonitor::MutableMetric> _clientRequestPackCost;
    std::unique_ptr<kmonitor::MutableMetric> _clientRequestSendCost;
    std::unique_ptr<kmonitor::MutableMetric> _clientRequestIovSize;
    std::unique_ptr<kmonitor::MutableMetric> _clientResponseWaitCost;
    std::unique_ptr<kmonitor::MutableMetric> _clientResponsePopCost;
    std::unique_ptr<kmonitor::MutableMetric> _clientResponseUnpackCost;

private:
    AUTIL_LOG_DECLARE();
};

} // namespace arpc
