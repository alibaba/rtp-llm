#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/metric/MetricReporterConfig.h"
#include "autil/Log.h"

namespace arpc {

class ClientRPCStats;

class ClientMetricReporter : public std::enable_shared_from_this<ClientMetricReporter> {
public:
    ClientMetricReporter(const MetricReporterConfig &config);
    virtual ~ClientMetricReporter() = default;

public:
    std::shared_ptr<ClientRPCStats> makeRPCStats();
    void reportRPCStats(ClientRPCStats *stats);

    virtual void stop() { _enable = false; }

protected:
    virtual void doReportMetric(ClientRPCStats *stats) = 0;

protected:
    std::atomic<bool> _enable{false};
    MetricReporterConfig _config;

private:
    AUTIL_LOG_DECLARE();
};

} // namespace arpc
