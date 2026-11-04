#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/metric/MetricReporterConfig.h"
#include "autil/Log.h"

namespace arpc {

class ServerRPCStats;

class ServerMetricReporter : public std::enable_shared_from_this<ServerMetricReporter> {
public:
    ServerMetricReporter(const MetricReporterConfig &config);
    virtual ~ServerMetricReporter() = default;

public:
    std::shared_ptr<ServerRPCStats> makeRPCStats();
    void reportRPCStats(ServerRPCStats *stats);

    virtual void stop() { _enable = false; }

protected:
    virtual void doReportMetric(ServerRPCStats *stats) = 0;

protected:
    std::atomic<bool> _enable{false};
    MetricReporterConfig _config;

private:
    AUTIL_LOG_DECLARE();
};

} // namespace arpc
