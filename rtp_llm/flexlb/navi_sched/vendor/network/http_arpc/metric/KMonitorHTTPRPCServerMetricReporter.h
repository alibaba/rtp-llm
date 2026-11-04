#pragma once

#include <map>
#include <memory>
#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/kmonitor/cpp_client/src/kmonitor/client/KMonitor.h"
#include "rtp_llm/flexlb/navi_sched/vendor/kmonitor/cpp_client/src/kmonitor/client/core/MutableMetric.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/metric/ANetMetricReporter.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/metric/HTTPRPCMetricReporterConfig.h"
#include "autil/Log.h"

namespace http_arpc {

class HTTPRPCStats;

class KMonitorHTTPRPCServerMetricReporter : public std::enable_shared_from_this<KMonitorHTTPRPCServerMetricReporter> {
public:
    KMonitorHTTPRPCServerMetricReporter(const HTTPRPCMetricReporterConfig &config, kmonitor::MetricLevel level);
    virtual ~KMonitorHTTPRPCServerMetricReporter() = default;

public:
    bool init(anet::Transport *transport);
    std::shared_ptr<HTTPRPCStats> makeHTTPRPCStats();
    void reportHTTPRPCStats(HTTPRPCStats *stats);

    void stop() { _enable = false; }

private:
    void doReportMetric(HTTPRPCStats *stats);

private:
    bool _enable{false};
    bool _initialized{false};
    HTTPRPCMetricReporterConfig _config;
    kmonitor::MetricLevel _level;
    kmonitor::MetricsTags _defaultTags;

    // HTTP RPC 指标
    std::unique_ptr<kmonitor::MutableMetric> _httpQps;
    std::unique_ptr<kmonitor::MutableMetric> _errorQps;
    std::unique_ptr<kmonitor::MutableMetric> _httpTotalCost;
    std::unique_ptr<kmonitor::MutableMetric> _httpRequestParseCost;
    std::unique_ptr<kmonitor::MutableMetric> _httpRequestPendingCost;
    std::unique_ptr<kmonitor::MutableMetric> _httpRequestCallCost;
    std::unique_ptr<kmonitor::MutableMetric> _httpResponsePackCost;
    std::unique_ptr<kmonitor::MutableMetric> _httpResponseSendCost;
    std::unique_ptr<kmonitor::MutableMetric> _httpResponseSize;
    std::unique_ptr<kmonitor::MutableMetric> _httpThreadItemCount;

    // ANET 指标报告器
    std::shared_ptr<anet::ANetMetricReporter> _anetMetricReporter;

private:
    AUTIL_LOG_DECLARE();
};

} // namespace http_arpc
