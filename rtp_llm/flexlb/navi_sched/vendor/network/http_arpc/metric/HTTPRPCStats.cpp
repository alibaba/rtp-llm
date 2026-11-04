#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/metric/HTTPRPCStats.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/metric/KMonitorHTTPRPCServerMetricReporter.h"

namespace http_arpc {

AUTIL_LOG_SETUP(http_arpc, HTTPRPCStats);

HTTPRPCStats::~HTTPRPCStats() {
    if (auto reporter = _metricReporter.lock()) {
        reporter->reportHTTPRPCStats(this);
    }
}

} // namespace http_arpc
