#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/metric/ServerMetricReporter.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/metric/ServerRPCStats.h"

namespace arpc {

AUTIL_LOG_SETUP(arpc, ServerMetricReporter);

ServerMetricReporter::ServerMetricReporter(const MetricReporterConfig &config) : _config(config) {
    if (_config.type.empty()) {
        _config.type = "unknown";
    }
    _enable = _config.enableArpcMetric || _config.enableSlowRequestLog;
}

std::shared_ptr<ServerRPCStats> ServerMetricReporter::makeRPCStats() {
    if (!_enable) {
        return nullptr;
    }
    return std::make_shared<ServerRPCStats>(shared_from_this());
}

void ServerMetricReporter::reportRPCStats(ServerRPCStats *stats) {
    if (stats == nullptr || !_enable) {
        return;
    }

    doReportMetric(stats);

    if (_config.enableSlowRequestLog && _config.slowRequestThresholdInUs < stats->totalCostUs()) {
        AUTIL_LOG(WARN,
                  "slow server request. requestId: %d, totalCostUs: %ld, requestUnpack: %ld, requestPending: %ld, "
                  "requestCall: %ld, responsePack: %ld, responseSend: %ld",
                  stats->requestId(),
                  stats->totalCostUs(),
                  stats->requestUnpackCostUs(),
                  stats->requestPendingCostUs(),
                  stats->requestCallCostUs(),
                  stats->responsePackCostUs(),
                  stats->responseSendCostUs());
    }
}
} // namespace arpc
