#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/metric/KMonitorANetServerMetricReporter.h"

namespace arpc {

AUTIL_LOG_SETUP(arpc, KMonitorANetServerMetricReporter);

KMonitorANetServerMetricReporter::KMonitorANetServerMetricReporter(const KMonitorANetMetricReporterConfig &config)
    : ServerMetricReporter(config.arpcConfig), _config{config} {}

bool KMonitorANetServerMetricReporter::init(anet::Transport *transport) {
    if (!_enable) {
        return true;
    }

    _arpcReporter = std::make_shared<KMonitorServerMetricReporter>(_config.arpcConfig, _config.metricLevel);
    if (_arpcReporter == nullptr || !_arpcReporter->init()) {
        AUTIL_LOG(WARN, "kmonitor ANet server metric reporter init failed, arpc reporter init failed");
        _enable = false;
        return false;
    }

    _anetReporter = std::make_shared<anet::KMonitorANetMetricReporter>(_config.anetConfig, _config.metricLevel);
    if (_anetReporter == nullptr || !_anetReporter->init(transport)) {
        AUTIL_LOG(WARN, "kmonitor ANet server metric reporter init failed, accl reporter init failed");
        _enable = false;
        return false;
    }
    AUTIL_LOG(INFO, "kmonitor ANet server metric reporter init done");
    return true;
}

void KMonitorANetServerMetricReporter::doReportMetric(ServerRPCStats *stats) {
    if (_arpcReporter != nullptr && stats != nullptr) {
        _arpcReporter->doReportMetric(stats);
    }
}

void KMonitorANetServerMetricReporter::stop() {
    _enable = false;
    _anetReporter.reset();
}

} // namespace arpc
