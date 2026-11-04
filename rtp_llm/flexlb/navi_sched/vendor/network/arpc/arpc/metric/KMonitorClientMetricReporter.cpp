#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/metric/KMonitorClientMetricReporter.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/metric/ClientRPCStats.h"
#include "kmonitor/client/KMonitorFactory.h"

namespace arpc {

AUTIL_LOG_SETUP(arpc, KMonitorClientMetricReporter);

const std::string kClientKMonitorName = "arpc_client_kmonitor";

KMonitorClientMetricReporter::KMonitorClientMetricReporter(const MetricReporterConfig &config,
                                                           kmonitor::MetricLevel level)
    : ClientMetricReporter(config), _level(level), _defaultTags("arpc_type", _config.type) {}

bool KMonitorClientMetricReporter::init() {
    if (!_enable) {
        return true;
    }

    auto kMonitor = kmonitor::KMonitorFactory::GetKMonitor(kClientKMonitorName);
    if (kMonitor == nullptr) {
        _enable = false;
        AUTIL_LOG(WARN, "arpc kmonitor client metric reporter init failed");
        return false;
    }
    std::string serviceName = "arpc.client";
    if (!_config.metricPrefix.empty()) {
        serviceName = _config.metricPrefix + "." + serviceName;
    }
    kMonitor->SetServiceName(serviceName);

#define CHECK_METRIC_NULLPTR(metric)                                                                                   \
    if (metric == nullptr) {                                                                                           \
        _enable = false;                                                                                               \
        AUTIL_LOG(WARN, "arpc kmonitor client metric reporter init failed, regist metric %s failed", #metric);         \
        return false;                                                                                                  \
    }

    _clientQps.reset(kMonitor->RegisterMetric("qps", kmonitor::QPS, _level));
    CHECK_METRIC_NULLPTR(_clientQps);

    _errorQps.reset(kMonitor->RegisterMetric("error_qps", kmonitor::QPS, _level));
    CHECK_METRIC_NULLPTR(_errorQps);

    _clientTotalCost.reset(kMonitor->RegisterMetric("total_cost_in_us", kmonitor::GAUGE, _level));
    CHECK_METRIC_NULLPTR(_clientTotalCost);

    _clientRequestPendingCost.reset(kMonitor->RegisterMetric("request_pending_cost", kmonitor::GAUGE, _level));
    CHECK_METRIC_NULLPTR(_clientRequestPendingCost);

    _clientRequestPackCost.reset(kMonitor->RegisterMetric("request_pack_cost", kmonitor::GAUGE, _level));
    CHECK_METRIC_NULLPTR(_clientRequestPackCost);

    _clientRequestSendCost.reset(kMonitor->RegisterMetric("request_send_cost", kmonitor::GAUGE, _level));
    CHECK_METRIC_NULLPTR(_clientRequestSendCost);

    _clientRequestIovSize.reset(kMonitor->RegisterMetric("request_iov_size", kmonitor::GAUGE, _level));
    CHECK_METRIC_NULLPTR(_clientRequestIovSize);

    _clientResponseWaitCost.reset(kMonitor->RegisterMetric("response_wait_cost", kmonitor::GAUGE, _level));
    CHECK_METRIC_NULLPTR(_clientResponseWaitCost);

    _clientResponsePopCost.reset(kMonitor->RegisterMetric("response_pop_cost", kmonitor::GAUGE, _level));
    CHECK_METRIC_NULLPTR(_clientResponsePopCost);

    _clientResponseUnpackCost.reset(kMonitor->RegisterMetric("response_unpack_cost", kmonitor::GAUGE, _level));
    CHECK_METRIC_NULLPTR(_clientResponseUnpackCost);

#undef CHECK_METRIC_NULLPTR

    AUTIL_LOG(INFO, "arpc kmonitor client metric reporter init done");
    return true;
}

void KMonitorClientMetricReporter::doReportMetric(ClientRPCStats *stats) {
    if (stats == nullptr || !_enable) {
        return;
    }

    _clientQps->Report(&_defaultTags, 1);
    _clientTotalCost->Report(&_defaultTags, stats->totalCostUs());
    _clientRequestPendingCost->Report(&_defaultTags, stats->requestPendingCostUs());
    _clientRequestPackCost->Report(&_defaultTags, stats->requestPackCostUs());
    _clientRequestSendCost->Report(&_defaultTags, stats->requestSendCostUs());
    _clientResponseWaitCost->Report(&_defaultTags, stats->responseWaitCostUs());
    _clientResponsePopCost->Report(&_defaultTags, stats->responsePopCostUs());
    _clientResponseUnpackCost->Report(&_defaultTags, stats->responseUnpackCostUs());

    auto &iovSizes = stats->iovSizes();
    kmonitor::MetricsTagsPtr iovTags;
        for (auto &iov : iovSizes) {
            if (_config.enableIovTag) {
                iovTags.reset(new kmonitor::MetricsTags(
                    {{"device", iov.device}, {"remote_device", iov.remoteDevice}, {"arpc_type", _config.type}}));
            }
            _clientRequestIovSize->Report(iovTags.get(), iov.size);
    }
    if (!stats->success()) {
        kmonitor::MetricsTags tags({{"error", ErrorCode_Name(stats->error())}, {"arpc_type", _config.type}});
        _errorQps->Report(&tags, 1);
    }
}

} // namespace arpc
