#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/metric/KMonitorHTTPRPCServerMetricReporter.h"

#include "rtp_llm/flexlb/navi_sched/vendor/kmonitor/cpp_client/src/kmonitor/client/KMonitorFactory.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/metric/KMonitorANetMetricReporter.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/metric/HTTPRPCStats.h"

namespace http_arpc {

AUTIL_LOG_SETUP(http_arpc, KMonitorHTTPRPCServerMetricReporter);

const std::string kHttpRpcServerKMonitorName = "http_arpc_server_kmonitor";

KMonitorHTTPRPCServerMetricReporter::KMonitorHTTPRPCServerMetricReporter(const HTTPRPCMetricReporterConfig &config,
                                                                         kmonitor::MetricLevel level)
    : _config(config), _level(level), _defaultTags("http_arpc_type", _config.type) {
    _enable = _config.enableHttpRpcMetric;
}

bool KMonitorHTTPRPCServerMetricReporter::init(anet::Transport *transport) {
    if (!_enable) {
        return true;
    }

    // 初始化HTTP RPC指标
    auto kMonitor = kmonitor::KMonitorFactory::GetKMonitor(kHttpRpcServerKMonitorName);
    if (!kMonitor) {
        _enable = false;
        AUTIL_LOG(WARN, "http arpc server kmonitor metric reporter init failed");
        return false;
    }
    std::string serviceName = "http_arpc.server";
    if (!_config.metricPrefix.empty()) {
        serviceName = _config.metricPrefix + "." + serviceName;
    }
    kMonitor->SetServiceName(serviceName);

#define CHECK_METRIC_NULLPTR(metric)                                                                                   \
    if (metric == nullptr) {                                                                                           \
        _enable = false;                                                                                               \
        AUTIL_LOG(WARN, "http arpc server kmonitor metric reporter init failed, regist metric %s failed", #metric);    \
        return false;                                                                                                  \
    }

    _httpQps.reset(kMonitor->RegisterMetric("qps", kmonitor::QPS, _level));
    CHECK_METRIC_NULLPTR(_httpQps);

    _errorQps.reset(kMonitor->RegisterMetric("error_qps", kmonitor::QPS, _level));
    CHECK_METRIC_NULLPTR(_errorQps);

    _httpTotalCost.reset(kMonitor->RegisterMetric("total_cost_in_us", kmonitor::GAUGE, _level));
    CHECK_METRIC_NULLPTR(_httpTotalCost);

    _httpRequestParseCost.reset(kMonitor->RegisterMetric("request_parse_cost", kmonitor::GAUGE, _level));
    CHECK_METRIC_NULLPTR(_httpRequestParseCost);

    _httpRequestPendingCost.reset(kMonitor->RegisterMetric("request_pending_cost", kmonitor::GAUGE, _level));
    CHECK_METRIC_NULLPTR(_httpRequestPendingCost);

    _httpRequestCallCost.reset(kMonitor->RegisterMetric("request_call_cost", kmonitor::GAUGE, _level));
    CHECK_METRIC_NULLPTR(_httpRequestCallCost);

    _httpResponsePackCost.reset(kMonitor->RegisterMetric("response_pack_cost", kmonitor::GAUGE, _level));
    CHECK_METRIC_NULLPTR(_httpResponsePackCost);

    _httpResponseSendCost.reset(kMonitor->RegisterMetric("response_send_cost", kmonitor::GAUGE, _level));
    CHECK_METRIC_NULLPTR(_httpResponseSendCost);

    _httpResponseSize.reset(kMonitor->RegisterMetric("response_size", kmonitor::GAUGE, _level));
    CHECK_METRIC_NULLPTR(_httpResponseSize);

    _httpThreadItemCount.reset(kMonitor->RegisterMetric("threadpool_item_count", kmonitor::GAUGE, _level));
    CHECK_METRIC_NULLPTR(_httpThreadItemCount);

#undef CHECK_METRIC_NULLPTR

    // 初始化ANET指标报告器
    if (transport) {
        anet::ANetMetricReporterConfig anetConfig(true);
        anetConfig.metricPrefix = _config.metricPrefix;
        _anetMetricReporter = std::make_shared<anet::KMonitorANetMetricReporter>(anetConfig, _level);
        if (!_anetMetricReporter->init(transport)) {
            AUTIL_LOG(WARN, "init anet metric reporter failed");
        }
    }

    AUTIL_LOG(INFO, "http arpc server kmonitor metric reporter init done");
    _initialized = true;
    return true;
}

std::shared_ptr<HTTPRPCStats> KMonitorHTTPRPCServerMetricReporter::makeHTTPRPCStats() {
    if (!_enable) {
        return nullptr;
    }
    return std::make_shared<HTTPRPCStats>(shared_from_this());
}

void KMonitorHTTPRPCServerMetricReporter::reportHTTPRPCStats(HTTPRPCStats *stats) {
    if (!_enable || stats == nullptr) {
        return;
    }
    doReportMetric(stats);
}

void KMonitorHTTPRPCServerMetricReporter::doReportMetric(HTTPRPCStats *stats) {
    if (stats == nullptr || !_enable || !_initialized) {
        return;
    }

    _httpQps->Report(&_defaultTags, 1);

    if (stats->totalCostUs() > 0) {
        _httpTotalCost->Report(&_defaultTags, stats->totalCostUs());
    }
    if (stats->requestParseCostUs() > 0) {
        _httpRequestParseCost->Report(&_defaultTags, stats->requestParseCostUs());
    }
    if (stats->requestPendingCostUs() > 0) {
        _httpRequestPendingCost->Report(&_defaultTags, stats->requestPendingCostUs());
    }
    if (stats->requestCallCostUs() > 0) {
        _httpRequestCallCost->Report(&_defaultTags, stats->requestCallCostUs());
    }
    if (stats->responsePackCostUs() > 0) {
        _httpResponsePackCost->Report(&_defaultTags, stats->responsePackCostUs());
    }
    if (stats->responseSendCostUs() > 0) {
        _httpResponseSendCost->Report(&_defaultTags, stats->responseSendCostUs());
    }

    _httpResponseSize->Report(&_defaultTags, stats->responseSize());

    // 错误QPS
    if (!stats->success()) {
        kmonitor::MetricsTags tags({{"http_status", std::to_string(stats->httpStatus())},
                                    {"http_method", stats->httpMethod()},
                                    {"http_arpc_type", _config.type}});
        _errorQps->Report(&tags, 1);
    }

    // 线程池队列长度总是汇报，0也是有效值（空队列）
    _httpThreadItemCount->Report(&_defaultTags, stats->threadPoolItemCount());
}

} // namespace http_arpc
