#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/metric/HTTPRPCMetricReporterConfig.h"

#include "autil/EnvUtil.h"

namespace http_arpc {

const std::string kReportHttpRpcMetricsEnvName = "HTTP_RPC_METRIC_ENABLE";
const std::string kReportHttpRpcMetricPrefixEnvName = "HTTP_RPC_METRIC_PREFIX";

HTTPRPCMetricReporterConfig::HTTPRPCMetricReporterConfig(bool enable) : enableHttpRpcMetric(enable) {
    enableHttpRpcMetric = autil::EnvUtil::getEnv(kReportHttpRpcMetricsEnvName, enableHttpRpcMetric);
    metricPrefix = autil::EnvUtil::getEnv(kReportHttpRpcMetricPrefixEnvName, metricPrefix);
}

void HTTPRPCMetricReporterConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
    json.Jsonize("enable_http_rpc_metric", enableHttpRpcMetric, enableHttpRpcMetric);
    json.Jsonize("type", type, type);
    json.Jsonize("metric_prefix", metricPrefix, metricPrefix);
}

} // namespace http_arpc
