#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/metric/MetricReporterConfig.h"

#include "autil/EnvUtil.h"

namespace arpc {

const std::string kReportArpcMetricsEnvName = "ARPC_METRIC_ENABLE";
const std::string kReportArpcMetricPrefixEnvName = "ARPC_METRIC_PREFIX";
const std::string kReportArpcIovEnvName = "ARPC_IOV_ENABLE";
const std::string kShowSlowRequestLogEnvName = "ARPC_SLOW_REQUEST_LOG";
const std::string kSlowRequestThresholdInUsEnvName = "ARPC_SLOW_REQUEST_THRESHOLD_IN_US";

MetricReporterConfig::MetricReporterConfig(bool enable) : enableArpcMetric(enable) 
{
    enableArpcMetric = autil::EnvUtil::getEnv(kReportArpcMetricsEnvName, enableArpcMetric);
    enableIovTag = autil::EnvUtil::getEnv(kReportArpcIovEnvName, enableIovTag);
    metricPrefix = autil::EnvUtil::getEnv(kReportArpcMetricPrefixEnvName, metricPrefix);
    enableSlowRequestLog = autil::EnvUtil::getEnv(kShowSlowRequestLogEnvName, enableSlowRequestLog);
    slowRequestThresholdInUs = autil::EnvUtil::getEnv(kSlowRequestThresholdInUsEnvName, slowRequestThresholdInUs);
}

void MetricReporterConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper& json)
{
    json.Jsonize("enable_arpc_metric", enableArpcMetric, enableArpcMetric);
    json.Jsonize("enable_slow_request_log", enableSlowRequestLog, enableSlowRequestLog);
    json.Jsonize("enable_iov_tag", enableIovTag, enableIovTag);
    json.Jsonize("slow_request_threshold_us", slowRequestThresholdInUs, slowRequestThresholdInUs);
    json.Jsonize("type", type, type);
    json.Jsonize("metric_prefix", metricPrefix, metricPrefix);
}

} // namespace arpc
