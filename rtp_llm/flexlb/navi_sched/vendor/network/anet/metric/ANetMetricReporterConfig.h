#pragma once

#include "autil/EnvUtil.h"
#include "autil/legacy/jsonizable.h"

namespace anet {

class ANetMetricReporterConfig : public autil::legacy::Jsonizable {
public:
    ANetMetricReporterConfig(bool enable = false) : enableANetMetric(enable) {
        enableANetMetric = autil::EnvUtil::getEnv("ANET_METRIC_ENABLE", enableANetMetric);
        intervalMs = autil::EnvUtil::getEnv("ANET_METRIC_INTERVAL_MS", intervalMs);
        metricPrefix = autil::EnvUtil::getEnv("ANET_METRIC_PREFIX", metricPrefix);
        enableAddressTag = autil::EnvUtil::getEnv("ANET_METRIC_ENABLE_ADDRESS_TAG", enableAddressTag);
        fastMode = autil::EnvUtil::getEnv("ANET_METRIC_FAST_MODE", fastMode);
    }
    ~ANetMetricReporterConfig() = default;

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override {
        json.Jsonize("enable_anet_metric", enableANetMetric, enableANetMetric);
        json.Jsonize("interval_ms", intervalMs, intervalMs);
        json.Jsonize("metric_prefix", metricPrefix, metricPrefix);
        json.Jsonize("enable_address_tag", enableAddressTag, enableAddressTag);
        json.Jsonize("fast_mode", fastMode, fastMode);
    }

public:
    bool enableANetMetric{false};
    int64_t intervalMs{1000};
    std::string metricPrefix;
    bool enableAddressTag{false};
    bool fastMode{true}; // fast mode only work with interval ms = 1000
};

} // namespace anet
