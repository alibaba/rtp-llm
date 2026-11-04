#pragma once

#include "autil/legacy/jsonizable.h"

namespace arpc {

class MetricReporterConfig : public autil::legacy::Jsonizable
{
public:
    MetricReporterConfig(bool enable = false);
    ~MetricReporterConfig() = default;

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;

public:
    bool enableArpcMetric {false};
    bool enableSlowRequestLog {false};
    bool enableIovTag {false};
    uint64_t slowRequestThresholdInUs {20 * 1000};
    std::string type{"unknown"};
    std::string metricPrefix;
};

} // namespace arpc
