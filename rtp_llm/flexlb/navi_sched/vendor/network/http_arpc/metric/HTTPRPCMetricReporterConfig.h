#pragma once

#include "autil/legacy/jsonizable.h"

namespace http_arpc {

class HTTPRPCMetricReporterConfig : public autil::legacy::Jsonizable {
public:
    HTTPRPCMetricReporterConfig(bool enable = false);
    ~HTTPRPCMetricReporterConfig() = default;

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) override;

public:
    bool enableHttpRpcMetric{false};
    std::string type{"http_rpc"};
    std::string metricPrefix;
};

} // namespace http_arpc
