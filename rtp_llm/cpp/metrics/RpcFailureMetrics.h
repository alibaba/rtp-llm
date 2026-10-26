#pragma once

#include <functional>
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {

inline std::function<void(const std::string&)>
makeConnectFailureReporter(std::shared_ptr<kmonitor::MetricsReporter> reporter) {
    if (!reporter) {
        return {};
    }
    return [reporter = std::move(reporter)](const std::string& direction) {
        RtpLLMGrpcConnectFailureMetricsCollector collector;
        collector.direction = direction;
        reporter->report<RtpLLMGrpcConnectFailureMetrics, RtpLLMGrpcConnectFailureMetricsCollector>(nullptr,
                                                                                                    &collector);
    };
}

}  // namespace rtp_llm
