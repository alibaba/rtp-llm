#pragma once

#include "kmonitor/client/MetricsReporter.h"
#include <cstdint>
#include <chrono>

namespace rtp_llm {

inline constexpr const char* KMON_SERVICE_STATUS_TAG    = "is_serving";
inline constexpr const char* KMON_SERVICE_STATUS_METRIC = "rtp_llm_service_status";

// The serving bit is process-wide: every reporter in a backend process must
// attach the same lifecycle state, including reporters owned by cache/RPC
// helper threads.
// Start/stop alongside the KMonitor factory, using the backend reporter's tags.
void                                  startKmonServiceStatus(const kmonitor::MetricsReporterPtr& reporter);
void                                  stopKmonServiceStatus();
void                                  reportKmonServiceStatus();
void                                  setKmonServiceServing(bool serving);
bool                                  isKmonServiceServing();
bool                                  isKmonMetricReportingEnabled();
std::chrono::steady_clock::time_point kmonMetricReportingStartTime();

kmonitor::MetricsTags kmonTagsWithServiceStatus(const kmonitor::MetricsTags* tags);

inline void reportKmonMetric(kmonitor::MutableMetric* metric, const kmonitor::MetricsTags* tags, double value) {
    if (metric == nullptr || !isKmonMetricReportingEnabled()) {
        return;
    }
    auto service_tags = kmonTagsWithServiceStatus(tags);
    metric->Report(&service_tags, value);
}

class ScopedKmonMetricsSuppression {
public:
    ScopedKmonMetricsSuppression();
    ~ScopedKmonMetricsSuppression();

    ScopedKmonMetricsSuppression(const ScopedKmonMetricsSuppression&)            = delete;
    ScopedKmonMetricsSuppression& operator=(const ScopedKmonMetricsSuppression&) = delete;
};

}  // namespace rtp_llm

// Explicit RTP macros keep third-party report macros unchanged.
#define RTP_REPORT_MUTABLE_METRIC(metric, value)                                                                       \
    do {                                                                                                               \
        rtp_llm::reportKmonMetric((metric), tags, (value));                                                            \
    } while (0)

#define RTP_REPORT_MUTABLE_QPS(metric) RTP_REPORT_MUTABLE_METRIC((metric), 1)
