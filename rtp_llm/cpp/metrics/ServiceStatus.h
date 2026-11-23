#pragma once

#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

// Draining is terminal for this backend process. Keep this independent of RPC
// admission so in-flight requests continue to report until the engine stops.
void                  markKmonServiceDraining();
kmonitor::MetricsTags kmonTagsWithServiceStatus(const kmonitor::MetricsTags* tags);

inline void reportKmonMetric(kmonitor::MutableMetric* metric, const kmonitor::MetricsTags* tags, double value) {
    if (metric) {
        auto service_tags = kmonTagsWithServiceStatus(tags);
        metric->Report(&service_tags, value);
    }
}

}  // namespace rtp_llm

#define RTP_REPORT_MUTABLE_METRIC(metric, value) ::rtp_llm::reportKmonMetric((metric), tags, (value))
#define RTP_REPORT_MUTABLE_QPS(metric) RTP_REPORT_MUTABLE_METRIC((metric), 1)
