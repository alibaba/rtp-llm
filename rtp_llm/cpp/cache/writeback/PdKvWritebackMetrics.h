#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "kmonitor/client/MetricsReporter.h"

namespace kmonitor {
class MetricsTags;
class MutableMetric;
}  // namespace kmonitor

namespace rtp_llm {

class PdKvWritebackMetricsCollector final {
public:
    bool    launch_qps         = false;
    bool    launch_failed_qps  = false;
    bool    launch_skipped_qps = false;
    int64_t launch_latency_us  = 0;
    bool    launch_rate_valid  = false;
    double  launch_rate        = 0.0;

    bool    rpc_qps        = false;
    bool    rpc_failed_qps = false;
    int64_t rpc_latency_us = 0;

    bool    transfer_qps        = false;
    bool    transfer_failed_qps = false;
    int64_t transfer_latency_us = 0;

    bool    receive_qps        = false;
    bool    receive_failed_qps = false;
    int64_t receive_latency_us = 0;

    int64_t malloc_latency_us = 0;
    int64_t commit_latency_us = 0;
    int64_t block_count       = 0;
    int64_t token_count       = 0;
};

class PdKvWritebackMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, PdKvWritebackMetricsCollector* collector);

private:
    kmonitor::MutableMetric* launch_qps_metric          = nullptr;
    kmonitor::MutableMetric* launch_failed_qps_metric   = nullptr;
    kmonitor::MutableMetric* launch_skipped_qps_metric  = nullptr;
    kmonitor::MutableMetric* launch_latency_us_metric   = nullptr;
    kmonitor::MutableMetric* launch_rate_metric         = nullptr;
    kmonitor::MutableMetric* rpc_qps_metric             = nullptr;
    kmonitor::MutableMetric* rpc_failed_qps_metric      = nullptr;
    kmonitor::MutableMetric* rpc_latency_us_metric      = nullptr;
    kmonitor::MutableMetric* transfer_qps_metric        = nullptr;
    kmonitor::MutableMetric* transfer_failed_qps_metric = nullptr;
    kmonitor::MutableMetric* transfer_latency_us_metric = nullptr;
    kmonitor::MutableMetric* receive_qps_metric         = nullptr;
    kmonitor::MutableMetric* receive_failed_qps_metric  = nullptr;
    kmonitor::MutableMetric* receive_latency_us_metric  = nullptr;
    kmonitor::MutableMetric* malloc_latency_us_metric   = nullptr;
    kmonitor::MutableMetric* commit_latency_us_metric   = nullptr;
    kmonitor::MutableMetric* block_count_metric         = nullptr;
    kmonitor::MutableMetric* token_count_metric         = nullptr;
};

using PdKvWritebackMetricExtraTags = std::vector<std::pair<std::string, std::string>>;

void reportPdKvWritebackMetric(const kmonitor::MetricsReporterPtr& reporter,
                               PdKvWritebackMetricsCollector&      collector,
                               const std::string&                  stage,
                               const std::string&                  status,
                               const std::string&                  reason,
                               const std::string&                  role,
                               int32_t                             tp_size,
                               const std::string&                  topology,
                               const PdKvWritebackMetricExtraTags& extra_tags = PdKvWritebackMetricExtraTags());

}  // namespace rtp_llm
