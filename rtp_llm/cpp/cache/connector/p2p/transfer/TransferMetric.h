#pragma once

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {
namespace transfer {

class TransferClientMetricsCollector final {
public:
    TransferClientMetricsCollector()  = default;
    ~TransferClientMetricsCollector() = default;

public:
    bool    success          = true;
    int64_t block_count      = 0;
    int64_t total_block_size = 0;
    int64_t latency_us       = 0;
};

class TransferServerMetricsCollector final {
public:
    TransferServerMetricsCollector()  = default;
    ~TransferServerMetricsCollector() = default;

public:
    bool    success                  = true;
    int64_t block_count              = 0;
    int64_t total_block_size         = 0;
    int64_t wait_task_run_latency_us = 0;
    int64_t total_cost_latency_us    = 0;
};

/// @brief KV cache 传输层指标，覆盖发送端（Prefill）和接收端（Decode）
class TransferMetric: public kmonitor::MetricsGroup {
public:
    ~TransferMetric() = default;

public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, TransferClientMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, TransferServerMetricsCollector* collector);

private:
    // client metrics
    kmonitor::MutableMetric* transfer_client_qps_metric              = nullptr;
    kmonitor::MutableMetric* transfer_client_error_qps_metric        = nullptr;
    kmonitor::MutableMetric* transfer_client_block_count_metric      = nullptr;
    kmonitor::MutableMetric* transfer_client_total_block_size_metric = nullptr;
    kmonitor::MutableMetric* transfer_client_latency_us_metric       = nullptr;

    // server metrics
    kmonitor::MutableMetric* transfer_server_qps_metric                  = nullptr;
    kmonitor::MutableMetric* transfer_server_error_qps_metric            = nullptr;
    kmonitor::MutableMetric* transfer_server_block_count_metric          = nullptr;
    kmonitor::MutableMetric* transfer_server_total_block_size_metric     = nullptr;
    kmonitor::MutableMetric* transfer_server_wait_task_latency_us_metric = nullptr;
    kmonitor::MutableMetric* transfer_server_latency_us_metric           = nullptr;
};

}  // namespace transfer
}  // namespace rtp_llm
