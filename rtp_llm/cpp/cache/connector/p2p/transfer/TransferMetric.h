#pragma once

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {

class TransferClientTransferMetricsCollector final {
public:
    TransferClientTransferMetricsCollector()  = default;
    ~TransferClientTransferMetricsCollector() = default;

public:
    bool    success          = true;
    int64_t block_count      = 0;
    int64_t total_block_size = 0;
    int64_t latency_us       = 0;
};

class TransferServerTransferMetricsCollector final {
public:
    TransferServerTransferMetricsCollector()  = default;
    ~TransferServerTransferMetricsCollector() = default;

public:
    bool    success                  = true;
    int64_t block_count              = 0;
    int64_t total_block_size         = 0;
    int64_t wait_task_run_latency_us = 0;
    int64_t total_cost_latency_us    = 0;
};

class TransferMetric: public kmonitor::MetricsGroup {
public:
    ~TransferMetric() = default;

public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, TransferClientTransferMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, TransferServerTransferMetricsCollector* collector);

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

}  // namespace rtp_llm