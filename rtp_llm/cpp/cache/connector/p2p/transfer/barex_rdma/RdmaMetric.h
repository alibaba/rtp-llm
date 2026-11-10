#pragma once

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

class RdmaMetricsCollector final {
public:
    bool    success              = true;
    int64_t start_time_us        = 0;
    int64_t pending_done_time_us = 0;
    int64_t write_done_time_us   = 0;
    int64_t total_block_size     = 0;
    int64_t block_count          = 0;
};

/// @brief RDMA 传输层指标，上报单次 RDMA Read 的延迟和成功率
class RdmaMetric: public kmonitor::MetricsGroup {
public:
    ~RdmaMetric() = default;

public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RdmaMetricsCollector* collector);

private:
    kmonitor::MutableMetric* rdma_qps_metric                = nullptr;
    kmonitor::MutableMetric* rdma_error_qps_metric          = nullptr;
    kmonitor::MutableMetric* rdma_pending_latency_us_metric = nullptr;
    kmonitor::MutableMetric* rdma_write_latency_us_metric   = nullptr;
    kmonitor::MutableMetric* rdma_total_block_size_metric   = nullptr;
    kmonitor::MutableMetric* rdma_block_count_metric        = nullptr;
};

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm