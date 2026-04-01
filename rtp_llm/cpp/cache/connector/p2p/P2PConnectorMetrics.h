#pragma once

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

class P2PConnectorMetrics;

class DecodeSchedulerMetricsCollector final {
public:
    DecodeSchedulerMetricsCollector(const std::shared_ptr<kmonitor::MetricsReporter>& metrics_reporter):
        start_time_us(currentTimeUs()), metrics_reporter_(metrics_reporter) {}
    ~DecodeSchedulerMetricsCollector() {
        if (metrics_reporter_) {
            metrics_reporter_->report<P2PConnectorMetrics, DecodeSchedulerMetricsCollector>(nullptr, this);
        }
    }

public:
    bool    success                  = true;
    int64_t start_time_us            = 0;
    int64_t server_call_cost_time_us = 0;
    int64_t tp_sync_cost_time_us     = 0;
    int64_t total_cost_time_us       = 0;

private:
    std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter_;
};

class DecodeWorkerMetricsCollector final {
public:
    DecodeWorkerMetricsCollector()  = default;
    ~DecodeWorkerMetricsCollector() = default;

public:
    bool    success                  = true;
    int64_t total_block_count        = 0;
    int64_t first_layer_wait_time_us = 0;
    int64_t total_cost_time_us       = 0;
};

class DecodeSchedulerStatusMetricsCollector final {
public:
    int64_t check_once_cost_time_us = 0;
    int64_t inflight_context_count  = 0;
};

class PrefillSchedulerMetricsCollector final {
public:
    PrefillSchedulerMetricsCollector()  = default;
    ~PrefillSchedulerMetricsCollector() = default;

public:
    bool    success            = true;
    int64_t total_cost_time_us = 0;
};

class PrefillWorkerStoreMetricsCollector final {
public:
    PrefillWorkerStoreMetricsCollector(): start_time_us(currentTimeUs()) {}
    ~PrefillWorkerStoreMetricsCollector() = default;

public:
    bool    success                 = true;
    int64_t total_block_count       = 0;
    int64_t store_wait_done_time_us = 0;
    int64_t start_time_us           = 0;
};

class PrefillWorkerStatusMetricsCollector final {
public:
    int64_t wait_store_event_count = 0;
    int64_t task_count             = 0;
    int64_t computed_request_count = 0;
};

class PrefillWorkerSendMetricsCollector final {
public:
    PrefillWorkerSendMetricsCollector()  = default;
    ~PrefillWorkerSendMetricsCollector() = default;

public:
    bool    success                  = true;
    int64_t first_layer_wait_time_us = 0;
    int64_t last_layer_wait_time_us  = 0;
    int64_t total_cost_time_us       = 0;
};

class StreamStoreCountMetricsCollector final {
public:
    int64_t stream_count = 0;
};

class StreamStoreWaitMetricsCollector final {
public:
    bool    timeout             = false;
    bool    cancelled           = false;
    int64_t stream_wait_time_us = 0;
};

/// @brief P2P connector 指标上报，聚合 Decode/Prefill 两侧的调度和传输指标
class P2PConnectorMetrics: public kmonitor::MetricsGroup {
public:
    P2PConnectorMetrics()  = default;
    ~P2PConnectorMetrics() = default;

public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, DecodeSchedulerMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, DecodeWorkerMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, DecodeSchedulerStatusMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, StreamStoreCountMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, StreamStoreWaitMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, PrefillSchedulerMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, PrefillWorkerSendMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, PrefillWorkerStatusMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, PrefillWorkerStoreMetricsCollector* collector);

private:
    // decode schedule metrics
    kmonitor::MutableMetric* decode_schedule_qps_metric          = nullptr;
    kmonitor::MutableMetric* decode_schedule_failed_qps_metric   = nullptr;
    kmonitor::MutableMetric* decode_schedule_cost_time_us_metric = nullptr;

    // decode worker metrics
    kmonitor::MutableMetric* decode_worker_qps_metric                      = nullptr;
    kmonitor::MutableMetric* decode_worker_failed_qps_metric               = nullptr;
    kmonitor::MutableMetric* decode_worker_total_block_count_metric        = nullptr;
    kmonitor::MutableMetric* decode_worker_first_layer_wait_time_us_metric = nullptr;
    kmonitor::MutableMetric* decode_worker_total_cost_time_us_metric       = nullptr;

    // decode scheduler status metrics
    kmonitor::MutableMetric* decode_scheduler_check_once_cost_time_us_metric = nullptr;
    kmonitor::MutableMetric* decode_scheduler_inflight_context_count_metric  = nullptr;

    // stream store metrics
    kmonitor::MutableMetric* stream_store_stream_count_metric        = nullptr;
    kmonitor::MutableMetric* stream_store_qps_metric                 = nullptr;
    kmonitor::MutableMetric* stream_store_timeout_qps_metric         = nullptr;
    kmonitor::MutableMetric* stream_store_cancel_qps_metric          = nullptr;
    kmonitor::MutableMetric* stream_store_stream_wait_time_us_metric = nullptr;

    // prefill scheduler metrics
    kmonitor::MutableMetric* prefill_scheduler_qps_metric                = nullptr;
    kmonitor::MutableMetric* prefill_scheduler_failed_qps_metric         = nullptr;
    kmonitor::MutableMetric* prefill_scheduler_total_cost_time_us_metric = nullptr;

    // prefill worker metrics
    kmonitor::MutableMetric* prefill_worker_store_qps_metric                     = nullptr;
    kmonitor::MutableMetric* prefill_worker_store_failed_qps_metric              = nullptr;
    kmonitor::MutableMetric* prefill_worker_store_total_block_count_metric       = nullptr;
    kmonitor::MutableMetric* prefill_worker_store_store_wait_done_time_us_metric = nullptr;

    // prefill worker write metrics
    kmonitor::MutableMetric* prefill_worker_write_qps_metric                      = nullptr;
    kmonitor::MutableMetric* prefill_worker_write_failed_qps_metric               = nullptr;
    kmonitor::MutableMetric* prefill_worker_write_first_layer_wait_time_us_metric = nullptr;
    kmonitor::MutableMetric* prefill_worker_write_last_layer_wait_time_us_metric  = nullptr;
    kmonitor::MutableMetric* prefill_worker_write_total_cost_time_us_metric       = nullptr;

    // prefill worker status metrics
    kmonitor::MutableMetric* prefill_worker_wait_store_event_count_metric = nullptr;
    kmonitor::MutableMetric* prefill_worker_task_count_metric             = nullptr;
    kmonitor::MutableMetric* prefill_worker_computed_request_count_metric = nullptr;
};

}  // namespace rtp_llm