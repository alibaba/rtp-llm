#pragma once

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

class P2PConnectorMetrics;

class P2PConnectorClientSchedulerMetricsCollector final {
public:
    P2PConnectorClientSchedulerMetricsCollector(const std::shared_ptr<kmonitor::MetricsReporter>& metrics_reporter):
        start_time_us(currentTimeUs()), metrics_reporter_(metrics_reporter) {}
    ~P2PConnectorClientSchedulerMetricsCollector() {
        if (metrics_reporter_) {
            metrics_reporter_->report<P2PConnectorMetrics, P2PConnectorClientSchedulerMetricsCollector>(nullptr, this);
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

class P2PConnectorClientWorkerMetricsCollector final {
public:
    P2PConnectorClientWorkerMetricsCollector()  = default;
    ~P2PConnectorClientWorkerMetricsCollector() = default;

public:
    bool    success                  = true;
    int64_t total_block_count        = 0;
    int64_t first_layer_wait_time_us = 0;
    int64_t total_cost_time_us       = 0;
};

class P2PConnectorClientSchedulerStatusMetricsCollector final {
public:
    int64_t check_once_cost_time_us = 0;
    int64_t inflight_context_count  = 0;
};

class P2PConnectorServerSchedulerMetricsCollector final {
public:
    P2PConnectorServerSchedulerMetricsCollector()  = default;
    ~P2PConnectorServerSchedulerMetricsCollector() = default;

public:
    bool    success            = true;
    int64_t total_cost_time_us = 0;
};

class P2PConnectorServerWorkerStoreMetricsCollector final {
public:
    P2PConnectorServerWorkerStoreMetricsCollector(): start_time_us(currentTimeUs()) {}
    ~P2PConnectorServerWorkerStoreMetricsCollector() = default;

public:
    bool    success                 = true;
    int64_t total_block_count       = 0;
    int64_t store_wait_done_time_us = 0;
    int64_t start_time_us           = 0;
};

class P2PConnectorServerWorkerStatusMetricsCollector final {
public:
    int64_t wait_store_event_count = 0;
    int64_t task_count             = 0;
    int64_t computed_request_count = 0;
};

class P2PConnectorServerWorkerWriteMetricsCollector final {
public:
    P2PConnectorServerWorkerWriteMetricsCollector()  = default;
    ~P2PConnectorServerWorkerWriteMetricsCollector() = default;

public:
    bool    success                  = true;
    int64_t first_layer_wait_time_us = 0;
    int64_t last_layer_wait_time_us  = 0;
    int64_t total_cost_time_us       = 0;
};

class P2PConnectorStreamStoreMetricsCollector1 final {
public:
    int64_t stream_count = 0;
};

class P2PConnectorStreamStoreMetricsCollector2 final {
public:
    bool    timeout             = false;
    int64_t stream_wait_time_us = 0;
};

class P2PConnectorMetrics: public kmonitor::MetricsGroup {
public:
    P2PConnectorMetrics()  = default;
    ~P2PConnectorMetrics() = default;

public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, P2PConnectorClientSchedulerMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, P2PConnectorClientWorkerMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, P2PConnectorClientSchedulerStatusMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, P2PConnectorStreamStoreMetricsCollector1* collector);
    void report(const kmonitor::MetricsTags* tags, P2PConnectorStreamStoreMetricsCollector2* collector);
    void report(const kmonitor::MetricsTags* tags, P2PConnectorServerSchedulerMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, P2PConnectorServerWorkerWriteMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, P2PConnectorServerWorkerStatusMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, P2PConnectorServerWorkerStoreMetricsCollector* collector);

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
    kmonitor::MutableMetric* stream_store_timeout_count_metric       = nullptr;
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