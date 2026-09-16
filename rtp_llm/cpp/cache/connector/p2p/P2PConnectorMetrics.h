#pragma once

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <atomic>

namespace rtp_llm {

class P2PConnectorMetrics;

class DecodeSchedulerMetricsCollector final {
public:
    DecodeSchedulerMetricsCollector(const std::shared_ptr<kmonitor::MetricsReporter>& metrics_reporter):
        start_time_us(currentTimeUs()), metrics_reporter_(metrics_reporter) {}
    ~DecodeSchedulerMetricsCollector() {
        if (total_cost_time_us < 0) {
            total_cost_time_us = currentTimeUs() - start_time_us;
        }
        if (metrics_reporter_) {
            metrics_reporter_->report<P2PConnectorMetrics, DecodeSchedulerMetricsCollector>(nullptr, this);
        }
    }

public:
    bool    success                  = true;
    int64_t start_time_us            = 0;
    std::atomic<int64_t> server_call_cost_time_us{-1};
    std::atomic<int64_t> tp_sync_cost_time_us{-1};
    int64_t              total_cost_time_us       = -1;
    int64_t              plan_cost_time_us        = -1;
    int64_t              kickoff_queue_time_us    = -1;
    int64_t              server_submit_time_us    = -1;
    int64_t              broadcast_submit_time_us = -1;
    int64_t              lease_query_time_us      = -1;
    int64_t              lease_hold_time_us       = -1;

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
    int64_t first_layer_wait_time_us = -1;
    int64_t total_cost_time_us       = -1;
    int64_t prepare_time_us          = -1;
    int64_t recv_wait_time_us        = -1;
    int64_t recv_task_time_us        = -1;
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
    int64_t total_cost_time_us       = -1;
    int64_t plan_cost_time_us        = -1;
    int64_t broadcast_submit_time_us = -1;
    int64_t broadcast_wait_time_us   = -1;
    // A separate sample from processRead, including rendezvous and side channel.
    int64_t process_read_time_us      = -1;
    int64_t resource_register_time_us = -1;
    int64_t check_plan_time_us        = -1;
    int64_t resource_wait_time_us     = -1;
    int64_t side_channel_wait_time_us = -1;
    int64_t side_channel_fill_time_us = -1;
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
    int64_t first_layer_wait_time_us = -1;
    int64_t last_layer_wait_time_us  = -1;
    int64_t total_cost_time_us       = -1;
    int64_t add_buffer_time_us       = -1;
    int64_t dispatch_time_us         = -1;
    int64_t callback_wait_time_us    = -1;
    // Per (layer, tag, route), reported independently of the request sample.
    int64_t sender_queue_time_us  = -1;
    int64_t send_submit_time_us   = -1;
    int64_t send_complete_time_us = -1;
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

/// Tracks per-layer cache write failures at the Op level (convertToGlobalLayerId / cache_key conversion).
class CacheWriteOpFailureMetricsCollector final {
public:
    CacheWriteOpFailureMetricsCollector()  = default;
    ~CacheWriteOpFailureMetricsCollector() = default;
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
    void report(const kmonitor::MetricsTags* tags, CacheWriteOpFailureMetricsCollector* collector);

private:
    kmonitor::MutableMetric* decode_schedule_server_call_cost_time_us_metric    = nullptr;
    kmonitor::MutableMetric* decode_schedule_tp_sync_cost_time_us_metric        = nullptr;
    kmonitor::MutableMetric* decode_schedule_plan_cost_time_us_metric           = nullptr;
    kmonitor::MutableMetric* decode_schedule_kickoff_queue_time_us_metric       = nullptr;
    kmonitor::MutableMetric* decode_schedule_server_submit_time_us_metric       = nullptr;
    kmonitor::MutableMetric* decode_schedule_broadcast_submit_time_us_metric    = nullptr;
    kmonitor::MutableMetric* decode_schedule_lease_query_time_us_metric         = nullptr;
    kmonitor::MutableMetric* decode_schedule_lease_hold_time_us_metric          = nullptr;
    kmonitor::MutableMetric* decode_worker_prepare_time_us_metric               = nullptr;
    kmonitor::MutableMetric* decode_worker_recv_wait_time_us_metric             = nullptr;
    kmonitor::MutableMetric* decode_worker_recv_task_time_us_metric             = nullptr;
    kmonitor::MutableMetric* prefill_scheduler_plan_cost_time_us_metric         = nullptr;
    kmonitor::MutableMetric* prefill_scheduler_broadcast_submit_time_us_metric  = nullptr;
    kmonitor::MutableMetric* prefill_scheduler_broadcast_wait_time_us_metric    = nullptr;
    kmonitor::MutableMetric* prefill_scheduler_process_read_time_us_metric      = nullptr;
    kmonitor::MutableMetric* prefill_scheduler_resource_wait_time_us_metric     = nullptr;
    kmonitor::MutableMetric* prefill_scheduler_side_channel_wait_time_us_metric = nullptr;
    kmonitor::MutableMetric* prefill_scheduler_side_channel_fill_time_us_metric = nullptr;
    kmonitor::MutableMetric* prefill_worker_write_add_buffer_time_us_metric     = nullptr;
    kmonitor::MutableMetric* prefill_worker_write_dispatch_time_us_metric       = nullptr;
    kmonitor::MutableMetric* prefill_worker_write_callback_wait_time_us_metric  = nullptr;
    kmonitor::MutableMetric* prefill_worker_write_sender_queue_time_us_metric   = nullptr;
    kmonitor::MutableMetric* prefill_worker_write_send_submit_time_us_metric    = nullptr;
    kmonitor::MutableMetric* prefill_worker_write_send_complete_time_us_metric  = nullptr;

    kmonitor::MutableMetric* prefill_scheduler_resource_register_time_us_metric = nullptr;

    kmonitor::MutableMetric* prefill_scheduler_check_plan_time_us_metric = nullptr;

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

    // cache write op-level failure metrics
    kmonitor::MutableMetric* cache_write_op_failure_qps_metric = nullptr;
};

}  // namespace rtp_llm
