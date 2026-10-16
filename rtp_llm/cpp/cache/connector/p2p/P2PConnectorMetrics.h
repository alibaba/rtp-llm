#pragma once

#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <atomic>

namespace rtp_llm {

/// @brief P2P connector 的指标采集器，与 P2PConnectorMetrics 一一对应。
///
/// 与 RtpLLMMetrics 中的各 collector 一致，本结构体是纯数据载体：一次 report 只填写
/// 一个使用场景的字段，其余字段保持哨兵默认值（时间类为 -1），report() 依据各场景的
/// 触发字段判断该场景是否需要上报指标与 QPS。
/// 字段名与指标名对齐，便于按指标反查上报点。
class P2PConnectorMetricsCollector final {
public:
    P2PConnectorMetricsCollector()  = default;
    ~P2PConnectorMetricsCollector() = default;

public:
    // ---- decode schedule：每次 asyncRead 一份采样，终态且 lease 释放后上报一次 ----
    bool                 decode_schedule_success       = true;
    int64_t              decode_schedule_start_time_us = 0;
    std::atomic<int64_t> decode_schedule_server_call_cost_time_us{-1};
    std::atomic<int64_t> decode_schedule_tp_sync_cost_time_us{-1};
    int64_t              decode_schedule_total_cost_time_us       = -1;
    int64_t              decode_schedule_plan_cost_time_us        = -1;
    int64_t              decode_schedule_kickoff_queue_time_us    = -1;
    int64_t              decode_schedule_server_submit_time_us    = -1;
    int64_t              decode_schedule_broadcast_submit_time_us = -1;
    int64_t              decode_schedule_lease_query_time_us      = -1;
    int64_t              decode_schedule_lease_hold_time_us       = -1;

    // ---- decode worker：请求级采样与单 recv task 采样 ----
    bool    decode_worker_success                  = true;
    int64_t decode_worker_total_block_count        = 0;
    int64_t decode_worker_first_layer_wait_time_us = -1;
    int64_t decode_worker_total_cost_time_us       = -1;
    int64_t decode_worker_prepare_time_us          = -1;
    int64_t decode_worker_recv_wait_time_us        = -1;
    int64_t decode_worker_recv_task_time_us        = -1;

    // ---- decode scheduler status：checker 周期采样 ----
    int64_t decode_scheduler_check_once_cost_time_us = -1;
    int64_t decode_scheduler_inflight_context_count  = -1;

    // ---- stream store：资源数量周期采样与单次等待采样 ----
    int64_t stream_store_stream_count        = -1;
    bool    stream_store_timeout             = false;
    bool    stream_store_cancelled           = false;
    int64_t stream_store_stream_wait_time_us = -1;

    // ---- prefill scheduler：sendKVCache / registerResource / processRead 采样 ----
    bool    prefill_scheduler_success                   = true;
    int64_t prefill_scheduler_total_cost_time_us        = -1;
    int64_t prefill_scheduler_plan_cost_time_us         = -1;
    int64_t prefill_scheduler_broadcast_submit_time_us  = -1;
    int64_t prefill_scheduler_broadcast_wait_time_us    = -1;
    int64_t prefill_scheduler_process_read_time_us      = -1;
    int64_t prefill_scheduler_resource_register_time_us = -1;
    int64_t prefill_scheduler_check_plan_time_us        = -1;
    int64_t prefill_scheduler_resource_wait_time_us     = -1;
    int64_t prefill_scheduler_side_channel_wait_time_us = -1;
    int64_t prefill_scheduler_side_channel_fill_time_us = -1;

    // ---- prefill worker store：单个 store 等待上下文采样 ----
    bool    prefill_worker_store_success                 = true;
    int64_t prefill_worker_store_start_time_us           = -1;
    int64_t prefill_worker_store_total_block_count       = 0;
    int64_t prefill_worker_store_store_wait_done_time_us = 0;

    // ---- prefill worker status：checker 周期采样 ----
    int64_t prefill_worker_wait_store_event_count = -1;
    int64_t prefill_worker_task_count             = -1;
    int64_t prefill_worker_computed_request_count = -1;

    // ---- prefill worker write：请求级采样与单 (layer, tag, route) 采样 ----
    bool    prefill_worker_write_success                  = true;
    int64_t prefill_worker_write_first_layer_wait_time_us = -1;
    int64_t prefill_worker_write_last_layer_wait_time_us  = -1;
    int64_t prefill_worker_write_total_cost_time_us       = -1;
    int64_t prefill_worker_write_add_buffer_time_us       = -1;
    int64_t prefill_worker_write_dispatch_time_us         = -1;
    int64_t prefill_worker_write_callback_wait_time_us    = -1;
    int64_t prefill_worker_write_sender_queue_time_us     = -1;
    int64_t prefill_worker_write_send_submit_time_us      = -1;
    int64_t prefill_worker_write_send_complete_time_us    = -1;
};

/// @brief P2P connector 指标上报，聚合 Decode/Prefill 两侧的调度和传输指标
class P2PConnectorMetrics: public kmonitor::MetricsGroup {
public:
    P2PConnectorMetrics()  = default;
    ~P2PConnectorMetrics() = default;

public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, P2PConnectorMetricsCollector* collector);

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
};

}  // namespace rtp_llm
