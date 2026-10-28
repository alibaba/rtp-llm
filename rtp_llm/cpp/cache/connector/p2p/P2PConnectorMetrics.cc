#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"

namespace rtp_llm {

bool P2PConnectorMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(decode_scheduler_retained_kv_bytes_metric,
                                  "rtp_llm_p2p_connector_decode_scheduler_retained_kv_bytes");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_scheduler_retained_kv_percent_metric,
                                  "rtp_llm_p2p_connector_decode_scheduler_retained_kv_percent");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_scheduler_retained_context_count_metric,
                                  "rtp_llm_p2p_connector_decode_scheduler_retained_context_count");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_scheduler_oldest_kv_hold_ms_metric,
                                  "rtp_llm_p2p_connector_decode_scheduler_oldest_kv_hold_ms");
    REGISTER_QPS_MUTABLE_METRIC(writeback_qps_metric, "rtp_llm_p2p_writeback_qps");
    REGISTER_QPS_MUTABLE_METRIC(writeback_skipped_qps_metric, "rtp_llm_p2p_writeback_skipped_qps");
    REGISTER_QPS_MUTABLE_METRIC(writeback_failed_qps_metric, "rtp_llm_p2p_writeback_failed_qps");
    REGISTER_QPS_MUTABLE_METRIC(writeback_no_transfer_qps_metric, "rtp_llm_p2p_writeback_no_transfer_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(writeback_cost_time_us_metric, "rtp_llm_p2p_writeback_cost_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(writeback_hold_time_us_metric, "rtp_llm_p2p_writeback_hold_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(writeback_planned_bytes_metric, "rtp_llm_p2p_writeback_planned_bytes");
    REGISTER_GAUGE_MUTABLE_METRIC(writeback_committed_token_count_metric,
                                  "rtp_llm_p2p_writeback_committed_token_count");
    REGISTER_GAUGE_MUTABLE_METRIC(writeback_drain_time_us_metric, "rtp_llm_p2p_writeback_drain_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(writeback_inflight_context_count_metric,
                                  "rtp_llm_p2p_writeback_inflight_context_count");
    REGISTER_GAUGE_MUTABLE_METRIC(writeback_oldest_inflight_ms_metric, "rtp_llm_p2p_writeback_oldest_inflight_ms");
    REGISTER_GAUGE_MUTABLE_METRIC(writeback_retained_context_count_metric,
                                  "rtp_llm_p2p_writeback_retained_context_count");
    REGISTER_GAUGE_MUTABLE_METRIC(writeback_retained_kv_bytes_metric, "rtp_llm_p2p_writeback_retained_kv_bytes");
    REGISTER_GAUGE_MUTABLE_METRIC(writeback_retained_kv_percent_metric, "rtp_llm_p2p_writeback_retained_kv_percent");
    REGISTER_GAUGE_MUTABLE_METRIC(writeback_oldest_kv_hold_ms_metric, "rtp_llm_p2p_writeback_oldest_kv_hold_ms");
    REGISTER_GAUGE_MUTABLE_METRIC(writeback_stage_cost_time_us_metric, "rtp_llm_p2p_writeback_stage_cost_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(writeback_transferred_bytes_metric, "rtp_llm_p2p_writeback_transferred_bytes");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_schedule_server_call_cost_time_us_metric,
                                  "rtp_llm_p2p_connector_decode_schedule_server_call_cost_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_schedule_tp_sync_cost_time_us_metric,
                                  "rtp_llm_p2p_connector_decode_schedule_tp_sync_cost_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_schedule_plan_cost_time_us_metric,
                                  "rtp_llm_p2p_connector_decode_schedule_plan_cost_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_schedule_kickoff_queue_time_us_metric,
                                  "rtp_llm_p2p_connector_decode_schedule_kickoff_queue_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_schedule_server_submit_time_us_metric,
                                  "rtp_llm_p2p_connector_decode_schedule_server_submit_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_schedule_broadcast_submit_time_us_metric,
                                  "rtp_llm_p2p_connector_decode_schedule_broadcast_submit_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_schedule_lease_query_time_us_metric,
                                  "rtp_llm_p2p_connector_decode_schedule_lease_query_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_schedule_lease_hold_time_us_metric,
                                  "rtp_llm_p2p_connector_decode_schedule_lease_hold_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_worker_prepare_time_us_metric,
                                  "rtp_llm_p2p_connector_decode_worker_prepare_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_worker_recv_wait_time_us_metric,
                                  "rtp_llm_p2p_connector_decode_worker_recv_wait_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_worker_recv_task_time_us_metric,
                                  "rtp_llm_p2p_connector_decode_worker_recv_task_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_scheduler_plan_cost_time_us_metric,
                                  "rtp_llm_p2p_connector_prefill_scheduler_plan_cost_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_scheduler_broadcast_submit_time_us_metric,
                                  "rtp_llm_p2p_connector_prefill_scheduler_broadcast_submit_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_scheduler_broadcast_wait_time_us_metric,
                                  "rtp_llm_p2p_connector_prefill_scheduler_broadcast_wait_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_scheduler_process_read_time_us_metric,
                                  "rtp_llm_p2p_connector_prefill_scheduler_process_read_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_scheduler_resource_wait_time_us_metric,
                                  "rtp_llm_p2p_connector_prefill_scheduler_resource_wait_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_scheduler_side_channel_wait_time_us_metric,
                                  "rtp_llm_p2p_connector_prefill_scheduler_side_channel_wait_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_scheduler_side_channel_fill_time_us_metric,
                                  "rtp_llm_p2p_connector_prefill_scheduler_side_channel_fill_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_worker_write_add_buffer_time_us_metric,
                                  "rtp_llm_p2p_connector_prefill_worker_write_add_buffer_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_worker_write_dispatch_time_us_metric,
                                  "rtp_llm_p2p_connector_prefill_worker_write_dispatch_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_worker_write_callback_wait_time_us_metric,
                                  "rtp_llm_p2p_connector_prefill_worker_write_callback_wait_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_worker_write_sender_queue_time_us_metric,
                                  "rtp_llm_p2p_connector_prefill_worker_write_sender_queue_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_worker_write_send_submit_time_us_metric,
                                  "rtp_llm_p2p_connector_prefill_worker_write_send_submit_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_worker_write_send_complete_time_us_metric,
                                  "rtp_llm_p2p_connector_prefill_worker_write_send_complete_time_us");

    REGISTER_GAUGE_MUTABLE_METRIC(prefill_scheduler_resource_register_time_us_metric,
                                  "rtp_llm_p2p_connector_prefill_scheduler_resource_register_time_us");

    REGISTER_GAUGE_MUTABLE_METRIC(prefill_scheduler_check_plan_time_us_metric,
                                  "rtp_llm_p2p_connector_prefill_scheduler_check_plan_time_us");

    // decode schedule metrics
    REGISTER_QPS_MUTABLE_METRIC(decode_schedule_qps_metric, "rtp_llm_p2p_connector_decode_schedule_qps");
    REGISTER_QPS_MUTABLE_METRIC(decode_schedule_failed_qps_metric, "rtp_llm_p2p_connector_decode_schedule_failed_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_schedule_cost_time_us_metric,
                                  "rtp_llm_p2p_connector_decode_schedule_cost_time_us");

    // decode worker metrics
    REGISTER_QPS_MUTABLE_METRIC(decode_worker_qps_metric, "rtp_llm_p2p_connector_decode_worker_qps");
    REGISTER_QPS_MUTABLE_METRIC(decode_worker_failed_qps_metric, "rtp_llm_p2p_connector_decode_worker_failed_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_worker_total_block_count_metric,
                                  "rtp_llm_p2p_connector_decode_worker_total_block_count");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_worker_first_layer_wait_time_us_metric,
                                  "rtp_llm_p2p_connector_decode_worker_first_layer_wait_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_worker_total_cost_time_us_metric,
                                  "rtp_llm_p2p_connector_decode_worker_total_cost_time_us");

    // decode scheduler status metrics
    REGISTER_GAUGE_MUTABLE_METRIC(decode_scheduler_check_once_cost_time_us_metric,
                                  "rtp_llm_p2p_connector_decode_scheduler_check_once_cost_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_scheduler_inflight_context_count_metric,
                                  "rtp_llm_p2p_connector_decode_scheduler_inflight_context_count");

    // stream store metrics
    REGISTER_GAUGE_MUTABLE_METRIC(stream_store_stream_count_metric, "rtp_llm_p2p_connector_stream_store_stream_count");
    REGISTER_QPS_MUTABLE_METRIC(stream_store_qps_metric, "rtp_llm_p2p_connector_stream_store_qps");
    REGISTER_QPS_MUTABLE_METRIC(stream_store_timeout_qps_metric, "rtp_llm_p2p_connector_stream_store_timeout_qps");
    REGISTER_QPS_MUTABLE_METRIC(stream_store_cancel_qps_metric, "rtp_llm_p2p_connector_stream_store_cancel_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(stream_store_stream_wait_time_us_metric,
                                  "rtp_llm_p2p_connector_stream_store_stream_wait_time_us");

    // prefill scheduler metrics
    REGISTER_QPS_MUTABLE_METRIC(prefill_scheduler_qps_metric, "rtp_llm_p2p_connector_prefill_scheduler_qps");
    REGISTER_QPS_MUTABLE_METRIC(prefill_scheduler_failed_qps_metric,
                                "rtp_llm_p2p_connector_prefill_scheduler_failed_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_scheduler_total_cost_time_us_metric,
                                  "rtp_llm_p2p_connector_prefill_scheduler_total_cost_time_us");

    // prefill worker store metrics
    REGISTER_QPS_MUTABLE_METRIC(prefill_worker_store_qps_metric, "rtp_llm_p2p_connector_prefill_worker_store_qps");
    REGISTER_QPS_MUTABLE_METRIC(prefill_worker_store_failed_qps_metric,
                                "rtp_llm_p2p_connector_prefill_worker_store_failed_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_worker_store_total_block_count_metric,
                                  "rtp_llm_p2p_connector_prefill_worker_store_total_block_count");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_worker_store_store_wait_done_time_us_metric,
                                  "rtp_llm_p2p_connector_prefill_worker_store_store_wait_done_time_us");

    // prefill worker write metrics
    REGISTER_QPS_MUTABLE_METRIC(prefill_worker_write_qps_metric, "rtp_llm_p2p_connector_prefill_worker_write_qps");
    REGISTER_QPS_MUTABLE_METRIC(prefill_worker_write_failed_qps_metric,
                                "rtp_llm_p2p_connector_prefill_worker_write_failed_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_worker_write_first_layer_wait_time_us_metric,
                                  "rtp_llm_p2p_connector_prefill_worker_write_first_layer_wait_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_worker_write_last_layer_wait_time_us_metric,
                                  "rtp_llm_p2p_connector_prefill_worker_write_last_layer_wait_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_worker_write_total_cost_time_us_metric,
                                  "rtp_llm_p2p_connector_prefill_worker_write_total_cost_time_us");

    // prefill worker status metrics
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_worker_wait_store_event_count_metric,
                                  "rtp_llm_p2p_connector_prefill_worker_wait_store_event_count");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_worker_task_count_metric, "rtp_llm_p2p_connector_prefill_worker_task_count");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_worker_computed_request_count_metric,
                                  "rtp_llm_p2p_connector_prefill_worker_computed_request_count");

    return true;
}

void P2PConnectorMetrics::report(const kmonitor::MetricsTags* tags, WriteSchedulerMetricsCollector* collector) {
    kmonitor::MetricsTags write_tags = tags ? *tags : kmonitor::MetricsTags{};
    write_tags.AddTag("side", collector->prefill ? "prefill" : "decode");
    write_tags.AddTag("submitted", collector->submitted ? "true" : "false");
    const auto& message = collector->error.ToString();
    const auto  reason  = message == "prefix_evicted_or_demoted" ? message : ErrorCodeToString(collector->error.code());
    write_tags.AddTag("reason", collector->skip_reason ? collector->skip_reason : reason);
    tags = &write_tags;
    if (collector->skip_reason) {
        REPORT_MUTABLE_QPS(writeback_skipped_qps_metric);
        return;
    }
    REPORT_MUTABLE_QPS(writeback_qps_metric);
    if (collector->error.hasError()) {
        REPORT_MUTABLE_QPS(writeback_failed_qps_metric);
    }
    if (collector->no_transfer && collector->error.ok()) {
        REPORT_MUTABLE_QPS(writeback_no_transfer_qps_metric);
    }
    REPORT_MUTABLE_METRIC(writeback_cost_time_us_metric, collector->total_cost_time_us);
    if (collector->prefill) {
        REPORT_MUTABLE_METRIC(writeback_hold_time_us_metric, collector->hold_time_us);
    }
    REPORT_MUTABLE_METRIC(writeback_planned_bytes_metric, collector->planned_bytes);
    if (collector->committed_token_count >= 0) {
        REPORT_MUTABLE_METRIC(writeback_committed_token_count_metric, collector->committed_token_count);
    }
    if (collector->drain_time_us >= 0) {
        REPORT_MUTABLE_METRIC(writeback_drain_time_us_metric, collector->drain_time_us);
    }
}

void P2PConnectorMetrics::report(const kmonitor::MetricsTags* tags, WritebackStatusMetricsCollector* collector) {
    kmonitor::MetricsTags status_tags = tags ? *tags : kmonitor::MetricsTags{};
    status_tags.AddTag("side", collector->prefill ? "prefill" : "decode");
    tags = &status_tags;
    REPORT_MUTABLE_METRIC(writeback_inflight_context_count_metric, collector->inflight_context_count);
    REPORT_MUTABLE_METRIC(writeback_oldest_inflight_ms_metric, collector->oldest_inflight_ms);
    REPORT_MUTABLE_METRIC(writeback_retained_context_count_metric, collector->retained_context_count);
    REPORT_MUTABLE_METRIC(writeback_retained_kv_bytes_metric, collector->retained_kv_bytes);
    REPORT_MUTABLE_METRIC(writeback_retained_kv_percent_metric, collector->retained_kv_percent);
    REPORT_MUTABLE_METRIC(writeback_oldest_kv_hold_ms_metric, collector->oldest_kv_hold_ms);
}

void P2PConnectorMetrics::report(const kmonitor::MetricsTags* tags, WritebackStageMetricsCollector* collector) {
    if (!collector->stage) {
        return;
    }
    kmonitor::MetricsTags stage_tags = tags ? *tags : kmonitor::MetricsTags{};
    stage_tags.AddTag("side", collector->prefill ? "prefill" : "decode");
    stage_tags.AddTag("stage", collector->stage);
    stage_tags.AddTag("success", collector->error.ok() ? "true" : "false");
    tags = &stage_tags;
    REPORT_MUTABLE_METRIC(writeback_stage_cost_time_us_metric, collector->cost_time_us);
    if (collector->transferred_bytes >= 0 && collector->error.ok()) {
        REPORT_MUTABLE_METRIC(writeback_transferred_bytes_metric, collector->transferred_bytes);
    }
}

void P2PConnectorMetrics::report(const kmonitor::MetricsTags* tags, P2PConnectorMetricsCollector* collector) {
    if (collector->decode_scheduler_retained_kv_bytes >= 0) {
        REPORT_MUTABLE_METRIC(decode_scheduler_retained_kv_bytes_metric, collector->decode_scheduler_retained_kv_bytes);
        REPORT_MUTABLE_METRIC(decode_scheduler_retained_kv_percent_metric, collector->decode_scheduler_retained_kv_percent);
        REPORT_MUTABLE_METRIC(decode_scheduler_retained_context_count_metric,
                              collector->decode_scheduler_retained_context_count);
        REPORT_MUTABLE_METRIC(decode_scheduler_oldest_kv_hold_ms_metric, collector->decode_scheduler_oldest_kv_hold_ms);
    }
    // decode schedule：请求级采样，total 由上报方保证已写入，作为本场景的触发字段。
    const auto decode_schedule_total_us = collector->decode_schedule_total_cost_time_us.load();
    if (decode_schedule_total_us >= 0) {
        if (collector->decode_schedule_server_call_cost_time_us >= 0) {
            REPORT_MUTABLE_METRIC(decode_schedule_server_call_cost_time_us_metric,
                                  collector->decode_schedule_server_call_cost_time_us.load());
        }
        if (collector->decode_schedule_tp_sync_cost_time_us >= 0) {
            REPORT_MUTABLE_METRIC(decode_schedule_tp_sync_cost_time_us_metric,
                                  collector->decode_schedule_tp_sync_cost_time_us.load());
        }
        if (collector->decode_schedule_plan_cost_time_us.load(std::memory_order_relaxed) >= 0) {
            REPORT_MUTABLE_METRIC(decode_schedule_plan_cost_time_us_metric,
                                  collector->decode_schedule_plan_cost_time_us.load(std::memory_order_relaxed));
        }
        if (collector->decode_schedule_kickoff_queue_time_us.load(std::memory_order_relaxed) >= 0) {
            REPORT_MUTABLE_METRIC(decode_schedule_kickoff_queue_time_us_metric,
                                  collector->decode_schedule_kickoff_queue_time_us.load(std::memory_order_relaxed));
        }
        if (collector->decode_schedule_server_submit_time_us.load(std::memory_order_relaxed) >= 0) {
            REPORT_MUTABLE_METRIC(decode_schedule_server_submit_time_us_metric,
                                  collector->decode_schedule_server_submit_time_us.load(std::memory_order_relaxed));
        }
        if (collector->decode_schedule_broadcast_submit_time_us.load(std::memory_order_relaxed) >= 0) {
            REPORT_MUTABLE_METRIC(decode_schedule_broadcast_submit_time_us_metric,
                                  collector->decode_schedule_broadcast_submit_time_us.load(std::memory_order_relaxed));
        }
        if (collector->decode_schedule_lease_query_time_us.load(std::memory_order_relaxed) >= 0) {
            REPORT_MUTABLE_METRIC(decode_schedule_lease_query_time_us_metric,
                                  collector->decode_schedule_lease_query_time_us.load(std::memory_order_relaxed));
        }
        if (collector->decode_schedule_lease_hold_time_us.load(std::memory_order_relaxed) >= 0) {
            REPORT_MUTABLE_METRIC(decode_schedule_lease_hold_time_us_metric,
                                  collector->decode_schedule_lease_hold_time_us.load(std::memory_order_relaxed));
        }
        REPORT_MUTABLE_QPS(decode_schedule_qps_metric);
        if (!collector->decode_schedule_success.load()) {
            REPORT_MUTABLE_QPS(decode_schedule_failed_qps_metric);
        }
        if (decode_schedule_total_us > 0) {
            REPORT_MUTABLE_METRIC(decode_schedule_cost_time_us_metric, decode_schedule_total_us);
        }
    }

    // decode worker：prepare/recv_wait/recv_task 为分段采样，request 级 QPS 由 total 触发。
    if (collector->decode_worker_prepare_time_us >= 0) {
        REPORT_MUTABLE_METRIC(decode_worker_prepare_time_us_metric, collector->decode_worker_prepare_time_us);
    }
    if (collector->decode_worker_recv_wait_time_us >= 0) {
        REPORT_MUTABLE_METRIC(decode_worker_recv_wait_time_us_metric, collector->decode_worker_recv_wait_time_us);
    }
    if (collector->decode_worker_recv_task_time_us >= 0) {
        kmonitor::MetricsTags transfer_tags = tags ? *tags : kmonitor::MetricsTags();
        transfer_tags.AddTag("success", collector->decode_worker_success ? "true" : "false");
        decode_worker_recv_task_time_us_metric->Report(&transfer_tags, collector->decode_worker_recv_task_time_us);
    }
    if (collector->decode_worker_total_cost_time_us >= 0) {
        REPORT_MUTABLE_QPS(decode_worker_qps_metric);
        if (!collector->decode_worker_success) {
            REPORT_MUTABLE_QPS(decode_worker_failed_qps_metric);
        }
        REPORT_MUTABLE_METRIC(decode_worker_total_block_count_metric, collector->decode_worker_total_block_count);
        if (collector->decode_worker_first_layer_wait_time_us >= 0) {
            REPORT_MUTABLE_METRIC(decode_worker_first_layer_wait_time_us_metric,
                                  collector->decode_worker_first_layer_wait_time_us);
        }
        if (collector->decode_worker_total_cost_time_us > 0) {
            REPORT_MUTABLE_METRIC(decode_worker_total_cost_time_us_metric, collector->decode_worker_total_cost_time_us);
        }
    }

    // decode scheduler status：checker 每次巡检采样。
    if (collector->decode_scheduler_check_once_cost_time_us >= 0) {
        REPORT_MUTABLE_METRIC(decode_scheduler_check_once_cost_time_us_metric,
                              collector->decode_scheduler_check_once_cost_time_us);
        REPORT_MUTABLE_METRIC(decode_scheduler_inflight_context_count_metric,
                              collector->decode_scheduler_inflight_context_count);
    }

    // stream store：资源数量与单次等待采样（stream_wait_time_us 为触发字段）。
    if (collector->stream_store_stream_count >= 0) {
        REPORT_MUTABLE_METRIC(stream_store_stream_count_metric, collector->stream_store_stream_count);
    }
    if (collector->stream_store_stream_wait_time_us >= 0) {
        REPORT_MUTABLE_QPS(stream_store_qps_metric);
        if (collector->stream_store_timeout) {
            REPORT_MUTABLE_QPS(stream_store_timeout_qps_metric);
        }
        if (collector->stream_store_cancelled) {
            REPORT_MUTABLE_QPS(stream_store_cancel_qps_metric);
        }
        if (collector->stream_store_stream_wait_time_us > 0) {
            REPORT_MUTABLE_METRIC(stream_store_stream_wait_time_us_metric, collector->stream_store_stream_wait_time_us);
        }
    }

    // prefill scheduler：各分段独立上报，QPS 由 total 触发。
    if (collector->prefill_scheduler_check_plan_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_scheduler_check_plan_time_us_metric,
                              collector->prefill_scheduler_check_plan_time_us);
    }
    if (collector->prefill_scheduler_resource_register_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_scheduler_resource_register_time_us_metric,
                              collector->prefill_scheduler_resource_register_time_us);
    }
    if (collector->prefill_scheduler_plan_cost_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_scheduler_plan_cost_time_us_metric,
                              collector->prefill_scheduler_plan_cost_time_us);
    }
    if (collector->prefill_scheduler_broadcast_submit_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_scheduler_broadcast_submit_time_us_metric,
                              collector->prefill_scheduler_broadcast_submit_time_us);
    }
    if (collector->prefill_scheduler_broadcast_wait_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_scheduler_broadcast_wait_time_us_metric,
                              collector->prefill_scheduler_broadcast_wait_time_us);
    }
    if (collector->prefill_scheduler_process_read_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_scheduler_process_read_time_us_metric,
                              collector->prefill_scheduler_process_read_time_us);
    }
    if (collector->prefill_scheduler_resource_wait_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_scheduler_resource_wait_time_us_metric,
                              collector->prefill_scheduler_resource_wait_time_us);
    }
    if (collector->prefill_scheduler_side_channel_wait_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_scheduler_side_channel_wait_time_us_metric,
                              collector->prefill_scheduler_side_channel_wait_time_us);
    }
    if (collector->prefill_scheduler_side_channel_fill_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_scheduler_side_channel_fill_time_us_metric,
                              collector->prefill_scheduler_side_channel_fill_time_us);
    }
    if (collector->prefill_scheduler_total_cost_time_us >= 0) {
        REPORT_MUTABLE_QPS(prefill_scheduler_qps_metric);
        if (!collector->prefill_scheduler_success) {
            REPORT_MUTABLE_QPS(prefill_scheduler_failed_qps_metric);
        }
        if (collector->prefill_scheduler_total_cost_time_us > 0) {
            REPORT_MUTABLE_METRIC(prefill_scheduler_total_cost_time_us_metric,
                                  collector->prefill_scheduler_total_cost_time_us);
        }
    }

    // prefill worker store：start_time_us 由创建方写入，作为本场景的触发字段。
    if (collector->prefill_worker_store_start_time_us >= 0) {
        REPORT_MUTABLE_QPS(prefill_worker_store_qps_metric);
        if (!collector->prefill_worker_store_success) {
            REPORT_MUTABLE_QPS(prefill_worker_store_failed_qps_metric);
        }
        if (collector->prefill_worker_store_total_block_count > 0) {
            REPORT_MUTABLE_METRIC(prefill_worker_store_total_block_count_metric,
                                  collector->prefill_worker_store_total_block_count);
        }
        if (collector->prefill_worker_store_store_wait_done_time_us > 0) {
            REPORT_MUTABLE_METRIC(prefill_worker_store_store_wait_done_time_us_metric,
                                  collector->prefill_worker_store_store_wait_done_time_us);
        }
    }

    // prefill worker status：checker 周期采样，三分量同时写入。
    if (collector->prefill_worker_wait_store_event_count >= 0) {
        REPORT_MUTABLE_METRIC(prefill_worker_wait_store_event_count_metric,
                              collector->prefill_worker_wait_store_event_count);
        REPORT_MUTABLE_METRIC(prefill_worker_task_count_metric, collector->prefill_worker_task_count);
        REPORT_MUTABLE_METRIC(prefill_worker_computed_request_count_metric,
                              collector->prefill_worker_computed_request_count);
    }

    // prefill worker write：分段采样独立上报，QPS 由 total 触发。
    if (collector->prefill_worker_write_add_buffer_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_worker_write_add_buffer_time_us_metric,
                              collector->prefill_worker_write_add_buffer_time_us);
    }
    if (collector->prefill_worker_write_dispatch_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_worker_write_dispatch_time_us_metric,
                              collector->prefill_worker_write_dispatch_time_us);
    }
    if (collector->prefill_worker_write_callback_wait_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_worker_write_callback_wait_time_us_metric,
                              collector->prefill_worker_write_callback_wait_time_us);
    }
    if (collector->prefill_worker_write_sender_queue_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_worker_write_sender_queue_time_us_metric,
                              collector->prefill_worker_write_sender_queue_time_us);
    }
    if (collector->prefill_worker_write_send_submit_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_worker_write_send_submit_time_us_metric,
                              collector->prefill_worker_write_send_submit_time_us);
    }
    if (collector->prefill_worker_write_send_complete_time_us >= 0) {
        kmonitor::MetricsTags transfer_tags = tags ? *tags : kmonitor::MetricsTags();
        transfer_tags.AddTag("success", collector->prefill_worker_write_success ? "true" : "false");
        prefill_worker_write_send_complete_time_us_metric->Report(
            &transfer_tags, collector->prefill_worker_write_send_complete_time_us);
    }
    if (collector->prefill_worker_write_total_cost_time_us >= 0) {
        REPORT_MUTABLE_QPS(prefill_worker_write_qps_metric);
        if (!collector->prefill_worker_write_success) {
            REPORT_MUTABLE_QPS(prefill_worker_write_failed_qps_metric);
        }
        if (collector->prefill_worker_write_first_layer_wait_time_us >= 0) {
            REPORT_MUTABLE_METRIC(prefill_worker_write_first_layer_wait_time_us_metric,
                                  collector->prefill_worker_write_first_layer_wait_time_us);
        }
        if (collector->prefill_worker_write_last_layer_wait_time_us >= 0) {
            REPORT_MUTABLE_METRIC(prefill_worker_write_last_layer_wait_time_us_metric,
                                  collector->prefill_worker_write_last_layer_wait_time_us);
        }
        if (collector->prefill_worker_write_total_cost_time_us > 0) {
            REPORT_MUTABLE_METRIC(prefill_worker_write_total_cost_time_us_metric,
                                  collector->prefill_worker_write_total_cost_time_us);
        }
    }
}

}  // namespace rtp_llm
