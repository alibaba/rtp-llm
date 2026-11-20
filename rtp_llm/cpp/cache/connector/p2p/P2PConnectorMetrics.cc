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
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_first_token_to_decode_receive_us_metric,
                                  "rtp_llm_p2p_connector_prefill_first_token_to_decode_receive_us");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_first_token_to_decode_kv_complete_us_metric,
                                  "rtp_llm_p2p_connector_prefill_first_token_to_decode_kv_complete_us");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_handoff_ready_to_second_token_output_us_metric,
                                  "rtp_llm_p2p_connector_decode_handoff_ready_to_second_token_output_us");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_handoff_ready_to_running_us_metric,
                                  "rtp_llm_p2p_connector_decode_handoff_ready_to_running_us");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_first_to_second_token_output_us_metric,
                                  "rtp_llm_p2p_connector_decode_first_to_second_token_output_us");
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

void P2PConnectorMetrics::report(const kmonitor::MetricsTags* tags, P2PConnectorMetricsCollector* collector) {
    if (collector->decode_handoff_ready_to_running_us != P2PConnectorMetricsCollector::kMissingHandoffDelta) {
        RTP_REPORT_MUTABLE_METRIC(decode_handoff_ready_to_running_us_metric,
                                  collector->decode_handoff_ready_to_running_us);
    }
    if (collector->decode_handoff_ready_to_second_token_output_us != P2PConnectorMetricsCollector::kMissingHandoffDelta) {
        RTP_REPORT_MUTABLE_METRIC(decode_handoff_ready_to_second_token_output_us_metric,
                                  collector->decode_handoff_ready_to_second_token_output_us);
    }
    if (collector->decode_first_to_second_token_output_us != P2PConnectorMetricsCollector::kMissingHandoffDelta) {
        RTP_REPORT_MUTABLE_METRIC(decode_first_to_second_token_output_us_metric,
                                  collector->decode_first_to_second_token_output_us);
    }
    if (collector->decode_scheduler_retained_kv_bytes >= 0) {
        RTP_REPORT_MUTABLE_METRIC(decode_scheduler_retained_kv_bytes_metric, collector->decode_scheduler_retained_kv_bytes);
        RTP_REPORT_MUTABLE_METRIC(decode_scheduler_retained_kv_percent_metric, collector->decode_scheduler_retained_kv_percent);
        RTP_REPORT_MUTABLE_METRIC(decode_scheduler_retained_context_count_metric,
                              collector->decode_scheduler_retained_context_count);
        RTP_REPORT_MUTABLE_METRIC(decode_scheduler_oldest_kv_hold_ms_metric, collector->decode_scheduler_oldest_kv_hold_ms);
    }
    // decode schedule：请求级采样，total 由上报方保证已写入，作为本场景的触发字段。
    const auto decode_schedule_total_us = collector->decode_schedule_total_cost_time_us.load();
    if (decode_schedule_total_us >= 0) {
        if (const auto delta = collector->prefill_first_token_to_decode_receive_us.load();
            delta != P2PConnectorMetricsCollector::kMissingHandoffDelta) {
            RTP_REPORT_MUTABLE_METRIC(prefill_first_token_to_decode_receive_us_metric, delta);
        }
        if (const auto delta = collector->prefill_first_token_to_decode_kv_complete_us.load();
            delta != P2PConnectorMetricsCollector::kMissingHandoffDelta) {
            RTP_REPORT_MUTABLE_METRIC(prefill_first_token_to_decode_kv_complete_us_metric, delta);
        }
        if (collector->decode_schedule_server_call_cost_time_us >= 0) {
            RTP_REPORT_MUTABLE_METRIC(decode_schedule_server_call_cost_time_us_metric,
                                  collector->decode_schedule_server_call_cost_time_us.load());
        }
        if (collector->decode_schedule_tp_sync_cost_time_us >= 0) {
            RTP_REPORT_MUTABLE_METRIC(decode_schedule_tp_sync_cost_time_us_metric,
                                  collector->decode_schedule_tp_sync_cost_time_us.load());
        }
        if (collector->decode_schedule_plan_cost_time_us.load(std::memory_order_relaxed) >= 0) {
            RTP_REPORT_MUTABLE_METRIC(decode_schedule_plan_cost_time_us_metric,
                                  collector->decode_schedule_plan_cost_time_us.load(std::memory_order_relaxed));
        }
        if (collector->decode_schedule_kickoff_queue_time_us.load(std::memory_order_relaxed) >= 0) {
            RTP_REPORT_MUTABLE_METRIC(decode_schedule_kickoff_queue_time_us_metric,
                                  collector->decode_schedule_kickoff_queue_time_us.load(std::memory_order_relaxed));
        }
        if (collector->decode_schedule_server_submit_time_us.load(std::memory_order_relaxed) >= 0) {
            RTP_REPORT_MUTABLE_METRIC(decode_schedule_server_submit_time_us_metric,
                                  collector->decode_schedule_server_submit_time_us.load(std::memory_order_relaxed));
        }
        if (collector->decode_schedule_broadcast_submit_time_us.load(std::memory_order_relaxed) >= 0) {
            RTP_REPORT_MUTABLE_METRIC(decode_schedule_broadcast_submit_time_us_metric,
                                  collector->decode_schedule_broadcast_submit_time_us.load(std::memory_order_relaxed));
        }
        if (collector->decode_schedule_lease_query_time_us.load(std::memory_order_relaxed) >= 0) {
            RTP_REPORT_MUTABLE_METRIC(decode_schedule_lease_query_time_us_metric,
                                  collector->decode_schedule_lease_query_time_us.load(std::memory_order_relaxed));
        }
        if (collector->decode_schedule_lease_hold_time_us.load(std::memory_order_relaxed) >= 0) {
            RTP_REPORT_MUTABLE_METRIC(decode_schedule_lease_hold_time_us_metric,
                                  collector->decode_schedule_lease_hold_time_us.load(std::memory_order_relaxed));
        }
        RTP_REPORT_MUTABLE_QPS(decode_schedule_qps_metric);
        if (!collector->decode_schedule_success.load()) {
            RTP_REPORT_MUTABLE_QPS(decode_schedule_failed_qps_metric);
        }
        if (decode_schedule_total_us > 0) {
            RTP_REPORT_MUTABLE_METRIC(decode_schedule_cost_time_us_metric, decode_schedule_total_us);
        }
    }

    // decode worker：prepare/recv_wait/recv_task 为分段采样，request 级 QPS 由 total 触发。
    if (collector->decode_worker_prepare_time_us >= 0) {
        RTP_REPORT_MUTABLE_METRIC(decode_worker_prepare_time_us_metric, collector->decode_worker_prepare_time_us);
    }
    if (collector->decode_worker_recv_wait_time_us >= 0) {
        RTP_REPORT_MUTABLE_METRIC(decode_worker_recv_wait_time_us_metric, collector->decode_worker_recv_wait_time_us);
    }
    if (decode_worker_recv_task_time_us_metric && collector->decode_worker_recv_task_time_us >= 0) {
        kmonitor::MetricsTags transfer_tags = tags ? *tags : kmonitor::MetricsTags();
        transfer_tags.AddTag("success", collector->decode_worker_success ? "true" : "false");
        reportKmonMetric(decode_worker_recv_task_time_us_metric, &transfer_tags, collector->decode_worker_recv_task_time_us);
    }
    if (collector->decode_worker_total_cost_time_us >= 0) {
        RTP_REPORT_MUTABLE_QPS(decode_worker_qps_metric);
        if (!collector->decode_worker_success) {
            RTP_REPORT_MUTABLE_QPS(decode_worker_failed_qps_metric);
        }
        RTP_REPORT_MUTABLE_METRIC(decode_worker_total_block_count_metric, collector->decode_worker_total_block_count);
        if (collector->decode_worker_first_layer_wait_time_us >= 0) {
            RTP_REPORT_MUTABLE_METRIC(decode_worker_first_layer_wait_time_us_metric,
                                  collector->decode_worker_first_layer_wait_time_us);
        }
        if (collector->decode_worker_total_cost_time_us > 0) {
            RTP_REPORT_MUTABLE_METRIC(decode_worker_total_cost_time_us_metric, collector->decode_worker_total_cost_time_us);
        }
    }

    // decode scheduler status：checker 每次巡检采样。
    if (collector->decode_scheduler_check_once_cost_time_us >= 0) {
        RTP_REPORT_MUTABLE_METRIC(decode_scheduler_check_once_cost_time_us_metric,
                              collector->decode_scheduler_check_once_cost_time_us);
        RTP_REPORT_MUTABLE_METRIC(decode_scheduler_inflight_context_count_metric,
                              collector->decode_scheduler_inflight_context_count);
    }

    // stream store：资源数量与单次等待采样（stream_wait_time_us 为触发字段）。
    if (collector->stream_store_stream_count >= 0) {
        RTP_REPORT_MUTABLE_METRIC(stream_store_stream_count_metric, collector->stream_store_stream_count);
    }
    if (collector->stream_store_stream_wait_time_us >= 0) {
        RTP_REPORT_MUTABLE_QPS(stream_store_qps_metric);
        if (collector->stream_store_timeout) {
            RTP_REPORT_MUTABLE_QPS(stream_store_timeout_qps_metric);
        }
        if (collector->stream_store_cancelled) {
            RTP_REPORT_MUTABLE_QPS(stream_store_cancel_qps_metric);
        }
        if (collector->stream_store_stream_wait_time_us > 0) {
            RTP_REPORT_MUTABLE_METRIC(stream_store_stream_wait_time_us_metric, collector->stream_store_stream_wait_time_us);
        }
    }

    // prefill scheduler：各分段独立上报，QPS 由 total 触发。
    if (collector->prefill_scheduler_check_plan_time_us >= 0) {
        RTP_REPORT_MUTABLE_METRIC(prefill_scheduler_check_plan_time_us_metric,
                              collector->prefill_scheduler_check_plan_time_us);
    }
    if (collector->prefill_scheduler_resource_register_time_us >= 0) {
        RTP_REPORT_MUTABLE_METRIC(prefill_scheduler_resource_register_time_us_metric,
                              collector->prefill_scheduler_resource_register_time_us);
    }
    if (collector->prefill_scheduler_plan_cost_time_us >= 0) {
        RTP_REPORT_MUTABLE_METRIC(prefill_scheduler_plan_cost_time_us_metric,
                              collector->prefill_scheduler_plan_cost_time_us);
    }
    if (collector->prefill_scheduler_broadcast_submit_time_us >= 0) {
        RTP_REPORT_MUTABLE_METRIC(prefill_scheduler_broadcast_submit_time_us_metric,
                              collector->prefill_scheduler_broadcast_submit_time_us);
    }
    if (collector->prefill_scheduler_broadcast_wait_time_us >= 0) {
        RTP_REPORT_MUTABLE_METRIC(prefill_scheduler_broadcast_wait_time_us_metric,
                              collector->prefill_scheduler_broadcast_wait_time_us);
    }
    if (collector->prefill_scheduler_process_read_time_us >= 0) {
        RTP_REPORT_MUTABLE_METRIC(prefill_scheduler_process_read_time_us_metric,
                              collector->prefill_scheduler_process_read_time_us);
    }
    if (collector->prefill_scheduler_resource_wait_time_us >= 0) {
        RTP_REPORT_MUTABLE_METRIC(prefill_scheduler_resource_wait_time_us_metric,
                              collector->prefill_scheduler_resource_wait_time_us);
    }
    if (collector->prefill_scheduler_side_channel_wait_time_us >= 0) {
        RTP_REPORT_MUTABLE_METRIC(prefill_scheduler_side_channel_wait_time_us_metric,
                              collector->prefill_scheduler_side_channel_wait_time_us);
    }
    if (collector->prefill_scheduler_side_channel_fill_time_us >= 0) {
        RTP_REPORT_MUTABLE_METRIC(prefill_scheduler_side_channel_fill_time_us_metric,
                              collector->prefill_scheduler_side_channel_fill_time_us);
    }
    if (collector->prefill_scheduler_total_cost_time_us >= 0) {
        RTP_REPORT_MUTABLE_QPS(prefill_scheduler_qps_metric);
        if (!collector->prefill_scheduler_success) {
            RTP_REPORT_MUTABLE_QPS(prefill_scheduler_failed_qps_metric);
        }
        if (collector->prefill_scheduler_total_cost_time_us > 0) {
            RTP_REPORT_MUTABLE_METRIC(prefill_scheduler_total_cost_time_us_metric,
                                  collector->prefill_scheduler_total_cost_time_us);
        }
    }

    // prefill worker store：start_time_us 由创建方写入，作为本场景的触发字段。
    if (collector->prefill_worker_store_start_time_us >= 0) {
        RTP_REPORT_MUTABLE_QPS(prefill_worker_store_qps_metric);
        if (!collector->prefill_worker_store_success) {
            RTP_REPORT_MUTABLE_QPS(prefill_worker_store_failed_qps_metric);
        }
        if (collector->prefill_worker_store_total_block_count > 0) {
            RTP_REPORT_MUTABLE_METRIC(prefill_worker_store_total_block_count_metric,
                                  collector->prefill_worker_store_total_block_count);
        }
        if (collector->prefill_worker_store_store_wait_done_time_us > 0) {
            RTP_REPORT_MUTABLE_METRIC(prefill_worker_store_store_wait_done_time_us_metric,
                                  collector->prefill_worker_store_store_wait_done_time_us);
        }
    }

    // prefill worker status：checker 周期采样，三分量同时写入。
    if (collector->prefill_worker_wait_store_event_count >= 0) {
        RTP_REPORT_MUTABLE_METRIC(prefill_worker_wait_store_event_count_metric,
                              collector->prefill_worker_wait_store_event_count);
        RTP_REPORT_MUTABLE_METRIC(prefill_worker_task_count_metric, collector->prefill_worker_task_count);
        RTP_REPORT_MUTABLE_METRIC(prefill_worker_computed_request_count_metric,
                              collector->prefill_worker_computed_request_count);
    }

    // prefill worker write：分段采样独立上报，QPS 由 total 触发。
    if (collector->prefill_worker_write_add_buffer_time_us >= 0) {
        RTP_REPORT_MUTABLE_METRIC(prefill_worker_write_add_buffer_time_us_metric,
                              collector->prefill_worker_write_add_buffer_time_us);
    }
    if (collector->prefill_worker_write_dispatch_time_us >= 0) {
        RTP_REPORT_MUTABLE_METRIC(prefill_worker_write_dispatch_time_us_metric,
                              collector->prefill_worker_write_dispatch_time_us);
    }
    if (collector->prefill_worker_write_callback_wait_time_us >= 0) {
        RTP_REPORT_MUTABLE_METRIC(prefill_worker_write_callback_wait_time_us_metric,
                              collector->prefill_worker_write_callback_wait_time_us);
    }
    if (collector->prefill_worker_write_sender_queue_time_us >= 0) {
        RTP_REPORT_MUTABLE_METRIC(prefill_worker_write_sender_queue_time_us_metric,
                              collector->prefill_worker_write_sender_queue_time_us);
    }
    if (collector->prefill_worker_write_send_submit_time_us >= 0) {
        RTP_REPORT_MUTABLE_METRIC(prefill_worker_write_send_submit_time_us_metric,
                              collector->prefill_worker_write_send_submit_time_us);
    }
    if (prefill_worker_write_send_complete_time_us_metric
        && collector->prefill_worker_write_send_complete_time_us >= 0) {
        kmonitor::MetricsTags transfer_tags = tags ? *tags : kmonitor::MetricsTags();
        transfer_tags.AddTag("success", collector->prefill_worker_write_success ? "true" : "false");
        reportKmonMetric(prefill_worker_write_send_complete_time_us_metric,
                         &transfer_tags, collector->prefill_worker_write_send_complete_time_us);
    }
    if (collector->prefill_worker_write_total_cost_time_us >= 0) {
        RTP_REPORT_MUTABLE_QPS(prefill_worker_write_qps_metric);
        if (!collector->prefill_worker_write_success) {
            RTP_REPORT_MUTABLE_QPS(prefill_worker_write_failed_qps_metric);
        }
        if (collector->prefill_worker_write_first_layer_wait_time_us >= 0) {
            RTP_REPORT_MUTABLE_METRIC(prefill_worker_write_first_layer_wait_time_us_metric,
                                  collector->prefill_worker_write_first_layer_wait_time_us);
        }
        if (collector->prefill_worker_write_last_layer_wait_time_us >= 0) {
            RTP_REPORT_MUTABLE_METRIC(prefill_worker_write_last_layer_wait_time_us_metric,
                                  collector->prefill_worker_write_last_layer_wait_time_us);
        }
        if (collector->prefill_worker_write_total_cost_time_us > 0) {
            RTP_REPORT_MUTABLE_METRIC(prefill_worker_write_total_cost_time_us_metric,
                                  collector->prefill_worker_write_total_cost_time_us);
        }
    }
}

}  // namespace rtp_llm
