#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"

namespace rtp_llm {

bool P2PConnectorMetrics::init(kmonitor::MetricsGroupManager* manager) {
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

    // cache write op-level failure metrics
    REGISTER_QPS_MUTABLE_METRIC(cache_write_op_failure_qps_metric, "rtp_llm_p2p_connector_cache_write_op_failure_qps");

    return true;
}

void P2PConnectorMetrics::report(const kmonitor::MetricsTags* tags, DecodeSchedulerMetricsCollector* collector) {
    if (collector->server_call_cost_time_us >= 0) {
        REPORT_MUTABLE_METRIC(decode_schedule_server_call_cost_time_us_metric,
                              collector->server_call_cost_time_us.load());
    }
    if (collector->tp_sync_cost_time_us >= 0) {
        REPORT_MUTABLE_METRIC(decode_schedule_tp_sync_cost_time_us_metric, collector->tp_sync_cost_time_us.load());
    }
    if (collector->plan_cost_time_us >= 0) {
        REPORT_MUTABLE_METRIC(decode_schedule_plan_cost_time_us_metric, collector->plan_cost_time_us);
    }
    if (collector->kickoff_queue_time_us >= 0) {
        REPORT_MUTABLE_METRIC(decode_schedule_kickoff_queue_time_us_metric, collector->kickoff_queue_time_us);
    }
    if (collector->server_submit_time_us >= 0) {
        REPORT_MUTABLE_METRIC(decode_schedule_server_submit_time_us_metric, collector->server_submit_time_us);
    }
    if (collector->broadcast_submit_time_us >= 0) {
        REPORT_MUTABLE_METRIC(decode_schedule_broadcast_submit_time_us_metric, collector->broadcast_submit_time_us);
    }
    if (collector->lease_query_time_us >= 0) {
        REPORT_MUTABLE_METRIC(decode_schedule_lease_query_time_us_metric, collector->lease_query_time_us);
    }
    if (collector->lease_hold_time_us >= 0) {
        REPORT_MUTABLE_METRIC(decode_schedule_lease_hold_time_us_metric, collector->lease_hold_time_us);
    }
    REPORT_MUTABLE_QPS(decode_schedule_qps_metric);
    if (!collector->success) {
        REPORT_MUTABLE_QPS(decode_schedule_failed_qps_metric);
    }
    if (collector->total_cost_time_us > 0) {
        REPORT_MUTABLE_METRIC(decode_schedule_cost_time_us_metric, collector->total_cost_time_us);
    }
}

void P2PConnectorMetrics::report(const kmonitor::MetricsTags* tags, DecodeWorkerMetricsCollector* collector) {
    if (collector->prepare_time_us >= 0) {
        REPORT_MUTABLE_METRIC(decode_worker_prepare_time_us_metric, collector->prepare_time_us);
    }
    if (collector->recv_wait_time_us >= 0) {
        REPORT_MUTABLE_METRIC(decode_worker_recv_wait_time_us_metric, collector->recv_wait_time_us);
    }
    if (collector->recv_task_time_us >= 0) {
        kmonitor::MetricsTags transfer_tags = tags ? *tags : kmonitor::MetricsTags();
        transfer_tags.AddTag("success", collector->success ? "true" : "false");
        decode_worker_recv_task_time_us_metric->Report(&transfer_tags, collector->recv_task_time_us);
    }
    if (collector->total_cost_time_us < 0) {
        return;
    }
    REPORT_MUTABLE_QPS(decode_worker_qps_metric);
    if (!collector->success) {
        REPORT_MUTABLE_QPS(decode_worker_failed_qps_metric);
    }
    REPORT_MUTABLE_METRIC(decode_worker_total_block_count_metric, collector->total_block_count);
    if (collector->first_layer_wait_time_us >= 0) {
        REPORT_MUTABLE_METRIC(decode_worker_first_layer_wait_time_us_metric, collector->first_layer_wait_time_us);
    }
    if (collector->total_cost_time_us > 0) {
        REPORT_MUTABLE_METRIC(decode_worker_total_cost_time_us_metric, collector->total_cost_time_us);
    }
}

void P2PConnectorMetrics::report(const kmonitor::MetricsTags* tags, DecodeSchedulerStatusMetricsCollector* collector) {
    REPORT_MUTABLE_METRIC(decode_scheduler_check_once_cost_time_us_metric, collector->check_once_cost_time_us);
    REPORT_MUTABLE_METRIC(decode_scheduler_inflight_context_count_metric, collector->inflight_context_count);
}

void P2PConnectorMetrics::report(const kmonitor::MetricsTags* tags, StreamStoreCountMetricsCollector* collector) {
    REPORT_MUTABLE_METRIC(stream_store_stream_count_metric, collector->stream_count);
}

void P2PConnectorMetrics::report(const kmonitor::MetricsTags* tags, StreamStoreWaitMetricsCollector* collector) {
    REPORT_MUTABLE_QPS(stream_store_qps_metric);
    if (collector->timeout) {
        REPORT_MUTABLE_QPS(stream_store_timeout_qps_metric);
    }
    if (collector->cancelled) {
        REPORT_MUTABLE_QPS(stream_store_cancel_qps_metric);
    }
    if (collector->stream_wait_time_us > 0) {
        REPORT_MUTABLE_METRIC(stream_store_stream_wait_time_us_metric, collector->stream_wait_time_us);
    }
}
//
void P2PConnectorMetrics::report(const kmonitor::MetricsTags* tags, PrefillSchedulerMetricsCollector* collector) {
    if (collector->check_plan_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_scheduler_check_plan_time_us_metric, collector->check_plan_time_us);
    }
    if (collector->resource_register_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_scheduler_resource_register_time_us_metric, collector->resource_register_time_us);
    }
    if (collector->plan_cost_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_scheduler_plan_cost_time_us_metric, collector->plan_cost_time_us);
    }
    if (collector->broadcast_submit_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_scheduler_broadcast_submit_time_us_metric, collector->broadcast_submit_time_us);
    }
    if (collector->broadcast_wait_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_scheduler_broadcast_wait_time_us_metric, collector->broadcast_wait_time_us);
    }
    if (collector->process_read_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_scheduler_process_read_time_us_metric, collector->process_read_time_us);
    }
    if (collector->resource_wait_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_scheduler_resource_wait_time_us_metric, collector->resource_wait_time_us);
    }
    if (collector->side_channel_wait_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_scheduler_side_channel_wait_time_us_metric, collector->side_channel_wait_time_us);
    }
    if (collector->side_channel_fill_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_scheduler_side_channel_fill_time_us_metric, collector->side_channel_fill_time_us);
    }
    if (collector->total_cost_time_us < 0) {
        return;
    }
    REPORT_MUTABLE_QPS(prefill_scheduler_qps_metric);
    if (!collector->success) {
        REPORT_MUTABLE_QPS(prefill_scheduler_failed_qps_metric);
    }
    if (collector->total_cost_time_us > 0) {
        REPORT_MUTABLE_METRIC(prefill_scheduler_total_cost_time_us_metric, collector->total_cost_time_us);
    }
}

void P2PConnectorMetrics::report(const kmonitor::MetricsTags* tags, PrefillWorkerSendMetricsCollector* collector) {
    if (collector->add_buffer_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_worker_write_add_buffer_time_us_metric, collector->add_buffer_time_us);
    }
    if (collector->dispatch_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_worker_write_dispatch_time_us_metric, collector->dispatch_time_us);
    }
    if (collector->callback_wait_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_worker_write_callback_wait_time_us_metric, collector->callback_wait_time_us);
    }
    if (collector->sender_queue_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_worker_write_sender_queue_time_us_metric, collector->sender_queue_time_us);
    }
    if (collector->send_submit_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_worker_write_send_submit_time_us_metric, collector->send_submit_time_us);
    }
    if (collector->send_complete_time_us >= 0) {
        kmonitor::MetricsTags transfer_tags = tags ? *tags : kmonitor::MetricsTags();
        transfer_tags.AddTag("success", collector->success ? "true" : "false");
        prefill_worker_write_send_complete_time_us_metric->Report(&transfer_tags, collector->send_complete_time_us);
    }
    if (collector->total_cost_time_us < 0) {
        return;
    }
    REPORT_MUTABLE_QPS(prefill_worker_write_qps_metric);
    if (!collector->success) {
        REPORT_MUTABLE_QPS(prefill_worker_write_failed_qps_metric);
    }
    if (collector->first_layer_wait_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_worker_write_first_layer_wait_time_us_metric,
                              collector->first_layer_wait_time_us);
    }
    if (collector->last_layer_wait_time_us >= 0) {
        REPORT_MUTABLE_METRIC(prefill_worker_write_last_layer_wait_time_us_metric, collector->last_layer_wait_time_us);
    }
    if (collector->total_cost_time_us > 0) {
        REPORT_MUTABLE_METRIC(prefill_worker_write_total_cost_time_us_metric, collector->total_cost_time_us);
    }
}

void P2PConnectorMetrics::report(const kmonitor::MetricsTags* tags, PrefillWorkerStatusMetricsCollector* collector) {
    REPORT_MUTABLE_METRIC(prefill_worker_wait_store_event_count_metric, collector->wait_store_event_count);
    REPORT_MUTABLE_METRIC(prefill_worker_task_count_metric, collector->task_count);
    REPORT_MUTABLE_METRIC(prefill_worker_computed_request_count_metric, collector->computed_request_count);
}

void P2PConnectorMetrics::report(const kmonitor::MetricsTags* tags, PrefillWorkerStoreMetricsCollector* collector) {
    REPORT_MUTABLE_QPS(prefill_worker_store_qps_metric);
    if (!collector->success) {
        REPORT_MUTABLE_QPS(prefill_worker_store_failed_qps_metric);
    }
    if (collector->total_block_count > 0) {
        REPORT_MUTABLE_METRIC(prefill_worker_store_total_block_count_metric, collector->total_block_count);
    }
    if (collector->store_wait_done_time_us > 0) {
        REPORT_MUTABLE_METRIC(prefill_worker_store_store_wait_done_time_us_metric, collector->store_wait_done_time_us);
    }
}

void P2PConnectorMetrics::report(const kmonitor::MetricsTags* tags, CacheWriteOpFailureMetricsCollector* collector) {
    REPORT_MUTABLE_QPS(cache_write_op_failure_qps_metric);
}
}  // namespace rtp_llm
