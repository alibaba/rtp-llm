#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"

namespace rtp_llm {

bool P2PConnectorMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_QPS_MUTABLE_METRIC(writeback_qps_metric, "rtp_llm_p2p_writeback_qps");
    REGISTER_QPS_MUTABLE_METRIC(writeback_skipped_qps_metric, "rtp_llm_p2p_writeback_skipped_qps");
    REGISTER_QPS_MUTABLE_METRIC(writeback_failed_qps_metric, "rtp_llm_p2p_writeback_failed_qps");
    REGISTER_QPS_MUTABLE_METRIC(writeback_no_transfer_qps_metric, "rtp_llm_p2p_writeback_no_transfer_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(writeback_cost_time_us_metric, "rtp_llm_p2p_writeback_cost_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(writeback_hold_time_us_metric, "rtp_llm_p2p_writeback_hold_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(writeback_planned_bytes_metric, "rtp_llm_p2p_writeback_planned_bytes");
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
    REPORT_MUTABLE_METRIC(writeback_hold_time_us_metric, collector->hold_time_us);
    REPORT_MUTABLE_METRIC(writeback_planned_bytes_metric, collector->planned_bytes);
}

void P2PConnectorMetrics::report(const kmonitor::MetricsTags* tags, DecodeSchedulerMetricsCollector* collector) {
    REPORT_MUTABLE_QPS(decode_schedule_qps_metric);
    if (!collector->success) {
        REPORT_MUTABLE_QPS(decode_schedule_failed_qps_metric);
    }
    if (collector->total_cost_time_us > 0) {
        REPORT_MUTABLE_METRIC(decode_schedule_cost_time_us_metric, collector->total_cost_time_us);
    }
}

void P2PConnectorMetrics::report(const kmonitor::MetricsTags* tags, DecodeWorkerMetricsCollector* collector) {
    REPORT_MUTABLE_QPS(decode_worker_qps_metric);
    if (!collector->success) {
        REPORT_MUTABLE_QPS(decode_worker_failed_qps_metric);
    }
    REPORT_MUTABLE_METRIC(decode_worker_total_block_count_metric, collector->total_block_count);
    if (collector->first_layer_wait_time_us > 0) {
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
    REPORT_MUTABLE_QPS(prefill_scheduler_qps_metric);
    if (!collector->success) {
        REPORT_MUTABLE_QPS(prefill_scheduler_failed_qps_metric);
    }
    if (collector->total_cost_time_us > 0) {
        REPORT_MUTABLE_METRIC(prefill_scheduler_total_cost_time_us_metric, collector->total_cost_time_us);
    }
}

void P2PConnectorMetrics::report(const kmonitor::MetricsTags* tags, PrefillWorkerSendMetricsCollector* collector) {
    REPORT_MUTABLE_QPS(prefill_worker_write_qps_metric);
    if (!collector->success) {
        REPORT_MUTABLE_QPS(prefill_worker_write_failed_qps_metric);
    }
    if (collector->first_layer_wait_time_us > 0) {
        REPORT_MUTABLE_METRIC(prefill_worker_write_first_layer_wait_time_us_metric,
                              collector->first_layer_wait_time_us);
    }
    if (collector->last_layer_wait_time_us > 0) {
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
