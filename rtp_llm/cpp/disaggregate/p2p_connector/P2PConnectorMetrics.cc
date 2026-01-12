#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorMetrics.h"

namespace rtp_llm {

bool P2PConnectorMetrics::init(kmonitor::MetricsGroupManager* manager) {
    // decode schedule metrics
    REGISTER_QPS_MUTABLE_METRIC(decode_schedule_qps_metric, "rtp_llm.p2p_connector.decode_schedule.qps");
    REGISTER_QPS_MUTABLE_METRIC(decode_schedule_failed_qps_metric, "rtp_llm.p2p_connector.decode_schedule.failed_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_schedule_cost_time_us_metric,
                                  "rtp_llm.p2p_connector.decode_schedule.cost_time_us");

    // decode worker metrics
    REGISTER_QPS_MUTABLE_METRIC(decode_worker_qps_metric, "rtp_llm.p2p_connector.decode_worker.qps");
    REGISTER_QPS_MUTABLE_METRIC(decode_worker_failed_qps_metric, "rtp_llm.p2p_connector.decode_worker.failed_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_worker_total_block_count_metric,
                                  "rtp_llm.p2p_connector.decode_worker.total_block_count");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_worker_first_layer_wait_time_us_metric,
                                  "rtp_llm.p2p_connector.decode_worker.first_layer_wait_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_worker_total_cost_time_us_metric,
                                  "rtp_llm.p2p_connector.decode_worker.total_cost_time_us");

    // decode scheduler status metrics
    REGISTER_GAUGE_MUTABLE_METRIC(decode_scheduler_check_once_cost_time_us_metric,
                                  "rtp_llm.p2p_connector.decode_scheduler.check_once_cost_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(decode_scheduler_inflight_context_count_metric,
                                  "rtp_llm.p2p_connector.decode_scheduler.inflight_context_count");

    // stream store metrics
    REGISTER_GAUGE_MUTABLE_METRIC(stream_store_stream_count_metric, "rtp_llm.p2p_connector.stream_store.stream_count");
    REGISTER_QPS_MUTABLE_METRIC(stream_store_timeout_count_metric, "rtp_llm.p2p_connector.stream_store.timeout_count");
    REGISTER_GAUGE_MUTABLE_METRIC(stream_store_stream_wait_time_us_metric,
                                  "rtp_llm.p2p_connector.stream_store.stream_wait_time_us");

    // prefill scheduler metrics
    REGISTER_QPS_MUTABLE_METRIC(prefill_scheduler_qps_metric, "rtp_llm.p2p_connector.prefill_scheduler.qps");
    REGISTER_QPS_MUTABLE_METRIC(prefill_scheduler_failed_qps_metric,
                                "rtp_llm.p2p_connector.prefill_scheduler.failed_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_scheduler_total_cost_time_us_metric,
                                  "rtp_llm.p2p_connector.prefill_scheduler.total_cost_time_us");

    // prefill worker store metrics
    REGISTER_QPS_MUTABLE_METRIC(prefill_worker_store_qps_metric, "rtp_llm.p2p_connector.prefill_worker_store.qps");
    REGISTER_QPS_MUTABLE_METRIC(prefill_worker_store_failed_qps_metric,
                                "rtp_llm.p2p_connector.prefill_worker_store.failed_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_worker_store_total_block_count_metric,
                                  "rtp_llm.p2p_connector.prefill_worker_store.total_block_count");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_worker_store_store_wait_done_time_us_metric,
                                  "rtp_llm.p2p_connector.prefill_worker_store.store_wait_done_time_us");

    // prefill worker write metrics
    REGISTER_QPS_MUTABLE_METRIC(prefill_worker_write_qps_metric, "rtp_llm.p2p_connector.prefill_worker_write.qps");
    REGISTER_QPS_MUTABLE_METRIC(prefill_worker_write_failed_qps_metric,
                                "rtp_llm.p2p_connector.prefill_worker_write.failed_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_worker_write_first_layer_wait_time_us_metric,
                                  "rtp_llm.p2p_connector.prefill_worker_write.first_layer_wait_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_worker_write_last_layer_wait_time_us_metric,
                                  "rtp_llm.p2p_connector.prefill_worker_write.last_layer_wait_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_worker_write_total_cost_time_us_metric,
                                  "rtp_llm.p2p_connector.prefill_worker_write.total_cost_time_us");

    // prefill worker status metrics
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_worker_wait_store_event_count_metric,
                                  "rtp_llm.p2p_connector.prefill_worker.wait_store_event_count");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_worker_task_count_metric, "rtp_llm.p2p_connector.prefill_worker.task_count");
    REGISTER_GAUGE_MUTABLE_METRIC(prefill_worker_computed_request_count_metric,
                                  "rtp_llm.p2p_connector.prefill_worker.computed_request_count");

    return true;
}

void P2PConnectorMetrics::report(const kmonitor::MetricsTags*                 tags,
                                 P2PConnectorClientSchedulerMetricsCollector* collector) {
    REPORT_MUTABLE_QPS(decode_schedule_qps_metric);
    if (!collector->success) {
        REPORT_MUTABLE_QPS(decode_schedule_failed_qps_metric);
    }
    if (collector->total_cost_time_us > 0) {
        REPORT_MUTABLE_METRIC(decode_schedule_cost_time_us_metric, collector->total_cost_time_us);
    }
}

void P2PConnectorMetrics::report(const kmonitor::MetricsTags*              tags,
                                 P2PConnectorClientWorkerMetricsCollector* collector) {
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

void P2PConnectorMetrics::report(const kmonitor::MetricsTags*                       tags,
                                 P2PConnectorClientSchedulerStatusMetricsCollector* collector) {
    REPORT_MUTABLE_METRIC(decode_scheduler_check_once_cost_time_us_metric, collector->check_once_cost_time_us);
    REPORT_MUTABLE_METRIC(decode_scheduler_inflight_context_count_metric, collector->inflight_context_count);
}

void P2PConnectorMetrics::report(const kmonitor::MetricsTags*              tags,
                                 P2PConnectorStreamStoreMetricsCollector1* collector) {
    REPORT_MUTABLE_METRIC(stream_store_stream_count_metric, collector->stream_count);
}

void P2PConnectorMetrics::report(const kmonitor::MetricsTags*              tags,
                                 P2PConnectorStreamStoreMetricsCollector2* collector) {
    if (collector->timeout) {
        REPORT_MUTABLE_QPS(stream_store_timeout_count_metric);
    }
    if (collector->stream_wait_time_us > 0) {
        REPORT_MUTABLE_METRIC(stream_store_stream_wait_time_us_metric, collector->stream_wait_time_us);
    }
}
//
void P2PConnectorMetrics::report(const kmonitor::MetricsTags*                 tags,
                                 P2PConnectorServerSchedulerMetricsCollector* collector) {
    REPORT_MUTABLE_QPS(prefill_scheduler_qps_metric);
    if (!collector->success) {
        REPORT_MUTABLE_QPS(prefill_scheduler_failed_qps_metric);
    }
    if (collector->total_cost_time_us > 0) {
        REPORT_MUTABLE_METRIC(prefill_scheduler_total_cost_time_us_metric, collector->total_cost_time_us);
    }
}

void P2PConnectorMetrics::report(const kmonitor::MetricsTags*                   tags,
                                 P2PConnectorServerWorkerWriteMetricsCollector* collector) {
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

void P2PConnectorMetrics::report(const kmonitor::MetricsTags*                    tags,
                                 P2PConnectorServerWorkerStatusMetricsCollector* collector) {
    REPORT_MUTABLE_METRIC(prefill_worker_wait_store_event_count_metric, collector->wait_store_event_count);
    REPORT_MUTABLE_METRIC(prefill_worker_task_count_metric, collector->task_count);
    REPORT_MUTABLE_METRIC(prefill_worker_computed_request_count_metric, collector->computed_request_count);
}

void P2PConnectorMetrics::report(const kmonitor::MetricsTags*                   tags,
                                 P2PConnectorServerWorkerStoreMetricsCollector* collector) {
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
}  // namespace rtp_llm