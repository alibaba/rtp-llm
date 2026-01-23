#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferMetric.h"

#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

bool TransferMetric::init(kmonitor::MetricsGroupManager* manager) {

    // client metrics
    REGISTER_QPS_MUTABLE_METRIC(transfer_client_qps_metric, "rtp_llm.transfer.client.qps");
    REGISTER_QPS_MUTABLE_METRIC(transfer_client_error_qps_metric, "rtp_llm.transfer.client.error_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(transfer_client_block_count_metric, "rtp_llm.transfer.client.block_count");
    REGISTER_GAUGE_MUTABLE_METRIC(transfer_client_total_block_size_metric, "rtp_llm.transfer.client.total_block_size");
    REGISTER_GAUGE_MUTABLE_METRIC(transfer_client_latency_us_metric, "rtp_llm.transfer.client.latency_us");

    // server metrics
    REGISTER_QPS_MUTABLE_METRIC(transfer_server_qps_metric, "rtp_llm.transfer.server.qps");
    REGISTER_QPS_MUTABLE_METRIC(transfer_server_error_qps_metric, "rtp_llm.transfer.server.error_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(transfer_server_block_count_metric, "rtp_llm.transfer.server.block_count");
    REGISTER_GAUGE_MUTABLE_METRIC(transfer_server_total_block_size_metric, "rtp_llm.transfer.server.block_size");
    REGISTER_GAUGE_MUTABLE_METRIC(transfer_server_wait_task_latency_us_metric,
                                  "rtp_llm.transfer.server.wait_task_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(transfer_server_latency_us_metric, "rtp_llm.transfer.server.latency_us");
    return true;
}

void TransferMetric::report(const kmonitor::MetricsTags* tags, TransferClientTransferMetricsCollector* collector) {
    REPORT_MUTABLE_QPS(transfer_client_qps_metric);
    if (!collector->success) {
        REPORT_MUTABLE_QPS(transfer_client_error_qps_metric);
    }
    if (collector->block_count > 0) {
        REPORT_MUTABLE_METRIC(transfer_client_block_count_metric, collector->block_count);
    }
    if (collector->total_block_size > 0) {
        REPORT_MUTABLE_METRIC(transfer_client_total_block_size_metric, collector->total_block_size);
    }
    if (collector->latency_us > 0) {
        REPORT_MUTABLE_METRIC(transfer_client_latency_us_metric, collector->latency_us);
    }
}

void TransferMetric::report(const kmonitor::MetricsTags* tags, TransferServerTransferMetricsCollector* collector) {
    REPORT_MUTABLE_QPS(transfer_server_qps_metric);
    if (!collector->success) {
        REPORT_MUTABLE_QPS(transfer_server_error_qps_metric);
    }
    if (collector->block_count > 0) {
        REPORT_MUTABLE_METRIC(transfer_server_block_count_metric, collector->block_count);
    }
    if (collector->total_block_size > 0) {
        REPORT_MUTABLE_METRIC(transfer_server_total_block_size_metric, collector->total_block_size);
    }
    if (collector->wait_task_run_latency_us > 0) {
        REPORT_MUTABLE_METRIC(transfer_server_wait_task_latency_us_metric, collector->wait_task_run_latency_us);
    }
    if (collector->total_cost_latency_us > 0) {
        REPORT_MUTABLE_METRIC(transfer_server_latency_us_metric, collector->total_cost_latency_us);
    }
}

}  // namespace rtp_llm