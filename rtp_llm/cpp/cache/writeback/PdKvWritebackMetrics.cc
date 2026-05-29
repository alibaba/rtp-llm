#include "rtp_llm/cpp/cache/writeback/PdKvWritebackMetrics.h"

#include <string>

namespace rtp_llm {

bool PdKvWritebackMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_QPS_MUTABLE_METRIC(launch_qps_metric, "rtp_llm_pd_kv_writeback_launch_qps");
    REGISTER_QPS_MUTABLE_METRIC(launch_failed_qps_metric, "rtp_llm_pd_kv_writeback_launch_failed_qps");
    REGISTER_QPS_MUTABLE_METRIC(launch_skipped_qps_metric, "rtp_llm_pd_kv_writeback_launch_skipped_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(launch_latency_us_metric, "rtp_llm_pd_kv_writeback_launch_latency_us");
    REGISTER_QPS_MUTABLE_METRIC(rpc_qps_metric, "rtp_llm_pd_kv_writeback_rpc_qps");
    REGISTER_QPS_MUTABLE_METRIC(rpc_failed_qps_metric, "rtp_llm_pd_kv_writeback_rpc_failed_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(rpc_latency_us_metric, "rtp_llm_pd_kv_writeback_rpc_latency_us");
    REGISTER_QPS_MUTABLE_METRIC(transfer_qps_metric, "rtp_llm_pd_kv_writeback_transfer_qps");
    REGISTER_QPS_MUTABLE_METRIC(transfer_failed_qps_metric, "rtp_llm_pd_kv_writeback_transfer_failed_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(transfer_latency_us_metric, "rtp_llm_pd_kv_writeback_transfer_latency_us");
    REGISTER_QPS_MUTABLE_METRIC(receive_qps_metric, "rtp_llm_pd_kv_writeback_receive_qps");
    REGISTER_QPS_MUTABLE_METRIC(receive_failed_qps_metric, "rtp_llm_pd_kv_writeback_receive_failed_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(receive_latency_us_metric, "rtp_llm_pd_kv_writeback_receive_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(malloc_latency_us_metric, "rtp_llm_pd_kv_writeback_malloc_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(commit_latency_us_metric, "rtp_llm_pd_kv_writeback_commit_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(block_count_metric, "rtp_llm_pd_kv_writeback_block_count");
    REGISTER_GAUGE_MUTABLE_METRIC(token_count_metric, "rtp_llm_pd_kv_writeback_token_count");
    return true;
}

void PdKvWritebackMetrics::report(const kmonitor::MetricsTags* tags, PdKvWritebackMetricsCollector* collector) {
    if (collector->launch_qps) {
        REPORT_MUTABLE_QPS(launch_qps_metric);
    }
    if (collector->launch_failed_qps) {
        REPORT_MUTABLE_QPS(launch_failed_qps_metric);
    }
    if (collector->launch_skipped_qps) {
        REPORT_MUTABLE_QPS(launch_skipped_qps_metric);
    }
    if (collector->launch_latency_us > 0) {
        REPORT_MUTABLE_METRIC(launch_latency_us_metric, collector->launch_latency_us);
    }
    if (collector->rpc_qps) {
        REPORT_MUTABLE_QPS(rpc_qps_metric);
    }
    if (collector->rpc_failed_qps) {
        REPORT_MUTABLE_QPS(rpc_failed_qps_metric);
    }
    if (collector->rpc_latency_us > 0) {
        REPORT_MUTABLE_METRIC(rpc_latency_us_metric, collector->rpc_latency_us);
    }
    if (collector->transfer_qps) {
        REPORT_MUTABLE_QPS(transfer_qps_metric);
    }
    if (collector->transfer_failed_qps) {
        REPORT_MUTABLE_QPS(transfer_failed_qps_metric);
    }
    if (collector->transfer_latency_us > 0) {
        REPORT_MUTABLE_METRIC(transfer_latency_us_metric, collector->transfer_latency_us);
    }
    if (collector->receive_qps) {
        REPORT_MUTABLE_QPS(receive_qps_metric);
    }
    if (collector->receive_failed_qps) {
        REPORT_MUTABLE_QPS(receive_failed_qps_metric);
    }
    if (collector->receive_latency_us > 0) {
        REPORT_MUTABLE_METRIC(receive_latency_us_metric, collector->receive_latency_us);
    }
    if (collector->malloc_latency_us > 0) {
        REPORT_MUTABLE_METRIC(malloc_latency_us_metric, collector->malloc_latency_us);
    }
    if (collector->commit_latency_us > 0) {
        REPORT_MUTABLE_METRIC(commit_latency_us_metric, collector->commit_latency_us);
    }
    if (collector->block_count > 0) {
        REPORT_MUTABLE_METRIC(block_count_metric, collector->block_count);
    }
    if (collector->token_count > 0) {
        REPORT_MUTABLE_METRIC(token_count_metric, collector->token_count);
    }
}

void reportPdKvWritebackMetric(const kmonitor::MetricsReporterPtr& reporter,
                               PdKvWritebackMetricsCollector&      collector,
                               const std::string&                  stage,
                               const std::string&                  status,
                               const std::string&                  reason,
                               const std::string&                  role,
                               int32_t                             tp_size,
                               const std::string&                  topology,
                               const PdKvWritebackMetricExtraTags& extra_tags) {
    if (!reporter) {
        return;
    }
    kmonitor::MetricsTags tags;
    tags.AddTag("stage", stage);
    tags.AddTag("status", status);
    tags.AddTag("reason", reason);
    tags.AddTag("role", role);
    tags.AddTag("tp_size", std::to_string(tp_size));
    tags.AddTag("topology", topology);
    for (const auto& tag : extra_tags) {
        tags.AddTag(tag.first, tag.second);
    }
    reporter->report<PdKvWritebackMetrics, PdKvWritebackMetricsCollector>(&tags, &collector);
}

}  // namespace rtp_llm
