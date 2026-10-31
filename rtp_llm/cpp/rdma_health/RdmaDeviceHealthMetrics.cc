#include "rtp_llm/cpp/rdma_health/RdmaDeviceHealthMetrics.h"

namespace rtp_llm {
namespace rdma_health {

bool RdmaDeviceHealthMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(query_result_metric, "rtp_llm.rdma_device_health.query_result");
    REGISTER_QPS_MUTABLE_METRIC(query_error_qps_metric, "rtp_llm.rdma_device_health.query_error_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(state_metric, "rtp_llm.rdma_device_health.state");
    REGISTER_GAUGE_MUTABLE_METRIC(total_count_metric, "rtp_llm.rdma_device_health.total_count");
    REGISTER_GAUGE_MUTABLE_METRIC(fault_count_metric, "rtp_llm.rdma_device_health.fault_count");
    REGISTER_GAUGE_MUTABLE_METRIC(confirmed_fault_metric, "rtp_llm.rdma_device_health.confirmed_fault");
    REGISTER_GAUGE_MUTABLE_METRIC(fault_duration_ms_metric, "rtp_llm.rdma_device_health.fault_duration_ms");
    REGISTER_GAUGE_MUTABLE_METRIC(recover_count_in_window_metric, "rtp_llm.rdma_device_health.recover_count_in_window");
    REGISTER_GAUGE_MUTABLE_METRIC(unavailable_ms_in_window_metric,
                                  "rtp_llm.rdma_device_health.unavailable_ms_in_window");
    REGISTER_GAUGE_MUTABLE_METRIC(probe_latency_us_metric, "rtp_llm.rdma_device_health.probe_latency_us");
    return true;
}

void RdmaDeviceHealthMetrics::report(const kmonitor::MetricsTags* tags, RdmaDeviceHealthMetricsCollector* collector) {
    REPORT_MUTABLE_METRIC(query_result_metric, collector->query_result);
    if (collector->query_failed) {
        REPORT_MUTABLE_QPS(query_error_qps_metric);
    }
    REPORT_MUTABLE_METRIC(state_metric, collector->state);
    REPORT_MUTABLE_METRIC(total_count_metric, collector->total_count);
    REPORT_MUTABLE_METRIC(fault_count_metric, collector->fault_count);
    REPORT_MUTABLE_METRIC(confirmed_fault_metric, collector->confirmed_fault);
    REPORT_MUTABLE_METRIC(fault_duration_ms_metric, collector->fault_duration_ms);
    REPORT_MUTABLE_METRIC(recover_count_in_window_metric, collector->recover_count_in_window);
    REPORT_MUTABLE_METRIC(unavailable_ms_in_window_metric, collector->unavailable_ms_in_window);
    REPORT_MUTABLE_METRIC(probe_latency_us_metric, collector->probe_latency_us);
}

}  // namespace rdma_health
}  // namespace rtp_llm
