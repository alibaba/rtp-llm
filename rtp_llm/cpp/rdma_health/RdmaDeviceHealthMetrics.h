#pragma once

#include <cstdint>

#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {
namespace rdma_health {

class RdmaDeviceHealthMetricsCollector final {
public:
    int64_t query_result = 0;
    bool    query_failed = false;
    // 状态编码对应 XTransportDeviceHealthState：0=AVAILABLE、1=DEGRADED、2=UNAVAILABLE、3=UNKNOWN。
    int64_t state                    = 0;
    int64_t total_count              = 0;
    int64_t fault_count              = 0;
    // 历史确认标记：首次设备故障确认前为 0，确认后保持 1，恢复不清零。
    int64_t confirmed_fault          = 0;
    // 当前总体故障期经过时间，包含中间 UNKNOWN/DEGRADED；AVAILABLE 关闭故障期后为 0。
    int64_t fault_duration_ms        = 0;
    int64_t recover_count_in_window  = 0;
    // 窗口内总体故障期累计经过时间，包含中间 UNKNOWN/DEGRADED，不仅是 UNAVAILABLE 样本时长。
    int64_t unavailable_ms_in_window = 0;
    int64_t probe_latency_us         = 0;
};

class RdmaDeviceHealthMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RdmaDeviceHealthMetricsCollector* collector);

private:
    kmonitor::MutableMetric* query_result_metric             = nullptr;
    kmonitor::MutableMetric* query_error_qps_metric          = nullptr;
    kmonitor::MutableMetric* state_metric                    = nullptr;
    kmonitor::MutableMetric* total_count_metric              = nullptr;
    kmonitor::MutableMetric* fault_count_metric              = nullptr;
    kmonitor::MutableMetric* confirmed_fault_metric          = nullptr;
    kmonitor::MutableMetric* fault_duration_ms_metric        = nullptr;
    kmonitor::MutableMetric* recover_count_in_window_metric  = nullptr;
    kmonitor::MutableMetric* unavailable_ms_in_window_metric = nullptr;
    kmonitor::MutableMetric* probe_latency_us_metric         = nullptr;
};

}  // namespace rdma_health
}  // namespace rtp_llm
