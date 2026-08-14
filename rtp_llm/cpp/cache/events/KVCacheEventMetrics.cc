#include "rtp_llm/cpp/cache/events/KVCacheEventMetrics.h"

namespace rtp_llm {

AUTIL_LOG_SETUP(rtp_llm, RtpLLMKVCacheEventMetrics);

bool RtpLLMKVCacheEventMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_STATUS_MUTABLE_METRIC(publisher_state_metric, "rtp_llm_kv_cache_event_publisher_state");
    REGISTER_GAUGE_MUTABLE_METRIC(queue_size_metric, "rtp_llm_kv_cache_event_queue_size");
    REGISTER_GAUGE_MUTABLE_METRIC(queue_high_watermark_metric, "rtp_llm_kv_cache_event_queue_high_watermark");
    REGISTER_GAUGE_MUTABLE_METRIC(accepted_count_metric, "rtp_llm_kv_cache_event_accepted_count");
    REGISTER_GAUGE_MUTABLE_METRIC(dropped_count_metric, "rtp_llm_kv_cache_event_dropped_count");
    REGISTER_GAUGE_MUTABLE_METRIC(request_failure_count_metric, "rtp_llm_kv_cache_event_request_failure_count");
    REGISTER_GAUGE_MUTABLE_METRIC(overflow_recovery_count_metric, "rtp_llm_kv_cache_event_overflow_recovery_count");
    REGISTER_GAUGE_MUTABLE_METRIC(snapshot_attempt_count_metric, "rtp_llm_kv_cache_event_snapshot_attempt_count");
    REGISTER_GAUGE_MUTABLE_METRIC(snapshot_commit_count_metric, "rtp_llm_kv_cache_event_snapshot_commit_count");
    REGISTER_GAUGE_MUTABLE_METRIC(cache_insert_rejected_count_metric,
                                  "rtp_llm_kv_cache_event_cache_insert_rejected_count");
    // MetricsReporter propagates this result to KVCacheManager's isolated
    // retry boundary. Do not claim success when kmonitor logged a failed
    // declaration and left one series permanently silent.
    return publisher_state_metric && queue_size_metric && queue_high_watermark_metric && accepted_count_metric
           && dropped_count_metric && request_failure_count_metric && overflow_recovery_count_metric
           && snapshot_attempt_count_metric && snapshot_commit_count_metric && cache_insert_rejected_count_metric;
}

void RtpLLMKVCacheEventMetrics::report(const kmonitor::MetricsTags*        tags,
                                       RtpLLMKVCacheEventMetricsCollector* collector) {
    REPORT_MUTABLE_METRIC(publisher_state_metric, collector->publisher_state);
    REPORT_MUTABLE_METRIC(cache_insert_rejected_count_metric, collector->cache_insert_rejected_count);
    if (collector->suppress_publisher_details) {
        return;
    }
    REPORT_MUTABLE_METRIC(queue_size_metric, collector->queue_size);
    REPORT_MUTABLE_METRIC(queue_high_watermark_metric, collector->queue_high_watermark);
    REPORT_MUTABLE_METRIC(accepted_count_metric, collector->accepted_count);
    REPORT_MUTABLE_METRIC(dropped_count_metric, collector->dropped_count);
    REPORT_MUTABLE_METRIC(request_failure_count_metric, collector->request_failure_count);
    REPORT_MUTABLE_METRIC(overflow_recovery_count_metric, collector->overflow_recovery_count);
    REPORT_MUTABLE_METRIC(snapshot_attempt_count_metric, collector->snapshot_attempt_count);
    REPORT_MUTABLE_METRIC(snapshot_commit_count_metric, collector->snapshot_commit_count);
}

}  // namespace rtp_llm
