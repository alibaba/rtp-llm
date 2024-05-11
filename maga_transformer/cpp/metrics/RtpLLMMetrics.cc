#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {

AUTIL_LOG_SETUP(rtp_llm, RtpLLMStreamMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpLLMSchedulerMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpLLMCacheMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpLLMExecutorMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpLLMEngineMetrics);

#define REPORT_QPS(name)                                                                                               \
    if (collector->name) {                                                                                             \
        REPORT_MUTABLE_QPS(name##_metric);                                                                             \
    }

#define REPORT_GAUGE(name)                                                                                             \
    if (collector->name) {                                                                                             \
        REPORT_MUTABLE_METRIC(name##_metric, collector->name);                                                         \
    }

bool RtpLLMStreamMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_QPS_MUTABLE_METRIC(qps_metric, "rtp_llm_framework_qps");
    REGISTER_QPS_MUTABLE_METRIC(error_qps_metric, "rtp_llm_framework_error_qps");
    REGISTER_QPS_MUTABLE_METRIC(cancel_qps_metric, "rtp_llm_cancal_qps");

    REGISTER_GAUGE_MUTABLE_METRIC(total_latency_us_metric, "rtp_llm_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(first_token_latency_us_metric, "rtp_llm_first_token_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(wait_latency_us_metric, "rtp_llm_wait_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(iterate_cout_metric, "rtp_llm_iterate_cout");
    REGISTER_GAUGE_MUTABLE_METRIC(reuse_length_metric, "rtp_llm_reuse_length");
    REGISTER_GAUGE_MUTABLE_METRIC(input_token_length_metric, "rtp_llm_input_token_length");
    REGISTER_GAUGE_MUTABLE_METRIC(output_token_length_metric, "rtp_llm_output_token_length");
    REGISTER_GAUGE_MUTABLE_METRIC(query_batch_size_metric, "rtp_llm_query_batch_size");
    return true;
}

void RtpLLMStreamMetrics::report(const kmonitor::MetricsTags* tags, RtpLLMStreamMetricsCollector* collector) {

    REPORT_QPS(qps);
    REPORT_QPS(cancel_qps);
    REPORT_QPS(error_qps);

    REPORT_GAUGE(total_latency_us);
    REPORT_GAUGE(first_token_latency_us);
    REPORT_GAUGE(wait_latency_us);
    REPORT_GAUGE(iterate_cout);
    REPORT_GAUGE(reuse_length);
    REPORT_GAUGE(input_token_length);
    REPORT_GAUGE(output_token_length);
    REPORT_GAUGE(query_batch_size);
}

bool RtpLLMSchedulerMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(fallback_stream_size_metric, "rtp_llm_fallback_stream_size");
    REGISTER_GAUGE_MUTABLE_METRIC(wait_stream_size_metric, "rtp_llm_wait_stream_size");
    REGISTER_GAUGE_MUTABLE_METRIC(running_stream_size_metric, "rtp_llm_running_stream_size");
    return true;
}

void RtpLLMSchedulerMetrics::report(const kmonitor::MetricsTags* tags, RtpLLMSchedulerMetricsCollector* collector) {
    REPORT_MUTABLE_METRIC(fallback_stream_size_metric, collector->fallback_stream_size);
    REPORT_MUTABLE_METRIC(wait_stream_size_metric, collector->wait_stream_size);
    REPORT_MUTABLE_METRIC(running_stream_size_metric, collector->running_stream_size);
}

bool RtpLLMEngineMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_QPS_MUTABLE_METRIC(update_lora_qps_metric, "rtp_llm_update_lora_qps");
    REGISTER_QPS_MUTABLE_METRIC(error_update_lora_qps_metric, "rtp_llm_error_update_lora_qps");

    REGISTER_GAUGE_MUTABLE_METRIC(step_latency_us_metric, "rtp_llm_step_latency_us");
    return true;
}

void RtpLLMEngineMetrics::report(const kmonitor::MetricsTags* tags, RtpLLMEngineMetricsCollector* collector) {

    REPORT_QPS(update_lora_qps);
    REPORT_QPS(error_update_lora_qps);

    REPORT_GAUGE(step_latency_us);
}

bool RtpLLMExecutorMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(context_batch_size_metric, "rtp_llm_context_batch_size");
    REGISTER_GAUGE_MUTABLE_METRIC(generate_batch_size_metric, "rtp_llm_generate_batch_size");
    REGISTER_GAUGE_MUTABLE_METRIC(execute_token_size_metric, "rtp_llm_execute_token_size");
    REGISTER_GAUGE_MUTABLE_METRIC(max_seq_len, "rtp_llm_max_seq_len");
    return true;
}

void RtpLLMExecutorMetrics::report(const kmonitor::MetricsTags* tags, RtpLLMExecutorMetricsCollector* collector) {
    REPORT_MUTABLE_METRIC(context_batch_size_metric, collector->context_batch_size);
    REPORT_MUTABLE_METRIC(generate_batch_size_metric, collector->generate_batch_size);
    REPORT_MUTABLE_METRIC(execute_token_size_metric, collector->execute_token_size);
    REPORT_MUTABLE_METRIC(max_seq_len, collector->max_seq_len);
}

bool RtpLLMCacheMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(kv_cache_item_num_metric, "rtp_llm_context_batch_size");
    REGISTER_GAUGE_MUTABLE_METRIC(kv_cache_left_seq_metric, "rtp_llm_generate_batch_size");
    return true;
}

void RtpLLMCacheMetrics::report(const kmonitor::MetricsTags* tags, RtpLLMCacheMetricsCollector* collector) {
    REPORT_MUTABLE_METRIC(kv_cache_item_num_metric, collector->kv_cache_item_num);
    REPORT_MUTABLE_METRIC(kv_cache_left_seq_metric, collector->kv_cache_left_seq);
}

#undef REPORT_QPS
#undef REPORT_GAUGE

}  // namespace rtp_llm
