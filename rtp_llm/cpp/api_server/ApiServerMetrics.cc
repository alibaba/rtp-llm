#include "rtp_llm/cpp/api_server/ApiServerMetrics.h"
#include "kmonitor/client/KMonitorFactory.h"

namespace rtp_llm {

AUTIL_LOG_SETUP(rtp_llm, ApiServerMetricReporter);

bool ApiServerMetricReporter::init() {
    auto kMonitor = kmonitor::KMonitorFactory::GetKMonitor("cache_store");
    if (kMonitor == nullptr) {
        AUTIL_LOG(WARN, "api server metric reporter init failed");
        return false;
    }
    kMonitor->SetServiceName("rtp_llm.api_server");

#define LOCAL_REGISTER_QPS_MUTABLE_METRIC(target, name)                                                                \
    do {                                                                                                               \
        std::string metricName = (name);                                                                               \
        target.reset(kMonitor->RegisterMetric(metricName, kmonitor::QPS, kmonitor::FATAL));                           \
        if (nullptr == target) {                                                                                       \
            AUTIL_LOG(ERROR, "failed to register metric:[%s]", metricName.c_str());                                    \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

#define LOCAL_REGISTER_GAUGE_MUTABLE_METRIC(target, name)                                                              \
    do {                                                                                                               \
        std::string metricName = (name);                                                                               \
        target.reset(kMonitor->RegisterMetric(metricName, kmonitor::GAUGE, kmonitor::FATAL));                         \
        if (nullptr == target) {                                                                                       \
            AUTIL_LOG(ERROR, "failed to register metric:[%s]", metricName.c_str());                                    \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

    LOCAL_REGISTER_QPS_MUTABLE_METRIC(cancel_qps_metric_, "cancel_qps");
    LOCAL_REGISTER_QPS_MUTABLE_METRIC(success_qps_metric_, "success_qps");
    LOCAL_REGISTER_QPS_MUTABLE_METRIC(framework_qps_metric_, "qps");
    LOCAL_REGISTER_QPS_MUTABLE_METRIC(framework_error_qps_metric_, "error_qps");
    LOCAL_REGISTER_QPS_MUTABLE_METRIC(framework_concurrency_exception_qps_metric_, "concurrency_exception_qps");

    LOCAL_REGISTER_GAUGE_MUTABLE_METRIC(framework_rt_metric_, "rt");
    LOCAL_REGISTER_GAUGE_MUTABLE_METRIC(response_first_token_rt_metric_, "response_first_token_rt");
    LOCAL_REGISTER_QPS_MUTABLE_METRIC(response_iterate_qps_metric_, "response_iterate_qps");
    LOCAL_REGISTER_GAUGE_MUTABLE_METRIC(response_iterate_rt_metric_, "response_iterate_rt");

    LOCAL_REGISTER_GAUGE_MUTABLE_METRIC(response_iterate_count_metric_, "response_iterate_count");

    LOCAL_REGISTER_QPS_MUTABLE_METRIC(update_qps_metric_, "update_qps");
    LOCAL_REGISTER_QPS_MUTABLE_METRIC(error_update_target_qps_metric_, "error_update_target_qps");
    LOCAL_REGISTER_GAUGE_MUTABLE_METRIC(update_framework_rt_metric_, "update_framework_rt");

    LOCAL_REGISTER_GAUGE_MUTABLE_METRIC(ft_iterate_count_metric_, "ft_iterate_count");
    LOCAL_REGISTER_GAUGE_MUTABLE_METRIC(ft_input_token_length_metric_, "ft_input_token_length");
    LOCAL_REGISTER_GAUGE_MUTABLE_METRIC(ft_output_token_length_metric_, "ft_output_token_length");
    LOCAL_REGISTER_GAUGE_MUTABLE_METRIC(ft_pre_token_processor_rt_metric_, "ft_pre_token_processor_rt");
    LOCAL_REGISTER_GAUGE_MUTABLE_METRIC(ft_post_token_processor_rt_metric_, "ft_post_token_processor_rt");
    LOCAL_REGISTER_GAUGE_MUTABLE_METRIC(ft_num_beans_metric_, "ft_num_beans");

#undef LOCAL_REGISTER_GAUGE_MUTABLE_METRIC
#undef LOCAL_REGISTER_QPS_MUTABLE_METRIC

    inited = true;
    return true;
}

#define REPORT_QPS_METRIC_IF_INITED(metric, source, errorMessage)                                                      \
    do {                                                                                                               \
        if (!inited) {                                                                                                 \
            AUTIL_LOG(ERROR, errorMessage);                                                                            \
            return;                                                                                                    \
        }                                                                                                              \
        kmonitor::MetricsTags tags;                                                                                    \
        if (source.empty()) {                                                                                          \
            tags.AddTag("source", "unknown");                                                                          \
        } else {                                                                                                       \
            tags.AddTag("source", source);                                                                             \
        }                                                                                                              \
        metric->Report(&tags, 1);                                                                                      \
    } while (0)

void ApiServerMetricReporter::reportQpsMetric(const std::string& source) {
    REPORT_QPS_METRIC_IF_INITED(framework_qps_metric_, source, "report qps metric failed, not inited");
}

void ApiServerMetricReporter::reportCancelQpsMetric(const std::string& source) {
    REPORT_QPS_METRIC_IF_INITED(cancel_qps_metric_, source, "report cancel qps metric failed, not inited");
}

void ApiServerMetricReporter::reportSuccessQpsMetric(const std::string& source) {
    REPORT_QPS_METRIC_IF_INITED(success_qps_metric_, source, "report success qps metric failed, not inited");
}

#undef REPORT_QPS_METRIC_IF_INITED

void ApiServerMetricReporter::reportErrorQpsMetric(const std::string& source, int error_code) {
    if (!inited) {
        AUTIL_LOG(ERROR, "report error qps metric failed, not inited");
        return;
    }

    kmonitor::MetricsTags tags;
    tags.AddTag("error_code", std::to_string(error_code));
    if (source.empty()) {
        tags.AddTag("source", "unknown");
    } else {
        tags.AddTag("source", source);
    }

    framework_error_qps_metric_->Report(&tags, 1);
}

#define REPORT_METRIC_IF_INITED(metric, value, errorMessage)                                                           \
    do {                                                                                                               \
        if (!inited) {                                                                                                 \
            AUTIL_LOG(ERROR, errorMessage);                                                                            \
            return;                                                                                                    \
        }                                                                                                              \
        metric->Report(value);                                                                                         \
    } while (0)

void ApiServerMetricReporter::reportConflictQpsMetric() {
    REPORT_METRIC_IF_INITED(
        framework_concurrency_exception_qps_metric_, 1, "report conflict qps metric failed, not inited");
}

void ApiServerMetricReporter::reportResponseIterateQpsMetric() {
    REPORT_METRIC_IF_INITED(response_iterate_qps_metric_, 1, "report response iterate qps metric failed, not inited");
}

void ApiServerMetricReporter::reportResponseLatencyMs(double val) {
    REPORT_METRIC_IF_INITED(framework_rt_metric_, val, "report response latency ms failed, not inited");
}

void ApiServerMetricReporter::reportResponseFirstTokenLatencyMs(double val) {
    REPORT_METRIC_IF_INITED(
        response_first_token_rt_metric_, val, "report response first token latency ms failed, not inited");
}

void ApiServerMetricReporter::reportResponseIterateLatencyMs(double val) {
    REPORT_METRIC_IF_INITED(response_iterate_rt_metric_, val, "report response iterate latency ms failed, not inited");
}

void ApiServerMetricReporter::reportResponseIterateCountMetric(int32_t val) {
    REPORT_METRIC_IF_INITED(
        response_iterate_count_metric_, val, "report response iterate count metric failed, not inited");
}

void ApiServerMetricReporter::reportUpdateQpsMetric() {
    REPORT_METRIC_IF_INITED(update_qps_metric_, 1, "report update qps metric failed, not inited");
}

void ApiServerMetricReporter::reportErrorUpdateTargetQpsMetric() {
    REPORT_METRIC_IF_INITED(
        error_update_target_qps_metric_, 1, "report error update target qps metric failed, not inited");
}

void ApiServerMetricReporter::reportUpdateLatencyMs(double val) {
    REPORT_METRIC_IF_INITED(update_framework_rt_metric_, val, "report update latency ms failed, not inited");
}

void ApiServerMetricReporter::reportFTIterateCountMetric(double val) {
    REPORT_METRIC_IF_INITED(ft_iterate_count_metric_, val, "report ft iterate count metric failed, not inited");
}

void ApiServerMetricReporter::reportFTInputTokenLengthMetric(double val) {
    REPORT_METRIC_IF_INITED(
        ft_input_token_length_metric_, val, "report ft input token length metric failed, not inited");
}

void ApiServerMetricReporter::reportFTOutputTokenLengthMetric(double val) {
    REPORT_METRIC_IF_INITED(
        ft_output_token_length_metric_, val, "report ft output token length metric failed, not inited");
}

void ApiServerMetricReporter::reportFTPreTokenProcessorRtMetric(double val) {
    REPORT_METRIC_IF_INITED(
        ft_pre_token_processor_rt_metric_, val, "report ft pre token_processor rt metric failed, not inited");
}

void ApiServerMetricReporter::reportFTPostTokenProcessorRtMetric(double val) {
    REPORT_METRIC_IF_INITED(
        ft_post_token_processor_rt_metric_, val, "report ft post token_processor rt metric failed, not inited");
}

void ApiServerMetricReporter::reportFTNumBeansMetric(double val) {
    REPORT_METRIC_IF_INITED(ft_num_beans_metric_, val, "report ft num beans metric failed, not inited");
}

#undef REPORT_METRIC_IF_INITED

}  // namespace rtp_llm
