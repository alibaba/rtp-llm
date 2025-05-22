#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "kmonitor/client/KMonitorFactory.h"
#include "rtp_llm/cpp/metrics/KmonParam.h"

namespace rtp_llm {

AUTIL_LOG_SETUP(rtp_llm, RpcMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpLLMStreamMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpEmbeddingGlobalMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpEmbeddingStreamMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpLLMSchedulerMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpLLMCacheMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpLLMCacheReuseMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpLLMExecutorMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpLLMTokenPSMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpLLMEngineMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpLLMKernelMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpLLMSpeculativeEngineMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpLLmEplbMetrics);

#define REPORT_QPS(name)                                                                                               \
    if (collector->name) {                                                                                             \
        REPORT_MUTABLE_QPS(name##_metric);                                                                             \
    }

#define REPORT_GAUGE(name)                                                                                             \
    if (collector->name) {                                                                                             \
        REPORT_MUTABLE_METRIC(name##_metric, collector->name);                                                         \
    }

bool RpcMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_QPS_MUTABLE_METRIC(qps_metric, "rtp_llm_rpc_qps");
    REGISTER_QPS_MUTABLE_METRIC(error_qps_metric, "rtp_llm_rpc_error_qps");
    REGISTER_QPS_MUTABLE_METRIC(cancel_qps_metric, "rtp_llm_rpc_cancel_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(onflight_request_metric, "rtp_llm_rpc_onflight_request");
    REGISTER_GAUGE_MUTABLE_METRIC(total_rt_us_metric, "rtp_llm_rpc_total_rt_us");

    REGISTER_GAUGE_MUTABLE_METRIC(retry_times_metric, "rtp_llm_rpc_retry_times");
    REGISTER_GAUGE_MUTABLE_METRIC(loading_cache_request_metric, "rtp_llm_rpc_loading_cache_request");

    REGISTER_GAUGE_MUTABLE_METRIC(get_rpc_connection_rt_us_metric, "rtp_llm_rpc_get_rpc_connection_rt_us");
    REGISTER_GAUGE_MUTABLE_METRIC(multimodal_process_rt_us_metric, "rtp_llm_rpc_multimodal_process_rt_us");
    REGISTER_GAUGE_MUTABLE_METRIC(remote_allocate_resource_rt_us_metric, "rtp_llm_rpc_remote_allocate_resource_rt_us");
    REGISTER_GAUGE_MUTABLE_METRIC(enqueue_request_rt_us_metric, "rtp_llm_rpc_enqueue_request_rt_us");
    REGISTER_GAUGE_MUTABLE_METRIC(remote_load_cache_start_rt_us_metric, "rtp_llm_rpc_remote_load_cache_start_rt_us");
    REGISTER_GAUGE_MUTABLE_METRIC(poll_local_output_rt_us_metric, "rtp_llm_rpc_poll_local_output_rt_us");
    REGISTER_GAUGE_MUTABLE_METRIC(remote_load_cache_end_rt_us_metric, "rtp_llm_rpc_remote_load_cache_end_rt_us");
    REGISTER_GAUGE_MUTABLE_METRIC(remote_generate_rt_us_metric, "rtp_llm_rpc_remote_generate_rt_us");
    REGISTER_GAUGE_MUTABLE_METRIC(poll_remote_output_rt_us_metric, "rtp_llm_rpc_poll_remote_output_rt_us");

    REGISTER_GAUGE_MUTABLE_METRIC(prepare_generate_context_rt_us_metric, "rtp_llm_rpc_prepare_generate_context_rt_us");
    REGISTER_GAUGE_MUTABLE_METRIC(allocate_resource_rt_us_metric, "rtp_llm_rpc_allocate_resource_rt_us");
    REGISTER_GAUGE_MUTABLE_METRIC(load_cache_from_prefill_rt_us_metric, "rtp_llm_rpc_load_cache_from_prefill_rt_us");
    REGISTER_GAUGE_MUTABLE_METRIC(local_generate_rt_us_metric, "rtp_llm_rpc_local_generate_rt_us");

    REGISTER_GAUGE_MUTABLE_METRIC(load_cache_min_rt_us_metric, "rtp_llm_rpc_load_cache_min_rt_us");
    REGISTER_GAUGE_MUTABLE_METRIC(load_cache_max_rt_us_metric, "rtp_llm_rpc_load_cache_max_rt_us");
    REGISTER_GAUGE_MUTABLE_METRIC(load_cache_polling_cost_us_metric, "rtp_llm_rpc_load_cache_polling_cost_us");

    return true;
}

void RpcMetrics::report(const kmonitor::MetricsTags* tags, RpcMetricsCollector* collector) {
    REPORT_QPS(qps);
    REPORT_QPS(cancel_qps);
    REPORT_QPS(error_qps);
    REPORT_GAUGE(onflight_request);
    REPORT_GAUGE(total_rt_us);

    REPORT_GAUGE(retry_times);
    REPORT_GAUGE(loading_cache_request);

    REPORT_GAUGE(get_rpc_connection_rt_us);
    REPORT_GAUGE(multimodal_process_rt_us);
    REPORT_GAUGE(remote_allocate_resource_rt_us);
    REPORT_GAUGE(enqueue_request_rt_us);
    REPORT_GAUGE(remote_load_cache_start_rt_us);
    REPORT_GAUGE(poll_local_output_rt_us);
    REPORT_GAUGE(remote_load_cache_end_rt_us);
    REPORT_GAUGE(remote_generate_rt_us);
    REPORT_GAUGE(poll_remote_output_rt_us);

    REPORT_GAUGE(prepare_generate_context_rt_us);
    REPORT_GAUGE(allocate_resource_rt_us);
    REPORT_GAUGE(load_cache_from_prefill_rt_us);
    REPORT_GAUGE(local_generate_rt_us);

    REPORT_GAUGE(load_cache_min_rt_us);
    REPORT_GAUGE(load_cache_max_rt_us);
    REPORT_GAUGE(load_cache_polling_cost_us);
}

bool RtpLLMStreamMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_QPS_MUTABLE_METRIC(qps_metric, "rtp_llm_framework_qps");
    REGISTER_QPS_MUTABLE_METRIC(error_qps_metric, "rtp_llm_framework_error_qps");
    REGISTER_QPS_MUTABLE_METRIC(cancel_qps_metric, "rtp_llm_cancel_qps");
    REGISTER_QPS_MUTABLE_METRIC(is_streaming_qps_metric, "rtp_llm_is_streaming_qps");
    REGISTER_QPS_MUTABLE_METRIC(not_streaming_qps_metric, "rtp_llm_not_streaming_qps");

    REGISTER_GAUGE_MUTABLE_METRIC(total_latency_us_metric, "rtp_llm_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(first_token_latency_us_metric, "rtp_llm_first_token_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(wait_latency_us_metric, "rtp_llm_wait_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(pause_latency_us_metric, "rtp_llm_pause_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(iterate_count_metric, "rtp_llm_iterate_count");
    REGISTER_GAUGE_MUTABLE_METRIC(reuse_length_metric, "rtp_llm_reuse_length");
    REGISTER_GAUGE_MUTABLE_METRIC(input_token_length_metric, "rtp_llm_input_token_length");
    REGISTER_GAUGE_MUTABLE_METRIC(output_token_length_metric, "rtp_llm_output_token_length");
    REGISTER_GAUGE_MUTABLE_METRIC(timeout_latency_us_metric, "rtp_llm_timeout_lantency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(query_batch_size_metric, "rtp_llm_query_batch_size");

    REGISTER_GAUGE_MUTABLE_METRIC(fallback_tokens_metric, "rtp_llm_fallback_tokens");
    REGISTER_GAUGE_MUTABLE_METRIC(fallback_times_metric, "rtp_llm_fallback_times");
    REGISTER_GAUGE_MUTABLE_METRIC(batch_with_prefill_times_metric, "rtp_llm_batch_with_prefill_times");
    REGISTER_GAUGE_MUTABLE_METRIC(batch_with_prefill_len_metric, "rtp_llm_batch_with_prefill_len");

    REGISTER_GAUGE_MUTABLE_METRIC(malloc_failed_times_metric, "rtp_llm_malloc_failed_times");

    return true;
}

void RtpLLMStreamMetrics::report(const kmonitor::MetricsTags* tags, RtpLLMStreamMetricsCollector* collector) {
    REPORT_QPS(qps);
    REPORT_QPS(cancel_qps);
    REPORT_QPS(error_qps);
    REPORT_QPS(is_streaming_qps);
    REPORT_QPS(not_streaming_qps);

    REPORT_GAUGE(total_latency_us);
    REPORT_GAUGE(first_token_latency_us);
    REPORT_GAUGE(wait_latency_us);
    REPORT_GAUGE(pause_latency_us);
    REPORT_GAUGE(iterate_count);
    REPORT_GAUGE(reuse_length);
    REPORT_GAUGE(input_token_length);
    REPORT_GAUGE(output_token_length);
    REPORT_GAUGE(timeout_latency_us);
    REPORT_GAUGE(query_batch_size);

    REPORT_GAUGE(fallback_tokens);
    REPORT_GAUGE(fallback_times);
    REPORT_GAUGE(batch_with_prefill_times);
    REPORT_GAUGE(batch_with_prefill_len);

    REPORT_GAUGE(malloc_failed_times);
}

// for rpc request
bool RtpEmbeddingGlobalMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(total_latency_us_metric, "py_rtp_framework_rt");
    REGISTER_QPS_MUTABLE_METRIC(error_qps_metric, "py_rtp_framework_error_qps");
    REGISTER_QPS_MUTABLE_METRIC(qps_metric, "py_rtp_framework_qps");
    REGISTER_QPS_MUTABLE_METRIC(success_qps_metric, "py_rtp_success_qps_metric");
    return true;
}

void RtpEmbeddingGlobalMetrics::report(const kmonitor::MetricsTags* tags, RtpEmbeddingGlobalMetricsCollector* collector) {
    REPORT_MUTABLE_QPS(qps_metric);
    if (collector->error) {
        REPORT_MUTABLE_QPS(error_qps_metric);
    } else {
        REPORT_MUTABLE_QPS(success_qps_metric);
        REPORT_GAUGE(total_latency_us);
    }
}

bool RtpEmbeddingStreamMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(total_latency_us_metric, "rtp_llm_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(wait_latency_us_metric, "rtp_llm_wait_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(input_token_length_metric, "rtp_llm_input_token_length");
    return true;
}

void RtpEmbeddingStreamMetrics::report(const kmonitor::MetricsTags* tags, RtpEmbeddingStreamMetricsCollector* collector) {
    REPORT_GAUGE(total_latency_us);
    REPORT_GAUGE(wait_latency_us);
    REPORT_GAUGE(input_token_length);
}

bool RtpLLMSchedulerMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(wait_stream_size_metric, "rtp_llm_wait_stream_size");
    REGISTER_GAUGE_MUTABLE_METRIC(running_stream_size_metric, "rtp_llm_running_stream_size");
    REGISTER_GAUGE_MUTABLE_METRIC(remote_running_stream_size_metric, "rtp_llm_remote_running_stream_size");
    REGISTER_GAUGE_MUTABLE_METRIC(fallback_stream_size_metric, "rtp_llm_fallback_stream_size");
    return true;
}

void RtpLLMSchedulerMetrics::report(const kmonitor::MetricsTags* tags, RtpLLMSchedulerMetricsCollector* collector) {
    REPORT_MUTABLE_METRIC(wait_stream_size_metric, collector->wait_stream_size);
    REPORT_MUTABLE_METRIC(running_stream_size_metric, collector->running_stream_size);
    REPORT_MUTABLE_METRIC(remote_running_stream_size_metric, collector->remote_running_stream_size);
    REPORT_MUTABLE_METRIC(fallback_stream_size_metric, collector->fallback_stream_size);
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
    REGISTER_GAUGE_MUTABLE_METRIC(context_batch_size_when_has_context_metric, "rtp_llm_context_batch_size_when_has_context");
    REGISTER_GAUGE_MUTABLE_METRIC(generate_batch_size_when_has_context_metric, "rtp_llm_generate_batch_size_when_has_context");
    REGISTER_GAUGE_MUTABLE_METRIC(execute_token_size_when_has_context_metric, "rtp_llm_execute_token_size_when_has_context");
    REGISTER_GAUGE_MUTABLE_METRIC(max_seq_len_when_has_context_metric, "rtp_llm_max_seq_len_when_has_context");
    REGISTER_GAUGE_MUTABLE_METRIC(execute_token_size_metric, "rtp_llm_execute_token_size");
    REGISTER_GAUGE_MUTABLE_METRIC(max_seq_len_metric, "rtp_llm_max_seq_len");

    REGISTER_GAUGE_MUTABLE_METRIC(gather_model_input_us_metric, "rtp_llm_gather_model_input_us");
    REGISTER_GAUGE_MUTABLE_METRIC(tp_sync_input_us_metric, "rtp_llm_tp_sync_input_us");
    REGISTER_GAUGE_MUTABLE_METRIC(model_forward_us_metric, "rtp_llm_model_forward_us");
    REGISTER_GAUGE_MUTABLE_METRIC(sample_input_us_metric, "rtp_llm_sample_input_us");
    REGISTER_GAUGE_MUTABLE_METRIC(dispatch_output_us_metric, "rtp_llm_dispatch_output_us_metric");

    REGISTER_GAUGE_MUTABLE_METRIC(eplb_step_latency_us_metric, "rtp_llm_eplb_step_latency_us");

    return true;
}

void RtpLLMExecutorMetrics::report(const kmonitor::MetricsTags* tags, RtpLLMExecutorMetricsCollector* collector) {
    REPORT_GAUGE(context_batch_size);
    REPORT_GAUGE(generate_batch_size);
    if (collector->context_batch_size != 0) {
        REPORT_GAUGE(context_batch_size_when_has_context);
        REPORT_GAUGE(generate_batch_size_when_has_context);
        REPORT_GAUGE(execute_token_size_when_has_context);
        REPORT_GAUGE(max_seq_len_when_has_context);
    }
    REPORT_GAUGE(execute_token_size);
    REPORT_GAUGE(max_seq_len);

    REPORT_GAUGE(gather_model_input_us);
    REPORT_GAUGE(tp_sync_input_us);
    REPORT_GAUGE(model_forward_us);
    REPORT_GAUGE(sample_input_us);
    REPORT_GAUGE(dispatch_output_us);
    REPORT_GAUGE(eplb_step_latency_us);
}

bool RtpLLMSpeculativeEngineMetrics::init(kmonitor::MetricsGroupManager *manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(step_latency_us_metric, "rtp_llm_sp_step_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(propose_step_latency_us_metric, "rtp_llm_sp_propose_step_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(score_step_latency_us_metric, "rtp_llm_sp_score_step_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(speculative_sampler_latency_us_metric, "rtp_llm_sp_speculative_sampler_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(updater_step_latency_us_metric, "rtp_llm_sp_updater_step_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(total_propose_token_num_metric, "rtp_llm_sp_total_propose_token_num");
    REGISTER_GAUGE_MUTABLE_METRIC(total_accepted_token_num_metric, "rtp_llm_sp_total_accepted_token_num");
    return true;
}


void RtpLLMSpeculativeEngineMetrics::report(const kmonitor::MetricsTags* tags, RtpLLMSpeculativeEngineMetricsCollector* collector) {
    REPORT_MUTABLE_METRIC(step_latency_us_metric, collector->step_latency_us);
    REPORT_MUTABLE_METRIC(propose_step_latency_us_metric, collector->propose_step_latency_us);
    REPORT_MUTABLE_METRIC(score_step_latency_us_metric, collector->score_step_latency_us);
    REPORT_MUTABLE_METRIC(speculative_sampler_latency_us_metric, collector->speculative_sampler_latency_us);
    REPORT_MUTABLE_METRIC(updater_step_latency_us_metric, collector->updater_step_latency_us);
    REPORT_MUTABLE_METRIC(total_propose_token_num_metric, collector->total_propose_token_num);
    REPORT_MUTABLE_METRIC(total_accepted_token_num_metric, collector->total_accepted_token_num);

}

bool RtpLLMTokenPSMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(context_tps_metric, "rtp_llm_context_tps");
    REGISTER_GAUGE_MUTABLE_METRIC(generate_tps_metric, "rtp_llm_generate_tps");
    REGISTER_GAUGE_MUTABLE_METRIC(total_tps_metric, "rtp_llm_total_tps");
    return true;
}

void RtpLLMTokenPSMetrics::report(const kmonitor::MetricsTags* tags, RtpLLMTokenPSMetricsCollector* collector) {
    REPORT_MUTABLE_METRIC(context_tps_metric, collector->context_tps);
    REPORT_MUTABLE_METRIC(generate_tps_metric, collector->generate_tps);
    REPORT_MUTABLE_METRIC(total_tps_metric, collector->total_tps);
}

bool RtpLLMCacheMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(kv_cache_item_num_metric, "rtp_llm_kv_cache_item_num");
    REGISTER_GAUGE_MUTABLE_METRIC(kv_cache_free_blocks_metric, "rtp_llm_kv_cache_free_blocks");
    REGISTER_GAUGE_MUTABLE_METRIC(kv_cache_available_blocks_metric, "rtp_llm_kv_cache_available_blocks");
    REGISTER_GAUGE_MUTABLE_METRIC(kv_cache_left_seq_metric, "rtp_llm_kv_cache_left_seq");
    REGISTER_GAUGE_MUTABLE_METRIC(kv_cache_used_ratio_metric, "rtp_llm_kv_cache_used_ratio");
    REGISTER_GAUGE_MUTABLE_METRIC(mr_cost_time_ms_metric, "rtp_llm_mr_cost_time_ms");

    return true;
}

void RtpLLMCacheMetrics::report(const kmonitor::MetricsTags* tags, RtpLLMCacheMetricsCollector* collector) {
    REPORT_MUTABLE_METRIC(kv_cache_item_num_metric, collector->kv_cache_item_num);
    REPORT_MUTABLE_METRIC(kv_cache_free_blocks_metric, collector->kv_cache_free_blocks);
    REPORT_MUTABLE_METRIC(kv_cache_available_blocks_metric, collector->kv_cache_available_blocks);
    REPORT_MUTABLE_METRIC(kv_cache_left_seq_metric, collector->kv_cache_left_seq);
    REPORT_MUTABLE_METRIC(kv_cache_used_ratio_metric, collector->kv_cache_used_ratio);
    REPORT_MUTABLE_METRIC(mr_cost_time_ms_metric, collector->mr_cost_time_ms);
}

bool RtpLLMCacheReuseMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(match_cost_time_us, "rtp_llm_match_cost_time_us");
    REGISTER_GAUGE_MUTABLE_METRIC(kv_cache_reuse_length, "rtp_llm_kv_cache_reuse_length");
    return true;
}

void RtpLLMCacheReuseMetrics::report(const kmonitor::MetricsTags* tags, RtpLLMCacheReuseMetricsCollector* collector) {
    REPORT_MUTABLE_METRIC(match_cost_time_us, collector->match_cost_time_us);
    REPORT_MUTABLE_METRIC(kv_cache_reuse_length, collector->kv_cache_reuse_length);
}

bool RtpLLMKernelMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(kernel_exec_time_metric, "rtp_llm_kenrel_exec_time");
    return true;
}

void RtpLLMKernelMetrics::report(const kmonitor::MetricsTags* tags, RtpLLMKernelMetricsCollector* collector) {
    REPORT_MUTABLE_METRIC(kernel_exec_time_metric, collector->kernel_exec_time);
}

bool RtpLLmEplbMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_QPS_MUTABLE_METRIC(update_weights_qps_metric, "rtp_llm_update_weights_qps");
    REGISTER_QPS_MUTABLE_METRIC(update_layer_weights_qps_metric, "rtp_llm_update_layer_weights_qps");
    REGISTER_GAUGE_MUTABLE_METRIC(update_weights_latency_ms_metric, "rtp_llm_update_weights_latency_ms");
    REGISTER_GAUGE_MUTABLE_METRIC(gpu_loads_metric, "rtp_llm_gpu_loads");
    return true;
}

void RtpLLmEplbMetrics::report(const kmonitor::MetricsTags* tags, RtpLLmEplbMetricsCollector* collector) {
    // ep stats metrics
    int num_layer = collector->gpu_loads.size();
    auto ep_tag = kmonitor::MetricsTags("ep_rank", std::to_string(collector->ep_rank));
    tags->MergeTags(&ep_tag);
    for (int i = 0; i < num_layer; ++i) {
        auto layer_tag = kmonitor::MetricsTags("layer", std::to_string(i));
        ep_tag.MergeTags(&layer_tag);
        if (gpu_loads_metric) {
            gpu_loads_metric->Report(&layer_tag, collector->gpu_loads[i]);
        }
    }

    // update weights metrics
    if (collector->update_weights_qps) {
        REPORT_MUTABLE_QPS(update_weights_qps_metric);
        REPORT_MUTABLE_METRIC(update_weights_latency_ms_metric, collector->update_weights_latency_ms);

        // report layer qps
        auto layer_tag = kmonitor::MetricsTags("layer", std::to_string(collector->update_layer_id));
        tags->MergeTags(&layer_tag);
        if (update_layer_weights_qps_metric) {
            update_layer_weights_qps_metric->Report(&layer_tag, 1);
        }
        collector->update_weights_qps = false;
    }
}

#undef REPORT_QPS
#undef REPORT_GAUGE

bool initKmonitorFactory() {
    KmonParam param;
    param.init();
    if (!param.kmonitorMetricsReporterCacheLimit.empty()) {
        size_t limit = 0;
        if (autil::StringUtil::fromString<size_t>(param.kmonitorMetricsReporterCacheLimit, limit) || limit > 0) {
            kmonitor::MetricsReporter::setMetricsReporterCacheLimit(limit);
            RTP_LLM_LOG_INFO("set metrics reporter cache limit [%lu].", limit);
        }
    }

    if (param.kmonitorNormalSamplePeriod > 0) {
        RTP_LLM_LOG_INFO("set kmonitor normal sample period [%d] seconds.", param.kmonitorNormalSamplePeriod);
        kmonitor::MetricLevelConfig config;
        config.period[kmonitor::NORMAL] = (unsigned int)param.kmonitorNormalSamplePeriod;
        kmonitor::MetricLevelManager::SetGlobalLevelConfig(config);
    }

    kmonitor::MetricsConfig metricsConfig;
    metricsConfig.set_tenant_name(param.kmonitorTenant);
    metricsConfig.set_service_name(param.kmonitorServiceName);
    std::string sink_address = param.kmonitorSinkAddress;
    if (!param.kmonitorPort.empty()) {
        sink_address += ":" + param.kmonitorPort;
    }
    metricsConfig.set_sink_address(sink_address.c_str());
    metricsConfig.set_enable_log_file_sink(param.kmonitorEnableLogFileSink);
    //metricsConfig.set_enable_prometheus_sink(param.kmonitorEnablePrometheusSink);
    metricsConfig.set_manually_mode(param.kmonitorManuallyMode);
    metricsConfig.set_inited(true);
    metricsConfig.AddGlobalTag("hippo_slave_ip", param.hippoSlaveIp);
    for (auto &pair : param.kmonitorTags) {
        metricsConfig.AddGlobalTag(pair.first, pair.second);
    }
    if (!kmonitor::KMonitorFactory::Init(metricsConfig)) {
        RTP_LLM_LOG_ERROR("init kmonitor factory failed with");
        return false;
    }

    // registerBuildInMetrics to refresh sg_buildin_kmonitor for KMonitorWorker::Start
    kmonitor::KMonitorFactory::registerBuildInMetrics(nullptr, param.kmonitorMetricsPrefix);
    RTP_LLM_LOG_INFO("KMonitorFactory::registerBuildInMetrics() finished");

    kmonitor::KMonitorFactory::Start();
    RTP_LLM_LOG_INFO("KMonitorFactory::Start() finished");
    return true;
}

void stopKmonitorFactory() {
    kmonitor::KMonitorFactory::Shutdown();
}

kmonitor::MetricsTags getHippoTags() {
    auto hippo_tags = kmonitor::MetricsTags();
    if (std::getenv("HIPPO_ROLE")) {
        auto host_ip = autil::EnvUtil::getEnv("HIPPO_SLAVE_IP", "");
        hippo_tags.AddTag("host_ip", host_ip);
        hippo_tags.AddTag("container_ip", autil::EnvUtil::getEnv("RequestedIP", host_ip));
        hippo_tags.AddTag("hippo_role", autil::EnvUtil::getEnv("HIPPO_ROLE", ""));
        hippo_tags.AddTag("hippo_app", autil::EnvUtil::getEnv("HIPPO_APP", ""));
        hippo_tags.AddTag("hippo_group", autil::EnvUtil::getEnv("HIPPO_SERVICE_NAME", ""));
    }
    return hippo_tags;
}

}  // namespace rtp_llm
