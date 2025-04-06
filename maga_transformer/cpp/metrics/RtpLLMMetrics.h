#pragma once

#include "autil/Log.h"
#include "kmonitor/client/MetricsReporter.h"
#include <chrono>
#include <cstdint>
#include <thread>
#include <unistd.h>

namespace kmonitor {
class MetricsTags;
class MutableMetric;
}  // namespace kmonitor

namespace rtp_llm {

class RpcMetricsCollector final {
public:
    // rpc server metrics
    bool    qps                             = false;
    bool    cancel_qps                      = false;
    bool    error_qps                       = false;
    int64_t onflight_request                = 0;
    int64_t total_rt_us                     = 0;

    // pd-sep prefill and decode metrics
    int     retry_times                     = 0;
    int     loading_cache_request           = 0;

    // pd-sep prefill metrics
    int64_t get_rpc_connection_rt_us        = 0;
    int64_t multimodal_process_rt_us        = 0;
    int64_t remote_allocate_resource_rt_us  = 0;
    int64_t enqueue_request_rt_us           = 0;
    int64_t remote_load_cache_start_rt_us   = 0;
    int64_t poll_local_output_rt_us         = 0;
    int64_t remote_load_cache_end_rt_us     = 0;
    int64_t remote_generate_rt_us           = 0;
    int64_t poll_remote_output_rt_us        = 0;

    // pd-sep decode stage metrics
    int64_t prepare_generate_context_rt_us  = 0;
    int64_t allocate_resource_rt_us         = 0;
    int64_t load_cache_from_prefill_rt_us   = 0;
    int64_t local_generate_rt_us            = 0;

    // for decode tp
    int64_t load_cache_min_rt_us            = 0;
    int64_t load_cache_max_rt_us            = 0;
    int64_t load_cache_polling_cost_us      = 0;
};

class RpcMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RpcMetricsCollector* collector);

public:
    kmonitor::MutableMetric* qps_metric                                 = nullptr;
    kmonitor::MutableMetric* cancel_qps_metric                          = nullptr;
    kmonitor::MutableMetric* error_qps_metric                           = nullptr;
    kmonitor::MutableMetric* onflight_request_metric                    = nullptr;
    kmonitor::MutableMetric* total_rt_us_metric                         = nullptr;

    kmonitor::MutableMetric* retry_times_metric                         = nullptr;
    kmonitor::MutableMetric* loading_cache_request_metric               = nullptr;

    kmonitor::MutableMetric* get_rpc_connection_rt_us_metric            = nullptr;
    kmonitor::MutableMetric* multimodal_process_rt_us_metric            = nullptr;
    kmonitor::MutableMetric* remote_allocate_resource_rt_us_metric      = nullptr;
    kmonitor::MutableMetric* enqueue_request_rt_us_metric               = nullptr;
    kmonitor::MutableMetric* remote_load_cache_start_rt_us_metric       = nullptr;
    kmonitor::MutableMetric* poll_local_output_rt_us_metric             = nullptr;
    kmonitor::MutableMetric* remote_load_cache_end_rt_us_metric         = nullptr;
    kmonitor::MutableMetric* remote_generate_rt_us_metric               = nullptr;
    kmonitor::MutableMetric* poll_remote_output_rt_us_metric            = nullptr;

    kmonitor::MutableMetric* prepare_generate_context_rt_us_metric      = nullptr;
    kmonitor::MutableMetric* allocate_resource_rt_us_metric             = nullptr;
    kmonitor::MutableMetric* load_cache_from_prefill_rt_us_metric       = nullptr;
    kmonitor::MutableMetric* local_generate_rt_us_metric                = nullptr;

    kmonitor::MutableMetric* load_cache_min_rt_us_metric                = nullptr;
    kmonitor::MutableMetric* load_cache_max_rt_us_metric                = nullptr;
    kmonitor::MutableMetric* load_cache_polling_cost_us_metric          = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMStreamMetricsCollector final {
public:
    bool qps        = false;
    bool cancel_qps = false;
    bool error_qps  = false;
    bool is_streaming_qps = false;
    bool not_streaming_qps = true;

    int64_t total_latency_us       = 0;
    int64_t first_token_latency_us = 0;
    int64_t wait_latency_us        = 0;
    int64_t pause_latency_us       = 0;
    int64_t iterate_count          = 0;
    int64_t reuse_length           = 0;
    int64_t input_token_length     = 0;
    int64_t output_token_length    = 0;
    // for timeout
    int64_t timeout_latency_us     = 0;

    int64_t query_batch_size       = 0;

    int64_t fallback_tokens        = 0;
    int64_t fallback_times         = 0;

    int32_t batch_with_prefill_times = 0;
    int32_t batch_with_prefill_len = 0;
};

class RtpLLMStreamMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMStreamMetricsCollector* collector);

public:
    kmonitor::MutableMetric* qps_metric                    = nullptr;
    kmonitor::MutableMetric* cancel_qps_metric             = nullptr;
    kmonitor::MutableMetric* error_qps_metric              = nullptr;
    kmonitor::MutableMetric* is_streaming_qps_metric       = nullptr;
    kmonitor::MutableMetric* not_streaming_qps_metric      = nullptr;

    kmonitor::MutableMetric* total_latency_us_metric       = nullptr;
    kmonitor::MutableMetric* first_token_latency_us_metric = nullptr;
    kmonitor::MutableMetric* wait_latency_us_metric        = nullptr;
    kmonitor::MutableMetric* pause_latency_us_metric       = nullptr;
    kmonitor::MutableMetric* iterate_count_metric          = nullptr;
    kmonitor::MutableMetric* reuse_length_metric           = nullptr;
    kmonitor::MutableMetric* input_token_length_metric     = nullptr;
    kmonitor::MutableMetric* output_token_length_metric    = nullptr;
    kmonitor::MutableMetric* query_batch_size_metric       = nullptr;

    kmonitor::MutableMetric* fallback_tokens_metric        = nullptr;
    kmonitor::MutableMetric* fallback_times_metric         = nullptr;
    kmonitor::MutableMetric* batch_with_prefill_times_metric = nullptr;
    kmonitor::MutableMetric* batch_with_prefill_len_metric = nullptr;
     
    kmonitor::MutableMetric* timeout_latency_us_metric       = nullptr;    

private:
    AUTIL_LOG_DECLARE();
};

// corresponding to python metrics
class RtpEmbeddingGlobalMetricsCollector final {
public:
    bool error = false;
    double total_latency_us = 0;
};

class RtpEmbeddingGlobalMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpEmbeddingGlobalMetricsCollector* collector);
public:
    kmonitor::MutableMetric* qps_metric              = nullptr;
    kmonitor::MutableMetric* success_qps_metric      = nullptr;
    kmonitor::MutableMetric* error_qps_metric        = nullptr;
    kmonitor::MutableMetric* total_latency_us_metric = nullptr;
private:
    AUTIL_LOG_DECLARE();
};

class RtpEmbeddingStreamMetricsCollector final {
public:
    int64_t total_latency_us       = 0;
    int64_t wait_latency_us        = 0;
    int64_t input_token_length     = 0;
};

class RtpEmbeddingStreamMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpEmbeddingStreamMetricsCollector* collector);

public:
    kmonitor::MutableMetric* total_latency_us_metric       = nullptr;
    kmonitor::MutableMetric* wait_latency_us_metric        = nullptr;
    kmonitor::MutableMetric* input_token_length_metric     = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMSchedulerMetricsCollector final {
public:
    int64_t wait_stream_size            = 0;
    int64_t running_stream_size         = 0;
    int64_t remote_running_stream_size  = 0;
    int64_t fallback_stream_size        = 0;
};

class RtpLLMSchedulerMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMSchedulerMetricsCollector* collector);

public:
    kmonitor::MutableMetric* wait_stream_size_metric            = nullptr;
    kmonitor::MutableMetric* running_stream_size_metric         = nullptr;
    kmonitor::MutableMetric* remote_running_stream_size_metric  = nullptr;
    kmonitor::MutableMetric* fallback_stream_size_metric        = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMEngineMetricsCollector final {
public:
    bool    update_lora_qps       = false;
    bool    error_update_lora_qps = false;
    int64_t step_latency_us       = 0;
};

class RtpLLMEngineMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMEngineMetricsCollector* collector);

public:
    kmonitor::MutableMetric* step_latency_us_metric       = nullptr;
    kmonitor::MutableMetric* update_lora_qps_metric       = nullptr;
    kmonitor::MutableMetric* error_update_lora_qps_metric = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMTokenPSMetricsCollector final {
public:
    void merge(const RtpLLMTokenPSMetricsCollector* collector) {
        if (collector) {
            context_tps += collector->context_tps;
            generate_tps += collector->generate_tps;
            total_tps += collector->total_tps;
        }
    }
public:
    int64_t context_tps  = 0;
    int64_t generate_tps = 0;
    int64_t total_tps    = 0;
};

class RtpLLMTokenPSMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMTokenPSMetricsCollector* collector);

public:
    kmonitor::MutableMetric* context_tps_metric  = nullptr;
    kmonitor::MutableMetric* generate_tps_metric = nullptr;
    kmonitor::MutableMetric* total_tps_metric    = nullptr;
private:
    AUTIL_LOG_DECLARE();
};

template<typename MetricsType, typename CollectType>
class MetricsLoopReporter {
public:
    explicit MetricsLoopReporter(const kmonitor::MetricsReporterPtr metrics_reporter, int interval_ms = 1000)
        :collector_(CollectType()),
         interval_ms_(interval_ms),
         metrics_reporter_(metrics_reporter) {
        if (metrics_reporter_) {
            metrics_reporter_thread_ = std::thread(&MetricsLoopReporter<MetricsType, CollectType>::reportLoop, this);
        }
    }

    ~MetricsLoopReporter() {
        stop_ = true;
        if (metrics_reporter_thread_.joinable()) {
            metrics_reporter_thread_.join();
        }
    }

    void report(const CollectType *collector) {
        std::lock_guard<std::mutex> lock(mutex_);
        collector_.merge(collector);
    }
private:
    void reportLoop() {
        while (metrics_reporter_ && !stop_) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                metrics_reporter_->report<MetricsType, CollectType>(nullptr, &collector_);
                collector_ = CollectType();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
        }
    }
private:
    std::mutex mutex_;
    bool stop_ = false;
    CollectType collector_;
    int interval_ms_ = 1000;
    std::thread metrics_reporter_thread_;
    kmonitor::MetricsReporterPtr metrics_reporter_ = nullptr;
};

class RtpLLMExecutorMetricsCollector final {
public:
    int64_t context_batch_size                      = 0;
    int64_t generate_batch_size                     = 0;
    int64_t context_batch_size_when_has_context     = 0;
    int64_t generate_batch_size_when_has_context    = 0;
    int64_t execute_token_size_when_has_context     = 0;
    int64_t max_seq_len_when_has_context            = 0;
    int64_t execute_token_size                      = 0;
    int64_t max_seq_len                             = 0;
    int64_t gather_model_input_us                   = 0;
    int64_t tp_sync_input_us                        = 0;
    int64_t model_forward_us                        = 0;
    int64_t sample_input_us                         = 0;
    int64_t dispatch_output_us                      = 0;
};

class RtpLLMExecutorMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMExecutorMetricsCollector* collector);

public:
    kmonitor::MutableMetric* context_batch_size_metric                      = nullptr;
    kmonitor::MutableMetric* generate_batch_size_metric                     = nullptr;
    kmonitor::MutableMetric* context_batch_size_when_has_context_metric     = nullptr;
    kmonitor::MutableMetric* generate_batch_size_when_has_context_metric    = nullptr;
    kmonitor::MutableMetric* execute_token_size_when_has_context_metric     = nullptr;
    kmonitor::MutableMetric* max_seq_len_when_has_context_metric            = nullptr;
    kmonitor::MutableMetric* execute_token_size_metric                      = nullptr;
    kmonitor::MutableMetric* max_seq_len_metric                             = nullptr;

    kmonitor::MutableMetric* gather_model_input_us_metric                   = nullptr;
    kmonitor::MutableMetric* tp_sync_input_us_metric                        = nullptr;
    kmonitor::MutableMetric* model_forward_us_metric                        = nullptr;
    kmonitor::MutableMetric* sample_input_us_metric                         = nullptr;
    kmonitor::MutableMetric* dispatch_output_us_metric                      = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMCacheMetricsCollector final {
public:
    int64_t kv_cache_item_num = 0;
    int64_t kv_cache_free_blocks = 0;
    int64_t kv_cache_available_blocks = 0;
    int64_t kv_cache_left_seq = 0;
    float kv_cache_used_ratio = 0;
    int64_t mr_cost_time_ms = 0;
};

class RtpLLMCacheMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMCacheMetricsCollector* collector);

public:
    kmonitor::MutableMetric* kv_cache_item_num_metric = nullptr;
    kmonitor::MutableMetric* kv_cache_free_blocks_metric = nullptr;
    kmonitor::MutableMetric* kv_cache_available_blocks_metric = nullptr;
    kmonitor::MutableMetric* kv_cache_left_seq_metric = nullptr;
    kmonitor::MutableMetric* kv_cache_used_ratio_metric = nullptr;
    kmonitor::MutableMetric* mr_cost_time_ms_metric = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMCacheReuseMetricsCollector final {
public:
    int64_t match_cost_time_us = 0;
    int64_t kv_cache_reuse_length = 0;
};

class RtpLLMCacheReuseMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMCacheReuseMetricsCollector* collector);

public:
    kmonitor::MutableMetric* match_cost_time_us = nullptr;
    kmonitor::MutableMetric* kv_cache_reuse_length = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMKernelMetricsCollector final {
public:
    float kernel_exec_time = 0;
};

class RtpLLMKernelMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMKernelMetricsCollector* collector);

public:
    kmonitor::MutableMetric* kernel_exec_time_metric = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMSpeculativeEngineMetricsCollector final {
public:
    int64_t step_latency_us = 0;
    int64_t propose_step_latency_us = 0;
    int64_t score_step_latency_us = 0;
    int64_t speculative_sampler_latency_us = 0;
    int64_t updater_step_latency_us = 0;
    int64_t total_propose_token_num = 0;
    int64_t total_accepted_token_num = 0;
};

class RtpLLMSpeculativeEngineMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMSpeculativeEngineMetricsCollector* collector);

public:
    kmonitor::MutableMetric* step_latency_us_metric = nullptr;
    kmonitor::MutableMetric* propose_step_latency_us_metric = nullptr;
    kmonitor::MutableMetric* score_step_latency_us_metric = nullptr;
    kmonitor::MutableMetric* speculative_sampler_latency_us_metric = nullptr;
    kmonitor::MutableMetric* updater_step_latency_us_metric = nullptr;
    kmonitor::MutableMetric* total_propose_token_num_metric = nullptr;
    kmonitor::MutableMetric* total_accepted_token_num_metric = nullptr;
private:
    AUTIL_LOG_DECLARE();
};

bool initKmonitorFactory();
void stopKmonitorFactory();

kmonitor::MetricsTags getHippoTags();

}  // namespace rtp_llm
