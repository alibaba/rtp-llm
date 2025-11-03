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
    bool    qps              = false;
    bool    cancel_qps       = false;
    bool    error_qps        = false;
    int64_t onflight_request = 0;
    int64_t total_rt_us      = 0;

    // pd-sep prefill and decode metrics
    int     retry_times           = 0;
    int64_t retry_cost_time_ms    = 0;
    int     loading_cache_request = 0;

    // pd-sep prefill metrics
    int64_t get_rpc_connection_rt_us       = 0;
    int64_t multimodal_process_rt_us       = 0;
    int64_t remote_allocate_resource_rt_us = 0;
    int64_t enqueue_request_rt_us          = 0;
    int64_t remote_load_cache_start_rt_us  = 0;
    int64_t poll_local_output_rt_us        = 0;
    int64_t remote_load_cache_end_rt_us    = 0;
    int64_t remote_generate_rt_us          = 0;
    int64_t poll_remote_output_rt_us       = 0;

    // pd-sep decode stage metrics
    int64_t prepare_generate_context_rt_us = 0;
    int64_t allocate_resource_rt_us        = 0;
    int64_t load_cache_from_prefill_rt_us  = 0;
    int64_t local_generate_rt_us           = 0;

    // pd-sep prefill metrics(decode entrance)
    int64_t notify_store_cache_rt_us   = 0;
    int64_t generate_first_token_rt_us = 0;
    int64_t wait_store_cache_rt_us     = 0;
    int64_t min_response_done_time_us  = 1lu << 60;
    int64_t max_response_done_time_us  = 0;

    // for decode tp
    int64_t load_cache_min_rt_us       = 0;
    int64_t load_cache_max_rt_us       = 0;
    int64_t load_cache_polling_cost_us = 0;
};

class RpcMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RpcMetricsCollector* collector);

public:
    kmonitor::MutableMetric* qps_metric              = nullptr;
    kmonitor::MutableMetric* cancel_qps_metric       = nullptr;
    kmonitor::MutableMetric* error_qps_metric        = nullptr;
    kmonitor::MutableMetric* onflight_request_metric = nullptr;
    kmonitor::MutableMetric* total_rt_us_metric      = nullptr;

    kmonitor::MutableMetric* retry_times_metric           = nullptr;
    kmonitor::MutableMetric* retry_cost_time_ms_metric    = nullptr;
    kmonitor::MutableMetric* loading_cache_request_metric = nullptr;

    kmonitor::MutableMetric* get_rpc_connection_rt_us_metric       = nullptr;
    kmonitor::MutableMetric* multimodal_process_rt_us_metric       = nullptr;
    kmonitor::MutableMetric* remote_allocate_resource_rt_us_metric = nullptr;
    kmonitor::MutableMetric* enqueue_request_rt_us_metric          = nullptr;
    kmonitor::MutableMetric* remote_load_cache_start_rt_us_metric  = nullptr;
    kmonitor::MutableMetric* poll_local_output_rt_us_metric        = nullptr;
    kmonitor::MutableMetric* remote_load_cache_end_rt_us_metric    = nullptr;
    kmonitor::MutableMetric* remote_generate_rt_us_metric          = nullptr;
    kmonitor::MutableMetric* poll_remote_output_rt_us_metric       = nullptr;

    kmonitor::MutableMetric* prepare_generate_context_rt_us_metric = nullptr;
    kmonitor::MutableMetric* allocate_resource_rt_us_metric        = nullptr;
    kmonitor::MutableMetric* load_cache_from_prefill_rt_us_metric  = nullptr;
    kmonitor::MutableMetric* local_generate_rt_us_metric           = nullptr;

    kmonitor::MutableMetric* load_cache_min_rt_us_metric       = nullptr;
    kmonitor::MutableMetric* load_cache_max_rt_us_metric       = nullptr;
    kmonitor::MutableMetric* load_cache_polling_cost_us_metric = nullptr;

    kmonitor::MutableMetric* notify_store_cache_rt_us_metric   = nullptr;
    kmonitor::MutableMetric* generate_first_token_rt_us_metric = nullptr;
    kmonitor::MutableMetric* wait_store_cache_rt_us_metric     = nullptr;
    kmonitor::MutableMetric* min_response_done_time_us_metric  = nullptr;
    kmonitor::MutableMetric* max_response_done_time_us_metric  = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RpcCacheStatusMetricsCollector final {
public:
    bool    qps         = false;
    int64_t total_rt_us = 0;
};

class RpcWorkerStatusMetricsCollector final {
public:
    bool    qps         = false;
    int64_t total_rt_us = 0;
};

class RpcWorkerStatusMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RpcWorkerStatusMetricsCollector* collector);

public:
    kmonitor::MutableMetric* qps_metric         = nullptr;
    kmonitor::MutableMetric* total_rt_us_metric = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RpcCacheStatusMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RpcCacheStatusMetricsCollector* collector);

public:
    kmonitor::MutableMetric* qps_metric         = nullptr;
    kmonitor::MutableMetric* total_rt_us_metric = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMStreamMetricsCollector final {
public:
    bool qps               = false;
    bool cancel_qps        = false;
    bool error_qps         = false;
    bool is_streaming_qps  = false;
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
    int64_t timeout_latency_us = 0;

    int64_t query_batch_size = 0;

    int32_t batch_with_prefill_times = 0;
    int32_t batch_with_prefill_len   = 0;
    int32_t malloc_failed_times      = 0;
};

class RtpLLMStreamMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMStreamMetricsCollector* collector);

public:
    kmonitor::MutableMetric* qps_metric               = nullptr;
    kmonitor::MutableMetric* cancel_qps_metric        = nullptr;
    kmonitor::MutableMetric* error_qps_metric         = nullptr;
    kmonitor::MutableMetric* is_streaming_qps_metric  = nullptr;
    kmonitor::MutableMetric* not_streaming_qps_metric = nullptr;

    kmonitor::MutableMetric* total_latency_us_metric       = nullptr;
    kmonitor::MutableMetric* first_token_latency_us_metric = nullptr;
    kmonitor::MutableMetric* wait_latency_us_metric        = nullptr;
    kmonitor::MutableMetric* pause_latency_us_metric       = nullptr;
    kmonitor::MutableMetric* iterate_count_metric          = nullptr;
    kmonitor::MutableMetric* reuse_length_metric           = nullptr;
    kmonitor::MutableMetric* input_token_length_metric     = nullptr;
    kmonitor::MutableMetric* output_token_length_metric    = nullptr;
    kmonitor::MutableMetric* query_batch_size_metric       = nullptr;

    kmonitor::MutableMetric* batch_with_prefill_times_metric = nullptr;
    kmonitor::MutableMetric* batch_with_prefill_len_metric   = nullptr;

    kmonitor::MutableMetric* timeout_latency_us_metric  = nullptr;
    kmonitor::MutableMetric* malloc_failed_times_metric = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

// corresponding to python metrics
class RtpEmbeddingGlobalMetricsCollector final {
public:
    bool   error            = false;
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
    int64_t total_latency_us   = 0;
    int64_t wait_latency_us    = 0;
    int64_t input_token_length = 0;
};

class RtpEmbeddingStreamMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpEmbeddingStreamMetricsCollector* collector);

public:
    kmonitor::MutableMetric* total_latency_us_metric   = nullptr;
    kmonitor::MutableMetric* wait_latency_us_metric    = nullptr;
    kmonitor::MutableMetric* input_token_length_metric = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMSchedulerMetricsCollector final {
public:
    int64_t wait_stream_size           = 0;
    int64_t running_stream_size        = 0;
    int64_t remote_running_stream_size = 0;
    int64_t loading_cache_stream_size  = 0;
};

class RtpLLMSchedulerMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMSchedulerMetricsCollector* collector);

public:
    kmonitor::MutableMetric* wait_stream_size_metric           = nullptr;
    kmonitor::MutableMetric* running_stream_size_metric        = nullptr;
    kmonitor::MutableMetric* remote_running_stream_size_metric = nullptr;
    kmonitor::MutableMetric* loading_cache_stream_size_metric  = nullptr;

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
    explicit MetricsLoopReporter(const kmonitor::MetricsReporterPtr metrics_reporter, int interval_ms = 1000):
        collector_(CollectType()), interval_ms_(interval_ms), metrics_reporter_(metrics_reporter) {
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

    void report(const CollectType* collector) {
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
    std::mutex                   mutex_;
    bool                         stop_ = false;
    CollectType                  collector_;
    int                          interval_ms_ = 1000;
    std::thread                  metrics_reporter_thread_;
    kmonitor::MetricsReporterPtr metrics_reporter_ = nullptr;
};

class RtpLLMExecutorMetricsCollector final {
public:
    int64_t context_batch_size                   = 0;
    int64_t generate_batch_size                  = 0;
    int64_t context_batch_size_when_has_context  = 0;
    int64_t generate_batch_size_when_has_context = 0;
    int64_t execute_token_size_when_has_context  = 0;
    int64_t max_seq_len_when_has_context         = 0;
    int64_t execute_token_size                   = 0;
    int64_t max_seq_len                          = 0;
    int64_t gather_model_input_us                = 0;
    int64_t tp_sync_input_us                     = 0;
    int64_t model_forward_us                     = 0;
    int64_t sample_input_us                      = 0;
    int64_t dispatch_output_us                   = 0;

    // eplb metrics
    int64_t eplb_step_latency_us = 0;
};

class RtpLLMExecutorMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMExecutorMetricsCollector* collector);

public:
    kmonitor::MutableMetric* context_batch_size_metric                   = nullptr;
    kmonitor::MutableMetric* generate_batch_size_metric                  = nullptr;
    kmonitor::MutableMetric* context_batch_size_when_has_context_metric  = nullptr;
    kmonitor::MutableMetric* generate_batch_size_when_has_context_metric = nullptr;
    kmonitor::MutableMetric* execute_token_size_when_has_context_metric  = nullptr;
    kmonitor::MutableMetric* max_seq_len_when_has_context_metric         = nullptr;
    kmonitor::MutableMetric* execute_token_size_metric                   = nullptr;
    kmonitor::MutableMetric* max_seq_len_metric                          = nullptr;

    kmonitor::MutableMetric* gather_model_input_us_metric = nullptr;
    kmonitor::MutableMetric* tp_sync_input_us_metric      = nullptr;
    kmonitor::MutableMetric* model_forward_us_metric      = nullptr;
    kmonitor::MutableMetric* sample_input_us_metric       = nullptr;
    kmonitor::MutableMetric* dispatch_output_us_metric    = nullptr;

    // eplb metrics
    kmonitor::MutableMetric* eplb_step_latency_us_metric = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMCacheMetricsCollector final {
public:
    int64_t kv_cache_item_num         = 0;
    int64_t kv_cache_free_blocks      = 0;
    int64_t kv_cache_available_blocks = 0;
    int64_t kv_cache_left_seq         = 0;
    float   kv_cache_used_ratio       = 0;
    int64_t mr_cost_time_ms           = 0;
};

class RtpLLMCacheMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMCacheMetricsCollector* collector);

public:
    kmonitor::MutableMetric* kv_cache_item_num_metric         = nullptr;
    kmonitor::MutableMetric* kv_cache_free_blocks_metric      = nullptr;
    kmonitor::MutableMetric* kv_cache_available_blocks_metric = nullptr;
    kmonitor::MutableMetric* kv_cache_left_seq_metric         = nullptr;
    kmonitor::MutableMetric* kv_cache_used_ratio_metric       = nullptr;
    kmonitor::MutableMetric* mr_cost_time_ms_metric           = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMCacheReuseMetricsCollector final {
public:
    int64_t match_cost_time_us    = 0;
    int64_t kv_cache_reuse_length = 0;
    int64_t gpu_input_length      = 0;
    int64_t gpu_reuse_length      = 0;
    float   gpu_cache_hit_rate    = 0;
};

class RtpLLMRemoteCacheMatchMetricsCollector final {
public:
    bool    remote_match_qps             = true;
    bool    remote_match_fail_qps        = true;
    int64_t remote_match_reuse_block_num = 0;
    int64_t remote_match_time_us         = 0;
};

class RtpLLMRemoteCacheReadMetricsCollector final {
public:
    bool    remote_read_qps               = true;
    bool    remote_read_fail_qps          = true;
    int64_t remote_read_task_cost_time_us = 0;
};

class RtpLLMRemoteCacheWriteMetricsCollector final {
public:
    bool    remote_write_qps                  = true;
    bool    remote_write_fail_qps             = true;
    int64_t remote_write_cache_block_num      = 0;
    int64_t remote_write_task_cost_time_us    = 0;
    int64_t remote_get_write_location_time_us = 0;
    int64_t remote_write_broadcast_time_us    = 0;
    int64_t remote_finish_write_time_us       = 0;
};

class RtpLLMRemoteCacheSDKMetricsCollector final {
public:
    bool    remote_sdk_fail_qps     = true;
    int64_t remote_sdk_block_num    = 0;
    int64_t remote_sdk_cost_time_us = 0;
};

class RtpLLMRemoteCacheMatchMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMRemoteCacheMatchMetricsCollector* collector);

public:
    kmonitor::MutableMetric* remote_match_qps_metric             = nullptr;
    kmonitor::MutableMetric* remote_match_fail_qps_metric        = nullptr;
    kmonitor::MutableMetric* remote_match_reuse_block_num_metric = nullptr;
    kmonitor::MutableMetric* remote_match_time_us_metric         = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMRemoteCacheReadMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMRemoteCacheReadMetricsCollector* collector);

public:
    kmonitor::MutableMetric* remote_read_qps_metric               = nullptr;
    kmonitor::MutableMetric* remote_read_fail_qps_metric          = nullptr;
    kmonitor::MutableMetric* remote_read_task_cost_time_us_metric = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMRemoteCacheWriteMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMRemoteCacheWriteMetricsCollector* collector);

public:
    kmonitor::MutableMetric* remote_write_qps_metric                  = nullptr;
    kmonitor::MutableMetric* remote_write_fail_qps_metric             = nullptr;
    kmonitor::MutableMetric* remote_write_cache_block_num_metric      = nullptr;
    kmonitor::MutableMetric* remote_write_task_cost_time_us_metric    = nullptr;
    kmonitor::MutableMetric* remote_get_write_location_time_us_metric = nullptr;
    kmonitor::MutableMetric* remote_write_broadcast_time_us_metric    = nullptr;
    kmonitor::MutableMetric* remote_finish_write_time_us_metric       = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMRemoteCacheSDKMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMRemoteCacheSDKMetricsCollector* collector);

public:
    kmonitor::MutableMetric* remote_sdk_fail_qps_metric     = nullptr;
    kmonitor::MutableMetric* remote_sdk_block_num_metric    = nullptr;
    kmonitor::MutableMetric* remote_sdk_cost_time_us_metric = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMCacheReuseMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMCacheReuseMetricsCollector* collector);

public:
    kmonitor::MutableMetric* match_cost_time_us    = nullptr;
    kmonitor::MutableMetric* kv_cache_reuse_length = nullptr;
    kmonitor::MutableMetric* gpu_input_length      = nullptr;
    kmonitor::MutableMetric* gpu_reuse_length      = nullptr;
    kmonitor::MutableMetric* gpu_cache_hit_rate    = nullptr;

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

class RtpLLMKVCacheInfoMetricsCollector final {
public:
    bool    qps              = false;
    int64_t total_latency_us = 0;
};

class RtpLLMKVCacheInfoMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMKVCacheInfoMetricsCollector* collector);

public:
    kmonitor::MutableMetric* qps_metric              = nullptr;
    kmonitor::MutableMetric* total_latency_us_metric = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMSpeculativeEngineMetricsCollector final {
public:
    int64_t step_latency_us                = 0;
    int64_t propose_step_latency_us        = 0;
    int64_t score_step_latency_us          = 0;
    int64_t speculative_sampler_latency_us = 0;
    int64_t total_propose_token_num        = 0;
    int64_t total_accepted_token_num       = 0;
    int64_t total_stream_num               = 0;
};

class RtpLLMSpeculativeEngineMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMSpeculativeEngineMetricsCollector* collector);

public:
    kmonitor::MutableMetric* step_latency_us_metric                = nullptr;
    kmonitor::MutableMetric* propose_step_latency_us_metric        = nullptr;
    kmonitor::MutableMetric* score_step_latency_us_metric          = nullptr;
    kmonitor::MutableMetric* speculative_sampler_latency_us_metric = nullptr;
    kmonitor::MutableMetric* total_propose_token_num_metric        = nullptr;
    kmonitor::MutableMetric* total_accepted_token_num_metric       = nullptr;
    kmonitor::MutableMetric* sp_avg_accept_token_num_metric        = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLmEplbMetricsCollector final {
public:
    int64_t ep_rank;
    int64_t update_layer_id;

    int64_t update_weights_latency_ms;
    bool    update_weights_qps;

    std::vector<int64_t> gpu_loads;
};

class RtpLLmEplbMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLmEplbMetricsCollector* collector);

public:
    kmonitor::MutableMetric* update_weights_qps_metric        = nullptr;
    kmonitor::MutableMetric* update_layer_weights_qps_metric  = nullptr;
    kmonitor::MutableMetric* update_weights_latency_ms_metric = nullptr;
    kmonitor::MutableMetric* gpu_loads_metric                 = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMCacheStoreLoadClientMetricsCollector final {
public:
    bool    success                  = true;
    int64_t block_count              = 0;
    int64_t total_block_size         = 0;
    int64_t latency_us               = 0;  // 从调用CacheStore::load 到load完成的时间
    int64_t wait_task_run_latency_us = 0;  // 从调用CacheStore::load 到 开始处理的时间
    int64_t prepare_call_latency_us  = 0;  // 准备发送请求/建连的时间
    int64_t server_call_latency_us   = 0;  // 从调用CacheStore::load 到 收到响应的时间
    int64_t response_send_cost_us    = 0;  // server到client response传输的时间
};

class RtpLLMCacheStoreLoadServerMetricsCollector final {
public:
    bool    success                      = true;
    int64_t block_count                  = 0;
    int64_t total_block_size             = 0;
    int64_t latency_us                   = 0;  // 从接收到请求 到 最后一个 block 传输完的时间
    int64_t request_send_cost_us         = 0;  // load 请求从 client 发送到 server 耗时
    int64_t first_block_ready_latency_us = 0;  // 等待第一block可以传输的耗时
    int64_t all_block_ready_latency_us   = 0;  // 从接收到请求到最后一个block ready 可以传输的时间.
    int64_t transfer_gap_latency_us      = 0;  // 从最后一个block ready 可以传输 到 最后一个 block 传输完成的时间差.

    std::vector<int64_t> write_block_count;       // 调用write的block数量
    std::vector<int64_t> write_total_block_size;  //  调用write的block size
    std::vector<int64_t> write_latency_us;        // 调用write的单次延迟
};

class RtpLLMCacheStoreStoreMetricsCollector final {
public:
    bool    success                    = true;
    int64_t block_count                = 0;
    int64_t total_block_size           = 0;
    int64_t latency_us                 = 0;
    int64_t wait_task_run_latency_us   = 0;
    int64_t wait_event_sync_latency_us = 0;
};

class RtpLLMCacheStoreRemoteStoreMetricsCollector final {
public:
    bool    success                      = true;
    int64_t block_count                  = 0;
    int64_t total_block_size             = 0;
    int64_t latency_us                   = 0;  // 从调用CacheStore::submitStoreTask 到store完成的时间
    int64_t first_block_ready_latency_us = 0;  // 第一block可以传输
    int64_t all_block_ready_latency_us = 0;  // 从调用CacheStore::submitStoreTask 到最后一个block ready 可以传输的时间.
    int64_t transfer_gap_latency_us    = 0;  // 从最后一个block ready 可以传输 到 最后一个 block 传输完成的时间差.
};

class RtpLLMCacheStoreTransferMetricsCollector final {
public:
    int64_t block_count      = 0;
    int64_t total_block_size = 0;
    int64_t latency_us       = 0;
};

class RtpLLMCacheStoreMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMCacheStoreLoadClientMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, RtpLLMCacheStoreLoadServerMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, RtpLLMCacheStoreStoreMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, RtpLLMCacheStoreRemoteStoreMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, RtpLLMCacheStoreTransferMetricsCollector* collector);

public:
    kmonitor::MutableMetric* load_client_qps_metric                      = nullptr;
    kmonitor::MutableMetric* load_client_error_qps_metric                = nullptr;
    kmonitor::MutableMetric* load_client_block_count_metric              = nullptr;
    kmonitor::MutableMetric* load_client_total_block_size_metric         = nullptr;
    kmonitor::MutableMetric* load_client_latency_us_metric               = nullptr;
    kmonitor::MutableMetric* load_client_wait_task_run_latency_us_metric = nullptr;
    kmonitor::MutableMetric* load_client_prepare_call_latency_us_metric  = nullptr;
    kmonitor::MutableMetric* load_client_server_call_latency_us_metric   = nullptr;
    kmonitor::MutableMetric* load_client_response_send_cost_us_metric    = nullptr;

    kmonitor::MutableMetric* load_server_qps_metric                          = nullptr;
    kmonitor::MutableMetric* load_server_error_qps_metric                    = nullptr;
    kmonitor::MutableMetric* load_server_block_count_metric                  = nullptr;
    kmonitor::MutableMetric* load_server_total_block_size_metric             = nullptr;
    kmonitor::MutableMetric* load_server_latency_us_metric                   = nullptr;
    kmonitor::MutableMetric* load_server_request_send_cost_us_metric         = nullptr;
    kmonitor::MutableMetric* load_server_first_block_ready_latency_us_metric = nullptr;
    kmonitor::MutableMetric* load_server_all_block_ready_latency_us_metric   = nullptr;
    kmonitor::MutableMetric* load_server_transfer_gap_latency_us_metric      = nullptr;
    kmonitor::MutableMetric* load_server_write_block_count_metric            = nullptr;
    kmonitor::MutableMetric* load_server_write_total_block_size              = nullptr;
    kmonitor::MutableMetric* load_server_write_latency_us_metric             = nullptr;

    kmonitor::MutableMetric* store_qps_metric                        = nullptr;
    kmonitor::MutableMetric* store_error_qps_metric                  = nullptr;
    kmonitor::MutableMetric* store_block_count_metric                = nullptr;
    kmonitor::MutableMetric* store_total_block_size_metric           = nullptr;
    kmonitor::MutableMetric* store_latency_us_metric                 = nullptr;
    kmonitor::MutableMetric* store_wait_task_run_latency_us_metric   = nullptr;
    kmonitor::MutableMetric* store_wait_event_sync_latency_us_metric = nullptr;

    kmonitor::MutableMetric* remote_store_qps_metric                          = nullptr;
    kmonitor::MutableMetric* remote_store_error_qps_metric                    = nullptr;
    kmonitor::MutableMetric* remote_store_block_count_metric                  = nullptr;
    kmonitor::MutableMetric* remote_store_total_block_size_metric             = nullptr;
    kmonitor::MutableMetric* remote_store_latency_us_metric                   = nullptr;
    kmonitor::MutableMetric* remote_store_first_block_ready_latency_us_metric = nullptr;
    kmonitor::MutableMetric* remote_store_all_block_ready_latency_us_metric   = nullptr;
    kmonitor::MutableMetric* remote_store_transfer_gap_latency_us_metric      = nullptr;

    kmonitor::MutableMetric* transfer_qps_metric              = nullptr;
    kmonitor::MutableMetric* transfer_error_qps_metric        = nullptr;
    kmonitor::MutableMetric* transfer_block_count_metric      = nullptr;
    kmonitor::MutableMetric* transfer_total_block_size_metric = nullptr;
    kmonitor::MutableMetric* transfer_latency_us_metric       = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMMemoryCacheMatchMetricsCollector final {
public:
    bool    failed        = false;
    int64_t latency_us    = 0;
    int64_t input_token   = 0;
    int64_t matched_token = 0;
};

class RtpLLMMemoryCacheReadMetricsCollector final {
public:
    bool    failed        = false;
    int64_t latency_us    = 0;
    int64_t input_token   = 0;
    int64_t matched_token = 0;
    int64_t read_token    = 0;
};

class RtpLLMMemoryCacheWriteMetricsCollector final {
public:
    bool    failed      = false;
    int64_t latency_us  = 0;
    int64_t input_token = 0;
    int64_t write_token = 0;
};

class RtpLLMMemoryCacheCopyMetricsCollector final {
public:
    bool    failed     = false;
    int64_t latency_us = 0;
    bool    from_gpu   = false;
};

class RtpLLMMemoryCacheStatusMetricsCollector final {
public:
    int64_t total_block_num     = 0;
    int64_t allocated_block_num = 0;  // 在cache中的block数量
    int64_t available_block_num = 0;  // 可用的block数量
};

class RtpLLMMemoryCacheMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMMemoryCacheMatchMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, RtpLLMMemoryCacheReadMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, RtpLLMMemoryCacheWriteMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, RtpLLMMemoryCacheCopyMetricsCollector* collector);
    void report(const kmonitor::MetricsTags* tags, RtpLLMMemoryCacheStatusMetricsCollector* collector);

public:
    kmonitor::MutableMetric* kv_cache_memory_cache_match_qps_metric         = nullptr;
    kmonitor::MutableMetric* kv_cache_memory_cache_match_failed_qps_metric  = nullptr;
    kmonitor::MutableMetric* kv_cache_memory_cache_match_none_qps_metric    = nullptr;
    kmonitor::MutableMetric* kv_cache_memory_cache_match_latency_metric     = nullptr;
    kmonitor::MutableMetric* kv_cache_memory_cache_match_input_token_metric = nullptr;
    kmonitor::MutableMetric* kv_cache_memory_cache_matched_token_metric     = nullptr;

    kmonitor::MutableMetric* kv_cache_memory_cache_read_qps_metric         = nullptr;
    kmonitor::MutableMetric* kv_cache_memory_cache_read_none_qps_metric    = nullptr;
    kmonitor::MutableMetric* kv_cache_memory_cache_read_failed_qps_metric  = nullptr;
    kmonitor::MutableMetric* kv_cache_memory_cache_read_latency_metric     = nullptr;
    kmonitor::MutableMetric* kv_cache_memory_cache_read_input_token_metric = nullptr;
    kmonitor::MutableMetric* kv_cache_memory_cache_read_token_metric       = nullptr;

    kmonitor::MutableMetric* kv_cache_memory_cache_write_qps_metric         = nullptr;
    kmonitor::MutableMetric* kv_cache_memory_cache_write_none_qps_metric    = nullptr;
    kmonitor::MutableMetric* kv_cache_memory_cache_write_failed_qps_metric  = nullptr;
    kmonitor::MutableMetric* kv_cache_memory_cache_write_latency_metric     = nullptr;
    kmonitor::MutableMetric* kv_cache_memory_cache_write_input_token_metric = nullptr;
    kmonitor::MutableMetric* kv_cache_memory_cache_write_token_metric       = nullptr;

    kmonitor::MutableMetric* kv_cache_memory_cache_copy_qps_metric        = nullptr;
    kmonitor::MutableMetric* kv_cache_memory_cache_copy_failed_qps_metric = nullptr;
    kmonitor::MutableMetric* kv_cache_memory_cache_copy_latency_metric    = nullptr;

    kmonitor::MutableMetric* kv_cache_memory_cache_status_total_block_num_metric     = nullptr;
    kmonitor::MutableMetric* kv_cache_memory_cache_status_allocated_block_num_metric = nullptr;
    kmonitor::MutableMetric* kv_cache_memory_cache_status_available_block_num_metric = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

bool initKmonitorFactory();
void stopKmonitorFactory();

void setHippoTags(kmonitor::MetricsConfig& config);

}  // namespace rtp_llm
