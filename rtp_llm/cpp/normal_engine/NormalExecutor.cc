#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include <cstdlib>
#include <memory>
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/models/GptModel.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/NativeDeviceGraphModel.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

using namespace std;

namespace rtp_llm {

NormalExecutor::NormalExecutor(const EngineInitParams&                   params,
                               const std::shared_ptr<KVCacheManager>&    cache_manager,
                               rtp_llm::DeviceBase*                      device,
                               const std::shared_ptr<lora::LoraManager>& lora_manager,
                               bool                                      warm_up):
    Executor(device),
    cache_manager_(cache_manager),
    lora_manager_(lora_manager),
    warm_up_(warm_up),
    use_all_gather_(params.moe_config.use_all_gather && !params.moe_config.use_deepep_low_latency),
    metrics_reporter_(params.metrics_reporter),
    tps_reporter_(MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(metrics_reporter_)) {
    enable_detail_log_ = params.profiling_debug_logging_config.enable_detail_log;
    RTP_LLM_LOG_INFO("enable_detail_log_ = %d", enable_detail_log_);

    if (params.eplb_config.enable_eplb() && params.model_config_.moe_style != 0) {
        // use first moe layer weight as moe weight type
        int  first_moe_layer = params.model_config_.moe_layer_index.front();
        auto moe_weight_type = params.gpt_weights.layers[first_moe_layer].ffn_weights.moe_gate_weight->kernel->type();
        bool is_gated_activation = params.model_config_.isGatedActivation();
        auto moe_inter_size =
            is_gated_activation ?
                params.gpt_weights.layers[first_moe_layer].ffn_weights.moe_gate_weight->kernel->shape()[1] / 2 :
                params.gpt_weights.layers[first_moe_layer].ffn_weights.moe_gate_weight->kernel->shape()[1];

        expert_balancer_ = make_shared<ExpertBalancer>(params.model_config_.expert_num,
                                                       params.eplb_config.phy_exp_num(params.model_config_.expert_num),
                                                       params.model_config_.num_layers,
                                                       moe_inter_size,
                                                       params.model_config_.hidden_size,
                                                       params.parallelism_config.ep_rank,
                                                       params.parallelism_config.ep_size,
                                                       params.py_eplb,
                                                       moe_weight_type,
                                                       device_,
                                                       params.model_config_.quant_algo,
                                                       metrics_reporter_,
                                                       params.eplb_config);
    }

    sampler_.reset(new Sampler(SamplerInitParams{device_}));

    GptModelInitParams model_init_params(
        {device_,
         params.gpt_weights,
         genModelDescription(params.model_config_, params.parallelism_config, params.eplb_config, params.moe_config),
         cache_manager ? std::make_optional(cache_manager->kvCacheBuffer()) : std::nullopt,
         params.model_id});

    if (params.ffn_disaggregate_config.enable_ffn_disaggregate) {
        RTP_LLM_LOG_INFO("using ffn as service");
        enable_ffn_disaggregate_ = true;
    }
    if (!params.py_model.is_none()) {
        RTP_LLM_LOG_INFO("init executor with python model");
        model_.reset(new PyWrappedModel(model_init_params, params.py_model));
    } else if (device_->initParams().hw_kernel_config.enable_native_cuda_graph) {
        RTP_LLM_LOG_INFO("init legacy c++ gpt model with native cuda graph");
        model_.reset(new NativeDeviceGraphModel(model_init_params));
    } else {
        RTP_LLM_LOG_INFO("init legacy c++ gpt model");
        model_.reset(new GptModel(model_init_params));
    }

    // when warmup, cache manager maybe nullptr
    const auto& cache_config = cache_manager ? cache_manager->cacheConfig() : CacheConfig();
    batch_stream_processor_.reset(new NormalBatchStreamProcessor(
        params.model_config_, params.pd_sep_config, params.profiling_debug_logging_config, cache_config, warm_up_));
    PrefixToCandidateTokens::instance()->reloadPrefixDictWithPrefix(params.model_config_.ckpt_path,
                                                                    params.sp_config.tree_decode_config);
    device_->profileStart();
}

absl::Status NormalExecutor::process(const std::list<GenerateStreamPtr>& streams) {
    StreamGroups                   stream_groups(streams);
    RtpLLMExecutorMetricsCollector executor_collector;
    RtpLLMTokenPSMetricsCollector  tps_collector;
    GptModelInputs                 model_input;
    GptModelOutputs                model_output;
    SamplerOutput                  sampler_output;
    {
        int64_t start_time_us      = autil::TimeUtility::currentTimeInMicroSeconds();
        auto    model_input_status = batch_stream_processor_->gatherModelInput(stream_groups);
        RETURN_IF_STATUS_OR_ERROR(model_input_status);
        model_input                              = std::move(model_input_status.value());
        executor_collector.gather_model_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        model_input.skip_run  = streams.empty() && !enable_ffn_disaggregate_;
        tpSyncModelInputs(model_input, device_);
        if (model_input.skip_run) {
            return absl::OkStatus();
        }
        executor_collector.tp_sync_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    // make sure last model input is released before forward
    model_->releaseBuffers();

    {
        // update kv cache
        if (model_input.kv_cache_update_mapping) {
            cache_manager_->blockBatchCopy(*model_input.kv_cache_update_mapping);
        }
    }
    // get lora input
    if (lora_manager_) {
        model_input.lora_model_input =
            lora_manager_->makeLoraModelInput(model_input.lora_ids, model_input.lora_input_lengths);
    }
    {
        bool force = device_->getDeviceProperties().tp_rank == 0 && enable_detail_log_;
        if (force) {
            RTP_LLM_LOG_INFO("model_input: %s", model_input.debugString(force).c_str());
        } else {
            RTP_LLM_LOG_TRACE("model_input: %s", model_input.debugString(force).c_str());
        }
        int64_t start_time_us               = autil::TimeUtility::currentTimeInMicroSeconds();
        model_output                        = std::move(model_->forward(model_input));
        executor_collector.model_forward_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        RTP_LLM_LOG_DEBUG("model forward done");
    }
    if (expert_balancer_) {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        expert_balancer_->stepForward(*model_, executor_collector);
        executor_collector.eplb_step_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    if (device_->getDeviceProperties().tp_rank > 0 || warm_up_ || streams.size() == 0) {
        device_->syncAndCheck();
        model_->releaseBuffers();
        return absl::OkStatus();
    }
    {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        CHECK_AND_RETURN_REF(sampler_input,
                             batch_stream_processor_->gatherSamplerInput(stream_groups, model_input, model_output));
        sampler_output = std::move(sampler_->forward(sampler_input));
        RTP_LLM_LOG_DEBUG("sampler forward done");
        executor_collector.sample_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        auto    result =
            batch_stream_processor_->dispatch(stream_groups, {std::move(model_output), std::move(sampler_output)});
        executor_collector.dispatch_output_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        reportMetrics(stream_groups, executor_collector, tps_collector);

        model_->releaseBuffers();

        return result;
    }
}

void NormalExecutor::reportMetrics(const StreamGroups&             stream_groups,
                                   RtpLLMExecutorMetricsCollector& executor_collector,
                                   RtpLLMTokenPSMetricsCollector&  tps_collector) {
    if (device_->getDeviceProperties().tp_rank > 0) {
        return;
    }
    if (metrics_reporter_) {
        executor_collector.context_batch_size  = stream_groups.totalContextBatchSize();
        executor_collector.generate_batch_size = stream_groups.totalDecodeBatchSize();
        executor_collector.execute_token_size  = stream_groups.modelExecuteTokenSize();
        executor_collector.max_seq_len         = stream_groups.maxSeqLen();
        if (executor_collector.context_batch_size != 0) {
            executor_collector.context_batch_size_when_has_context  = executor_collector.context_batch_size;
            executor_collector.generate_batch_size_when_has_context = executor_collector.generate_batch_size;
            executor_collector.execute_token_size_when_has_context  = executor_collector.execute_token_size;
            executor_collector.max_seq_len_when_has_context         = executor_collector.max_seq_len;
        }
        metrics_reporter_->report<RtpLLMExecutorMetrics, RtpLLMExecutorMetricsCollector>(nullptr, &executor_collector);

        tps_collector.context_tps  = stream_groups.modelExecuteTokenSize() - stream_groups.totalDecodeBatchSize();
        tps_collector.generate_tps = stream_groups.totalDecodeBatchSize();
        tps_collector.total_tps    = stream_groups.modelExecuteTokenSize();
        tps_reporter_.report(&tps_collector);
    }
}

bool NormalExecutor::updateEplbConfig(const EPLBConfig& config) {
    if (expert_balancer_) {
        return expert_balancer_->updateEplbConfig(config);
    }
    return true;
}

}  // namespace rtp_llm
