#include "maga_transformer/cpp/normal_engine/NormalExecutor.h"
#include <cstdlib>
#include "maga_transformer/cpp/utils/StatusUtil.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/models/Sampler.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"

using namespace std;

namespace rtp_llm {

NormalExecutor::NormalExecutor(const EngineInitParams& params,
                               const std::shared_ptr<CacheManager>& cache_manager,
                               ft::DeviceBase* device,
                               const std::shared_ptr<lora::LoraManager>& lora_manager,
                               bool warm_up):
    Executor(device),
    cache_manager_(cache_manager),
    lora_manager_(lora_manager),
    warm_up_(warm_up),
    metrics_reporter_(params.metrics_reporter),
    tps_reporter_(MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(metrics_reporter_))
{
    int eos_id = params.gpt_init_parameter.special_tokens_.eos_token_id_;
    SamplerInitParams sampler_params{device_, eos_id, device->initParams().max_batch_size}; // set static max batch size to avoid sampler reset memory
    sampler_.reset(new Sampler(sampler_params));

    model_.reset(new GptModel({device_, params.gpt_weights, genModelDescription(params.gpt_init_parameter)}));
    // when warmup, cache manager maybe nullptr
    const auto& cache_config = cache_manager ? cache_manager->cacheConfig() : CacheConfig();
    batch_stream_processor_.reset(new NormalBatchStreamProcessor(
        params.gpt_init_parameter, cache_config, warm_up_));
}

absl::Status NormalExecutor::process(const std::list<GenerateStreamPtr>& streams) {
    StreamGroups stream_groups(streams);
    reportMetrics(stream_groups);
    CHECK_AND_RETURN_REF(model_input, batch_stream_processor_->gatherModelInput(stream_groups));
    tpSyncModelInputs(model_input, device_);
    // get lora input
    if (lora_manager_) {
        model_input.lora_model_input = lora_manager_->makeLoraModelInput(model_input.lora_ids,
                                                                         model_input.lora_input_lengths);
    }
    if (!warm_up_) {
        auto kv_cache_buffer = cache_manager_->kvCacheBuffer();
        model_input.k_cache_buffer = kv_cache_buffer.k_blocks;
        model_input.v_cache_buffer = kv_cache_buffer.v_blocks;
        model_input.k_scale_buffer = kv_cache_buffer.k_scale;
        model_input.v_scale_buffer = kv_cache_buffer.v_scale;
    }
    FT_LOG_DEBUG("model_input: %s", model_input.debugString().c_str());
    GptModelOutputs model_output = std::move(model_->forward(model_input));
    FT_LOG_DEBUG("model forward done");
    if (device_->getDeviceProperties().tp_rank > 0 || warm_up_) {
        return absl::OkStatus();
    }
    CHECK_AND_RETURN_REF(sampler_input, batch_stream_processor_->gatherSamplerInput(stream_groups, model_input, model_output));
    SamplerOutput sampler_output = std::move(sampler_->forward(sampler_input));
    FT_LOG_DEBUG("sampler forward done");
    return batch_stream_processor_->dispatch(stream_groups, {std::move(model_output), std::move(sampler_output)});
}

void NormalExecutor::reportMetrics(const StreamGroups& stream_groups) {
    if (metrics_reporter_) {
        RtpLLMExecutorMetricsCollector executor_collector;
        executor_collector.context_batch_size  = stream_groups.totalContextBatchSize();
        executor_collector.generate_batch_size = stream_groups.totalDecodeBatchSize();
        if (executor_collector.context_batch_size != 0) {
            executor_collector.context_batch_size_when_has_context = executor_collector.context_batch_size;
            executor_collector.generate_batch_size_when_has_context = executor_collector.generate_batch_size;
        }
        executor_collector.execute_token_size  = stream_groups.modelExecuteTokenSize();
        executor_collector.max_seq_len         = stream_groups.maxSeqLen();
        metrics_reporter_->report<RtpLLMExecutorMetrics, RtpLLMExecutorMetricsCollector>(nullptr, &executor_collector);

        RtpLLMTokenPSMetricsCollector tps_collector;
        tps_collector.context_tps = stream_groups.modelExecuteTokenSize() - stream_groups.totalDecodeBatchSize();
        tps_collector.generate_tps = stream_groups.totalDecodeBatchSize();
        tps_collector.total_tps = stream_groups.modelExecuteTokenSize();
        tps_reporter_.report(&tps_collector);
    }
}

}  // namespace rtp_llm
