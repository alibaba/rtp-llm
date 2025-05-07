#include "maga_transformer/cpp/normal_engine/NormalExecutor.h"
#include <cstdlib>
#include <memory>
#include "maga_transformer/cpp/utils/StatusUtil.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/models/Sampler.h"
#include "maga_transformer/cpp/th_op/GptInitParameter.h"
#include "torch/csrc/autograd/profiler_kineto.h"

using namespace std;

namespace rtp_llm {
namespace tap = torch::autograd::profiler;
namespace tpi = torch::profiler::impl;

class CudaProfiler {
public:
    CudaProfiler(const std::string& prefix): prefix_(prefix) {
        tap::prepareProfiler(config_, activities_);
    }
    ~CudaProfiler() {
        if (!stoped_) {
            stoped_ = true;
            stop();
        }
    }
    void start() {
        count += 1;
        stoped_ = false;
        tap::enableProfiler(config_, activities_);
    }
    void stop() {
        std::unique_ptr<tap::ProfilerResult> res = tap::disableProfiler();
        std::string file_name = prefix_ + std::to_string(count) + ".json";
        res->save(file_name);
        stoped_ = true;
    }
protected:
    static size_t count;
    std::string   prefix_;
    tpi::ProfilerConfig config_ = tpi::ProfilerConfig(tpi::ProfilerState::KINETO);
    std::set<tpi::ActivityType> activities_{tpi::ActivityType::CUDA};
    bool stoped_ = true;
};
size_t CudaProfiler::count = 0;

NormalExecutor::NormalExecutor(const EngineInitParams& params,
                               const std::shared_ptr<CacheManager>& cache_manager,
                               rtp_llm::DeviceBase* device,
                               const std::shared_ptr<lora::LoraManager>& lora_manager,
                               bool warm_up):
    Executor(device),
    cache_manager_(cache_manager),
    lora_manager_(lora_manager),
    warm_up_(warm_up),
    gen_timeline_sync_(autil::EnvUtil::getEnv("GEN_TIMELINE_SYNC", 0L)),
    metrics_reporter_(params.metrics_reporter),
    tps_reporter_(MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(metrics_reporter_))
{
    auto& gpt_param = params.gpt_init_parameter;
    if (gpt_param.enable_eplb_ && gpt_param.moe_style_ != 0) {
        // use first moe layer weight as moe weight type
        int  first_moe_layer = gpt_param.moe_layer_index_.front();
        auto moe_weight_type = params.gpt_weights.layers[first_moe_layer].ffn_weights.moe_gate_weight->kernel->type();

        expert_balancer_ = make_shared<ExpertBalancer>(gpt_param.expert_num_,
                                                       gpt_param.phy_exp_num_,
                                                       gpt_param.num_layers_,
                                                       gpt_param.moe_inter_padding_size_,
                                                       gpt_param.hidden_size_,
                                                       gpt_param.eplb_update_time_,
                                                       gpt_param.ep_rank_,
                                                       gpt_param.ep_size_,
                                                       gpt_param.py_eplb_,
                                                       moe_weight_type,
                                                       device_,
                                                       gpt_param.eplb_mode_,
                                                       gpt_param.quant_algo_,
                                                       metrics_reporter_);
    }

    int eos_id = params.gpt_init_parameter.special_tokens_.eos_token_id_;
    SamplerInitParams sampler_params{device_, eos_id, device->initParams().max_batch_size}; // set static max batch size to avoid sampler reset memory
    sampler_.reset(new Sampler(sampler_params));

    // CacheManager::KVCacheBuffer kv_cache_buffer;
    // CacheConfig cache_config;
    // if (warmup) {
    //     kv_cache_buffer.k_blocks =
    // } else {
    //     kv_cache_buffer = cache_manager->kvCacheBuffer();
    //     cache_config = cache_manager->cacheConfig();
    // }

    model_.reset(new GptModel({
        device_,
        params.gpt_weights,
        genModelDescription(params.gpt_init_parameter),
        cache_manager ? ((optional<CacheManager::KVCacheBuffer>)cache_manager->kvCacheBuffer()) : nullopt
    }));

    // when warmup, cache manager maybe nullptr
    const auto& cache_config = cache_manager ? cache_manager->cacheConfig() : CacheConfig();
    batch_stream_processor_.reset(new NormalBatchStreamProcessor(
        params.gpt_init_parameter, cache_config, warm_up_));
}

absl::Status NormalExecutor::process(const std::list<GenerateStreamPtr>& streams) {
    StreamGroups stream_groups(streams);
    bool gen_timeline = stream_groups.genTimeline();
    if (gen_timeline_sync_) {
        auto gen_timeline_buffer = device_->allocateBuffer(
                {rtp_llm::DataType::TYPE_BOOL, {device_->getDeviceProperties().dp_size}, rtp_llm::AllocationType::HOST});
        *(gen_timeline_buffer->dataWithOffset<bool>(device_->getDeviceProperties().dp_rank)) = gen_timeline;
        device_->allGather({{gen_timeline_buffer}, rtp_llm::ParallelMode::DP_AND_TP});
        device_->syncCommunication();
        gen_timeline = std::any_of(gen_timeline_buffer->data<bool>(), gen_timeline_buffer->dataWithOffset<bool>(device_->getDeviceProperties().dp_size), [](auto s) { return s;});
    }
    std::shared_ptr<CudaProfiler> profiler;
    if (gen_timeline) {
        profiler = std::make_shared<CudaProfiler>("cuda_profiler_dp" + std::to_string(device_->getDeviceProperties().dp_rank) + "_");
        profiler->start();
    }
    RtpLLMExecutorMetricsCollector executor_collector;
    RtpLLMTokenPSMetricsCollector tps_collector;
    GptModelInputs model_input;
    GptModelOutputs model_output;
    SamplerOutput sampler_output;
    {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        auto model_input_status = batch_stream_processor_->gatherModelInput(stream_groups);
        RETURN_IF_STATUS_OR_ERROR(model_input_status);
        model_input = std::move(model_input_status.value());
        executor_collector.gather_model_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        dpAndTpSyncModelInputs(model_input, device_);
        executor_collector.tp_sync_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    // get lora input
    if (lora_manager_) {
        model_input.lora_model_input = lora_manager_->makeLoraModelInput(model_input.lora_ids,
                                                                         model_input.lora_input_lengths);
    }
    {
        RTP_LLM_LOG_DEBUG("model_input: %s", model_input.debugString().c_str());
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        model_output = std::move(model_->forward(model_input));
        executor_collector.model_forward_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        RTP_LLM_LOG_DEBUG("model forward done");
    }
    if (expert_balancer_) {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        expert_balancer_->stepForward(*model_, executor_collector);
        executor_collector.eplb_step_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    if (device_->getDeviceProperties().tp_rank > 0 || warm_up_) {
        return absl::OkStatus();
    }
    {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        CHECK_AND_RETURN_REF(sampler_input, batch_stream_processor_->gatherSamplerInput(stream_groups, model_input, model_output));
        sampler_output = std::move(sampler_->forward(sampler_input));
        RTP_LLM_LOG_DEBUG("sampler forward done");
        executor_collector.sample_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        auto result = batch_stream_processor_->dispatch(stream_groups, {std::move(model_output), std::move(sampler_output)});
        executor_collector.dispatch_output_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        reportMetrics(stream_groups, executor_collector, tps_collector);
        return result;
    }
}

void NormalExecutor::reportMetrics(const StreamGroups& stream_groups,
                                   RtpLLMExecutorMetricsCollector& executor_collector,
                                   RtpLLMTokenPSMetricsCollector& tps_collector) {
    if (device_->getDeviceProperties().tp_rank > 0) {
        return;
    }
    if (metrics_reporter_) {
        executor_collector.context_batch_size  = stream_groups.totalContextBatchSize();
        executor_collector.generate_batch_size = stream_groups.totalDecodeBatchSize();
        executor_collector.execute_token_size  = stream_groups.modelExecuteTokenSize();
        executor_collector.max_seq_len         = stream_groups.maxSeqLen();
        if (executor_collector.context_batch_size != 0) {
            executor_collector.context_batch_size_when_has_context = executor_collector.context_batch_size;
            executor_collector.generate_batch_size_when_has_context = executor_collector.generate_batch_size;
            executor_collector.execute_token_size_when_has_context = executor_collector.execute_token_size;
            executor_collector.max_seq_len_when_has_context = executor_collector.max_seq_len;
        }
        metrics_reporter_->report<RtpLLMExecutorMetrics, RtpLLMExecutorMetricsCollector>(nullptr, &executor_collector);

        tps_collector.context_tps = stream_groups.modelExecuteTokenSize() - stream_groups.totalDecodeBatchSize();
        tps_collector.generate_tps = stream_groups.totalDecodeBatchSize();
        tps_collector.total_tps = stream_groups.modelExecuteTokenSize();
        tps_reporter_.report(&tps_collector);
    }
}

}  // namespace rtp_llm
