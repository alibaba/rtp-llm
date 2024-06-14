#include "maga_transformer/cpp/normal_engine/NormalExecutor.h"
#include <cstdlib>
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/models/Sampler.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "src/fastertransformer/devices/Weights.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"

using namespace std;

namespace rtp_llm {

NormalExecutor::NormalExecutor(const EngineInitParams& params, ft::DeviceBase* device):
    Executor(device),
    metrics_reporter_(params.metrics_reporter),
    dtype_(ft::getDataType(params.gpt_init_parameter.data_type_)),
    is_causal_(params.gpt_init_parameter.is_causal_)
{
    size_t max_batch_size =
        params.gpt_init_parameter.max_context_batch_size_ + params.gpt_init_parameter.max_generate_batch_size_;
    int eos_id = params.gpt_init_parameter.special_tokens_.eos_token_id_;
    SamplerInitParams sampler_params{device_, eos_id, 1024}; // set static max batch size to avoid sampler reset memory
    sampler_.reset(new Sampler(sampler_params));

    use_new_device_impl_ = std::getenv("USE_NEW_DEVICE_IMPL");
    FT_LOG_INFO("model exec use new device impl: %d", (int)use_new_device_impl_);
    if (use_new_device_impl_) {
        model_.reset(new GptModel({device_, params.gpt_weights, genModelDescription(params.gpt_init_parameter)}));
        batch_stream_processor_.reset(new NormalBatchStreamProcessor(params.gpt_init_parameter));
        need_attention_mask_ = device_->getDeviceProperties().attention_need_mask;
#if USING_CUDA
    } else {
        model_wrapper_.reset(
            new ParallelModelWrapper(params.gpt_init_parameter, params.global_weights, params.layers_weights, params.linear_bias_slopes));
        batch_stream_processor_.reset(
            new NormalBatchStreamProcessor(params.gpt_init_parameter));
        need_attention_mask_ = !model_wrapper_->useFMHA();
#endif
    }
}

absl::Status
NormalExecutor::addLoRA(const int64_t                                                           lora_id,
                        const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_a_weights,
                        const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_b_weights) {
#if USING_CUDA
    model_wrapper_->addLoRA(lora_id, lora_a_weights, lora_b_weights);
#endif
    return absl::OkStatus();
}

absl::Status NormalExecutor::removeLoRA(const int64_t lora_id) {
#if USING_CUDA
    model_wrapper_->removeLoRA(lora_id);
#endif
    return absl::OkStatus();
}

ModelRequest NormalExecutor::generateOldModelRequest(GptModelInputs& model_input) {
    ModelRequest model_request;
    model_request.generate_batch_size = model_input.sequence_lengths->shape()[0];
    model_request.context_batch_size  = model_input.input_lengths->shape()[0] - model_request.generate_batch_size;
    model_request.combo_tokens        = model_input.combo_tokens;
    model_request.input_lengths       = model_input.input_lengths;
    model_request.sequence_lengths    = model_input.sequence_lengths;
    model_request.prefix_lengths      = model_input.prefix_lengths;
    model_request.count_lengths       = model_input.count_lengths;
    model_request.max_prefix_length   = model_input.max_prefix_length;
    model_request.combo_position_ids  = model_input.combo_position_ids;
    model_request.kv_cache_blocks     = model_input.kv_cache_blocks;
    model_request.kv_cache_scales     = model_input.kv_cache_scales;
    model_request.attention_mask      = model_input.attention_mask;
    model_request.lora_ids            = model_input.lora_ids;
    model_request.lora_input_lengths  = model_input.lora_input_lengths;
    return model_request;
}

absl::Status NormalExecutor::process(const std::list<GenerateStreamPtr>& streams) {
    StreamGroups stream_groups(streams);
    reportMetrics(stream_groups);
    auto model_input_status = batch_stream_processor_->gatherModelInput(stream_groups);
    RETURN_IF_STATUS_OR_ERROR(model_input_status);
    auto& model_input = model_input_status.value();
    tpSyncModelInputs(model_input, device_);
    if (need_attention_mask_ && model_input.input_lengths->size() > model_input.sequence_lengths->size()) {
        const auto generate_batch_size = model_input.sequence_lengths->size();
        const auto context_batch_size = model_input.input_lengths->size() - generate_batch_size;
        model_input.attention_mask = NormalBatchStreamProcessor::createAttentionMask({
                model_input.input_lengths->view(generate_batch_size, context_batch_size),
                model_input.prefix_lengths->view(generate_batch_size, context_batch_size),
                dtype_, is_causal_, device_});
    }
    FT_LOG_DEBUG("model_input: %s", model_input.debugString().c_str());
    auto            merged_output = std::make_unique<MergedOutput>();
    GptModelOutputs model_output;
    if (use_new_device_impl_) {
        model_output = std::move(model_->forward(model_input));
#if USING_CUDA
    } else {
        ModelRequest model_request = std::move(generateOldModelRequest(model_input));
        model_output               = std::move(model_wrapper_->forward(model_request));
#endif
    }
    FT_LOG_DEBUG("model forward done");
    if (device_->getDeviceProperties().tp_rank > 0) {
        return absl::OkStatus();
    }
    auto sampler_input_status = batch_stream_processor_->gatherSamplerInput(stream_groups, model_input, model_output);
    RETURN_IF_STATUS_OR_ERROR(sampler_input_status);
    auto& sampler_input           = sampler_input_status.value();
    merged_output->model_output   = std::move(model_output);
    merged_output->sampler_output = std::move(sampler_->forward(sampler_input));
    FT_LOG_DEBUG("sampler forward done");
    return batch_stream_processor_->dispatch(stream_groups, sampler_input, merged_output);
}

void NormalExecutor::reportMetrics(const StreamGroups& stream_groups) {
    if (metrics_reporter_) {
        RtpLLMExecutorMetricsCollector collector;
        collector.context_batch_size  = stream_groups.contextStreams().size();
        collector.generate_batch_size = stream_groups.totalModelBatchSize() - stream_groups.contextStreams().size();
        collector.execute_token_size  = stream_groups.modelExecuteTokenSize();
        collector.max_seq_len         = stream_groups.maxSeqLen();
        metrics_reporter_->report<RtpLLMExecutorMetrics, RtpLLMExecutorMetricsCollector>(nullptr, &collector);
    }
}

}  // namespace rtp_llm
