#include "maga_transformer/cpp/normal_engine/NormalExecutor.h"
#include "maga_transformer/cpp/batch_stream_processor/NormalBatchStreamProcessor.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/deprecated/ParallelModelWrapper.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/models/Sampler.h"
#include "src/fastertransformer/devices/DeviceFactory.h"

using namespace std;

namespace rtp_llm {

NormalExecutor::NormalExecutor(const MagaInitParams&                                                   params,
                               const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights,
                               const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights) {
    // need init model and sampler
    unique_ptr<GptModelInitParams> model_params;
    // model_.reset(new GptModel(*model_params));
    SamplerInitParams sampler_params;
    device_               = ft::DeviceFactory::getDevice(ft::DeviceType::Cuda);
    sampler_params.device = device_;
    sampler_.reset(new Sampler(sampler_params));
    batch_stream_processor_.reset(new NormalBatchStreamProcessor(*params.gpt_init_parameter));
    model_wrapper_.reset(
        new ParallelModelWrapper(*params.gpt_init_parameter, 1, "localhost", 0, weights, layer_weights));
}

void NormalExecutor::addLoRA(const int64_t                                                   lora_id,
                             const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_a_weights,
                             const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& lora_b_weights) {
    model_wrapper_->addLoRA(lora_id, lora_a_weights, lora_b_weights);
}

void NormalExecutor::removeLoRA(const int64_t lora_id) {
    model_wrapper_->removeLoRA(lora_id);
}

ModelRequest NormalExecutor::generateOldModelRequest(GptModelInputs& model_input) {
    ModelRequest model_request;
    model_request.generate_batch_size = model_input.sequence_lengths->shape()[0];
    model_request.context_batch_size  = model_input.input_lengths->shape()[0] - model_request.generate_batch_size;
    model_request.combo_tokens        = std::move(model_input.combo_tokens);
    model_request.input_lengths       = std::move(model_input.input_lengths);
    model_request.sequence_lengths    = std::move(model_input.sequence_lengths);
    model_request.kv_cache_blocks     = std::move(model_input.kv_cache_blocks);
    return model_request;
}
absl::Status NormalExecutor::process(const std::list<GenerateStreamPtr>& streams) {
    StreamGroups stream_groups(streams);
    auto         model_input_status = batch_stream_processor_->gatherModelInput(stream_groups);
    RETURN_IF_STATUS_OR_ERROR(model_input_status);
    auto& model_input = model_input_status.value();
    FT_LOG_DEBUG("model_input: %s", model_input.debugString().c_str());
    auto         merged_output        = std::make_unique<MergedOutput>();
    ModelRequest model_request        = std::move(generateOldModelRequest(model_input));
    auto         model_output         = std::move(model_wrapper_->forward(model_request));
    auto         sampler_input_status = batch_stream_processor_->gatherSamplerInput(stream_groups, *model_output);
    RETURN_IF_STATUS_OR_ERROR(sampler_input_status);
    auto& sampler_input           = sampler_input_status.value();
    merged_output->sampler_output = std::move(sampler_->forward(sampler_input));
    return batch_stream_processor_->dispatch(stream_groups, merged_output);
}

}  // namespace rtp_llm
