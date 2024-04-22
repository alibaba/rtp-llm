#include "maga_transformer/cpp/speculative_engine/SpeculativeExecutor.h"
#include "maga_transformer/cpp/batch_stream_processor/NormalBatchStreamProcessor.h"
#include "maga_transformer/cpp/batch_stream_processor/SpeculativeBatchStreamProcessor.h"
#include "maga_transformer/cpp/common/status_util.h"

using namespace std;

namespace rtp_llm {

SpeculativeExecutor::SpeculativeExecutor(
    const MagaInitParams&                                                   params,
    const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights,
    const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights) {
    unique_ptr<GptModelInitParams> model_params;
    model_.reset(new GptModel(*model_params));
    sp_model_.reset(new GptModel(*model_params));

    SamplerInitParams sampler_params;
    sampler_.reset(new Sampler(sampler_params));
    sp_sampler_.reset(new SpeculativeSampler(sampler_params));
    batch_stream_processor_.reset(new SpeculativeBatchStreamProcessor(*params.gpt_init_parameter));
}

absl::Status SpeculativeExecutor::process(const std::list<GenerateStreamPtr>& streams) {
    // ModelOutput model_output;
    // SamplerOutput sample_output;
    // auto merged_input_status = batch_stream_processor_->gather(streams);
    // RETURN_IF_STATUS_OR_ERROR(merged_input_status);
    StreamGroups                 stream_groups(streams);
    std::unique_ptr<MergedInput> merged_input;
    // auto& merged_input = merged_input_status.value();
    auto model_input = std::move(merged_input->model_input);
    for (auto i = 0; i < gen_num_; ++i) {
        auto model_output   = sp_model_->forward(model_input);
        auto sampler_output = sampler_->forward(merged_input->sampler_input);
        RETURN_IF_STATUS_ERROR(SpeculativeBatchStreamProcessor::updateSPInput(model_input, sampler_output));
    }
    // kv cache is error, need fix
    RETURN_IF_STATUS_ERROR(SpeculativeBatchStreamProcessor::createValidateInput(model_input));
    auto merged_output            = std::make_unique<MergedOutput>();
    merged_output->model_output   = sp_model_->forward(model_input);
    merged_output->sampler_output = sp_sampler_->forward(merged_input->sampler_input);
    return batch_stream_processor_->dispatch(stream_groups, merged_output);
}

}  // namespace rtp_llm
