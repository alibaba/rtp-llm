#include "maga_transformer/cpp/speculative_engine/SpeculativeExecutor.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/deprecated/ParallelModelWrapper.h"
#include "maga_transformer/cpp/normal_engine/NormalExecutor.h"
#include "maga_transformer/cpp/speculative_engine/SpeculativeBatchStreamProcessor.h"
#include "maga_transformer/cpp/speculative_engine/SpeculativeStream.h"

using namespace std;

namespace rtp_llm {

SpeculativeExecutor::SpeculativeExecutor(
    const MagaInitParams&                                                   params,
    const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights,
    const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights)
{
    unique_ptr<GptModelInitParams> model_params;
    model_.reset(new GptModel(*model_params));

    // SamplerInitParams sampler_params;
    // sampler_.reset(new SpeculativeSampler(sampler_params));
    // model_wrapper_.reset(
    //         new ParallelModelWrapper(*params.gpt_init_parameter, weights, layer_weights));
    batch_stream_processor_.reset(new SpeculativeBatchStreamProcessor(params.gpt_init_parameter, !model_wrapper_->useFMHA()));
}

ModelRequest SpeculativeExecutor::generateOldModelRequest(GptModelInputs& model_input) {
    ModelRequest model_request;
    model_request.generate_batch_size = model_input.sequence_lengths->shape()[0];
    model_request.context_batch_size  = model_input.input_lengths->shape()[0] - model_request.generate_batch_size;
    model_request.combo_tokens        = std::move(model_input.combo_tokens);
    model_request.input_lengths       = std::move(model_input.input_lengths);
    model_request.sequence_lengths    = std::move(model_input.sequence_lengths);
    model_request.kv_cache_blocks     = std::move(model_input.kv_cache_blocks);
    return model_request;
}

absl::StatusOr<list<GenerateStreamPtr>> SpeculativeExecutor::getTargetStreams(const list<GenerateStreamPtr>& streams) {
    list<GenerateStreamPtr> target_streams;
    for (auto& stream : streams) {
        SpeculativeStream* stream_ = dynamic_cast<SpeculativeStream*>(stream.get());
        stream_->updateDraftToken();
        target_streams.emplace_back(stream_->targetStream());
    }
    return target_streams;
}

absl::Status SpeculativeExecutor::updateTargetProb(const list<GenerateStreamPtr>& streams, const ft::Buffer& logits) {
    size_t batch_index = 0;
    for (auto& stream : streams) {
        SpeculativeStream* stream_    = dynamic_cast<SpeculativeStream*>(stream.get());
        auto               batch_size = stream->batchSize();
        stream_->updateTargetProb(logits.view(batch_index, batch_size));
        batch_index += batch_size;
    }
    return absl::OkStatus();
}

absl::Status SpeculativeExecutor::process(const std::list<GenerateStreamPtr>& streams) {
    const auto target_streams_status = getTargetStreams(streams);
    RETURN_IF_STATUS_OR_ERROR(target_streams_status);
    StreamGroups stream_groups(target_streams_status.value());
    auto         model_input_status = batch_stream_processor_->gatherModelInput(stream_groups);
    RETURN_IF_STATUS_OR_ERROR(model_input_status);
    auto& model_input = model_input_status.value();
    FT_LOG_DEBUG("model_input: %s", model_input.debugString().c_str());
    auto         merged_output = std::make_unique<MergedOutput>();
    ModelRequest model_request = std::move(generateOldModelRequest(model_input));
    auto         model_output  = std::move(model_wrapper_->forward(model_request));
    (void)updateTargetProb(streams, *(model_output->logits));
    auto sampler_input_status = batch_stream_processor_->gatherSpeculativeSamplerInput(stream_groups, *model_output);
    RETURN_IF_STATUS_OR_ERROR(sampler_input_status);
    auto& sampler_input  = sampler_input_status.value();
    auto  sampler_output = std::move(sampler_->forward(sampler_input));
    return batch_stream_processor_->dispatch(stream_groups, *model_output, sampler_output);
}

}  // namespace rtp_llm
