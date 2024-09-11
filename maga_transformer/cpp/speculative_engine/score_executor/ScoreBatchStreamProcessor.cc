#include <random>

#include "maga_transformer/cpp/speculative_engine/score_executor/ScoreBatchStreamProcessor.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/utils/assert_utils.h"
#include "src/fastertransformer/utils/logger.h"

using namespace std;
using namespace fastertransformer;
namespace rtp_llm {

absl::StatusOr<GptModelInputs> ScoreBatchStreamProcessor::gatherModelInput(const StreamGroups& stream_groups) const {
    CHECK_AND_RETURN_REF(model_input, NormalBatchStreamProcessor::gatherModelInput(stream_groups));
    model_input.need_all_logits = true;
    return model_input;
}


absl::StatusOr<SamplerInputs>
ScoreBatchStreamProcessor::gatherSamplerInput(const StreamGroups&    stream_groups,
                                               const GptModelInputs&  model_inputs,
                                               const GptModelOutputs& model_output) const {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(!stream_groups.empty());
    auto all_streams = stream_groups.allStreams();
    bool return_all_probs = stream_groups.needReturnAllProbs();

    for (auto &stream : all_streams) {
        FT_CHECK_WITH_INFO(stream->tileNum() == 1, "stream tile num must be 1 in ScoreExecutor");
    }

    const auto& context_streams = stream_groups.contextStreams();
    size_t total_decode_batch_size = stream_groups.totalDecodeBatchSize();
    size_t total_batch_size = stream_groups.totalScoreBatchSize();

    SamplerInputs sampler_inputs = allocateSamplerInputs(stream_groups, total_batch_size, model_inputs.sequence_lengths);
    setCommonSamplerInputs(sampler_inputs, all_streams, true);
 
    int batch_idx   = 0;
    for (auto& stream : all_streams) {
        const auto& complete_token_ids = stream->completeTokenIds();
        auto        seq_len            = stream->seqLength();
        auto        current_batch_size = stream->scoreLen();

        for (int i = 0; i < current_batch_size; ++i) {
            memcpy(sampler_inputs.token_ids->dataWithOffset<int32_t>((batch_idx) * (sampler_inputs.step + 1)),
                   complete_token_ids->dataWithOffset<int32_t>(0),
                   (seq_len - current_batch_size + i) * sizeof(int));
            batch_idx += 1;
        }

        FT_LOG_DEBUG("stream [%d], complete token ids = [%s]", stream->streamId(), complete_token_ids->debugStringWithData<int32_t>(sampler_inputs.step).c_str());
        FT_LOG_DEBUG("stream [%d], sampler inputs token ids = [%s]", stream->streamId(), sampler_inputs.token_ids->debugStringWithData<int32_t>().c_str());
    }

    auto vocab_size = model_output.logits->shape()[1];
    sampler_inputs.logits = device_->allocateBuffer({model_output.logits->type(), {total_batch_size, vocab_size}, ft::AllocationType::DEVICE}, {});
    if (return_all_probs) {
        sampler_inputs.all_probs = device_->allocateBuffer({ft::DataType::TYPE_FP32, {total_batch_size, vocab_size}, ft::AllocationType::DEVICE}, {});
        device_->bufMemset(*sampler_inputs.all_probs, 0);
    }

    batch_idx = 0;
    device_->copy({sampler_inputs.logits->view(0, total_decode_batch_size), model_output.logits->view(0, total_decode_batch_size)});
    batch_idx += total_decode_batch_size;
    size_t offset = 0;
    for (auto& stream : context_streams) {
        size_t  current_batch_size = stream->scoreLen();
        offset += stream->contextLength();
        for (int i = 0; i < current_batch_size; ++i) {
            device_->copy({sampler_inputs.logits->view(batch_idx, 1), model_output.all_logits->view(offset - current_batch_size + i - 1, 1)});
            batch_idx += 1;
        }
    }

    FT_LOG_DEBUG("sampler inputs logits [%s]",
                device_->clone({*sampler_inputs.logits, ft::AllocationType::HOST})->debugStringWithData<float>(10).c_str());

    FT_LOG_DEBUG("gatherSamplerInput done");
    return std::move(sampler_inputs);
}

absl::Status ScoreBatchStreamProcessor::dispatch(const StreamGroups&                  stream_groups,
                                                  const MergedOutput& merge_outputs) const {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const auto& model_output      = merge_outputs.model_output;
    const auto& sampler_output    = merge_outputs.sampler_output;
    const auto& new_all_token_ids = sampler_output.token_ids;
    FT_LOG_DEBUG("new_all_token_ids = [%s]", new_all_token_ids->debugStringWithData<int32_t>().c_str());
    const size_t step = new_all_token_ids->shape()[1];
    size_t batch_idx = 0;
    size_t offset = 0;
    bool return_all_probs = stream_groups.needReturnAllProbs();
    for (auto& stream : stream_groups.allStreams()) {
        auto current_batch_size = stream->scoreLen();
        
        // TODO(xyz): specutial handle like gathersamplerInput
        auto batch_logits = model_output.all_logits->slice(offset, current_batch_size);
        auto batch_hidden_states = model_output.all_hidden_states->slice(offset, current_batch_size);
        auto all_probs = return_all_probs ? sampler_output.all_probs->slice(batch_idx, current_batch_size) : nullptr;

        ft::BufferPtr new_tokens = device_->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)1, (size_t)current_batch_size}, ft::AllocationType::HOST}, {});
        for (int i = 0; i < current_batch_size; ++i) {
            memcpy(new_tokens->dataWithOffset<int32_t>(i), new_all_token_ids->dataWithOffset<int32_t>(batch_idx * step + step - 1), sizeof(int32_t));
            batch_idx += 1;
        }
        FT_LOG_DEBUG("stream [%d], new_tokens = [%s]", stream->streamId(), new_tokens->debugStringWithData<int32_t>().c_str());
        stream->updateOutput(new_tokens, batch_hidden_states, batch_logits, nullptr, all_probs);
        offset += stream->contextLength();
    }
    FT_LOG_DEBUG("dispatch done");
    return absl::OkStatus();
}

}  // namespace rtp_llm
