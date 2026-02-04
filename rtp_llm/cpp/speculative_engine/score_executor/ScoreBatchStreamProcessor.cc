#include <random>

#include "rtp_llm/cpp/speculative_engine/score_executor/ScoreBatchStreamProcessor.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

using namespace std;

namespace rtp_llm {

absl::StatusOr<GptModelInputs> ScoreBatchStreamProcessor::gatherModelInput(const StreamGroups& stream_groups) const {
    CHECK_AND_RETURN_REF(model_input, NormalBatchStreamProcessor::gatherModelInput(stream_groups));
    size_t total_batch_size = stream_groups.totalScoreBatchSize();
    model_input.lm_output_indexes =
        device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {total_batch_size}, rtp_llm::AllocationType::HOST}, {});
    int* lm_output_indexes = (int*)model_input.lm_output_indexes->data();
    int* lm_output_lengths = (int*)model_input.lm_output_lengths->data();
    int  token_idx         = 0;
    int  batch_idx         = 0;
    auto context_streams   = stream_groups.contextStreams();
    auto decode_streams    = stream_groups.decodeStreams();
    for (const auto& stream : decode_streams) {
        auto current_batch_size = stream->scoreLen();
        for (auto i = 0; i < current_batch_size; ++i) {
            lm_output_indexes[token_idx] = token_idx;
            token_idx++;
        }
        lm_output_lengths[batch_idx] = current_batch_size;
        batch_idx++;
    }
    int cum_output_seq_len = token_idx;
    for (const auto& stream : context_streams) {
        auto current_batch_size = stream->scoreLen();
        auto input_tokens       = stream->currentExecuteTokens(0);
        cum_output_seq_len += input_tokens.size();
        for (auto i = 0; i < current_batch_size; ++i) {
            lm_output_indexes[token_idx] = cum_output_seq_len - current_batch_size + i;
            token_idx++;
        }
        lm_output_lengths[batch_idx] = current_batch_size;
        batch_idx++;
    }
    return model_input;
}

absl::StatusOr<SamplerInputs> ScoreBatchStreamProcessor::gatherSamplerInput(const StreamGroups&    stream_groups,
                                                                            const GptModelInputs&  model_inputs,
                                                                            const GptModelOutputs& model_output) const {
    RTP_LLM_CHECK(!stream_groups.empty());
    auto               all_streams      = stream_groups.allStreams();
    ReturnAllProbsMode return_all_probs = stream_groups.needReturnAllProbs();

    for (auto& stream : all_streams) {
        RTP_LLM_CHECK_WITH_INFO(stream->maxBatchSize() == 1, "stream tile num must be 1 in ScoreExecutor");
    }

    size_t        total_batch_size = stream_groups.totalScoreBatchSize();
    SamplerInputs sampler_inputs =
        allocateSamplerInputs(stream_groups, total_batch_size, total_batch_size, model_inputs.sequence_lengths);
    setCommonSamplerInputs(sampler_inputs, all_streams, true);

    int batch_idx = 0;
    for (auto& stream : all_streams) {
        const auto& complete_token_ids = stream->completeTokenIds();
        auto        seq_len            = stream->seqLength();
        auto        current_batch_size = stream->scoreLen();

        for (int i = 0; i < current_batch_size; ++i) {
            memcpy(sampler_inputs.token_ids->dataWithOffset<int32_t>((batch_idx) * (sampler_inputs.step + 1)),
                   complete_token_ids->dataWithOffset<int32_t>(0),
                   (seq_len - current_batch_size + i + 1) * sizeof(int));
            batch_idx += 1;
        }

        RTP_LLM_LOG_DEBUG("stream [%ld], complete token ids = [%s]",
                          stream->streamId(),
                          complete_token_ids->debugStringWithData<int32_t>(sampler_inputs.step).c_str());
        RTP_LLM_LOG_DEBUG("stream [%ld], sampler inputs token ids = [%s]",
                          stream->streamId(),
                          sampler_inputs.token_ids->debugStringWithData<int32_t>().c_str());
    }

    auto vocab_size       = model_output.logits->shape()[1];
    sampler_inputs.logits = device_->allocateBuffer(
        {model_output.logits->type(), {total_batch_size, vocab_size}, rtp_llm::AllocationType::DEVICE}, {});
    if (return_all_probs != ReturnAllProbsMode::NONE) {
        sampler_inputs.all_probs = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {total_batch_size, vocab_size}, rtp_llm::AllocationType::DEVICE}, {});
        device_->bufMemset(*sampler_inputs.all_probs, 0);
        if (return_all_probs == ReturnAllProbsMode::ORIGINAL) {
            sampler_inputs.return_original_all_probs = true;
        }
    }

    device_->copy({*sampler_inputs.logits, *model_output.logits});

    RTP_LLM_LOG_DEBUG("sampler inputs logits [%s]",
                      device_->clone({*sampler_inputs.logits, rtp_llm::AllocationType::HOST})
                          ->debugStringWithData<float>(10)
                          .c_str());

    RTP_LLM_LOG_DEBUG("gatherSamplerInput done");
    return std::move(sampler_inputs);
}

absl::Status ScoreBatchStreamProcessor::dispatch(const StreamGroups& stream_groups,
                                                 const MergedOutput& merge_outputs) const {
    const GptModelOutputs& model_output      = merge_outputs.model_output;
    const SamplerOutput&   sampler_output    = merge_outputs.sampler_output;
    const auto&            new_all_token_ids = sampler_output.token_ids;
    RTP_LLM_LOG_DEBUG("new_all_token_ids = [%s]", new_all_token_ids->debugStringWithData<int32_t>().c_str());
    const size_t step             = new_all_token_ids->shape()[1];
    size_t       batch_idx        = 0;
    bool         return_all_probs = stream_groups.needReturnAllProbs() != ReturnAllProbsMode::NONE;
    size_t       token_offset     = 0;
    for (auto& stream : stream_groups.allStreams()) {
        auto current_batch_size = stream->scoreLen();
        auto token_size         = stream->currentExecuteTokenSize();

        BufferPtr current_softmax_result;
        BufferPtr batch_logits = nullptr;
        if (stream->returnLogits() || stream->calculateSoftmaxProbs()) {
            batch_logits = model_output.logits->slice(batch_idx, current_batch_size);
        }
        // auto batch_softmax_result = model_output.softmax_result->slice(batch_idx, current_batch_size);
        BufferPtr batch_softmax_result;
        BufferPtr batch_hidden_states = nullptr;
        if (stream->generateConfig()->return_hidden_states) {
            batch_hidden_states = model_output.hidden_states->slice(batch_idx, current_batch_size);
        }
        BufferPtr all_probs = nullptr;
        if (return_all_probs) {
            all_probs = sampler_output.all_probs->slice(batch_idx, current_batch_size, false);
            all_probs->updateParent(sampler_output.all_probs);
        }

        BufferPtr loss = nullptr;
        if (stream->calculateLoss()) {
            auto all_logits          = model_output.all_logits->view(token_offset, token_size - current_batch_size);
            auto tokens              = stream->currentExecuteTokens(0);
            rtp_llm::BufferPtr label = device_->clone({{rtp_llm::MemoryType::MEMORY_CPU,
                                                        rtp_llm::DataType::TYPE_INT32,
                                                        {tokens.size() - current_batch_size},
                                                        tokens.data() + 1}});
            loss                     = device_->loss({all_logits, *label});
        }

        BufferPtr all_hidden_states = nullptr;
        if (stream->needReturnHiddenStates()) {
            all_hidden_states = model_output.all_hidden_states->slice(token_offset, token_size, false);
            all_hidden_states->updateParent(model_output.all_hidden_states);
        }
        rtp_llm::BufferPtr new_tokens = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {(size_t)1, (size_t)current_batch_size}, rtp_llm::AllocationType::HOST},
            {});
        if (stream->calculateSoftmaxProbs()) {
            current_softmax_result = device_->allocateBuffer(
                {rtp_llm::DataType::TYPE_FP32, {(size_t)1, (size_t)current_batch_size}, rtp_llm::AllocationType::HOST},
                {});
            batch_softmax_result =
                device_->softmax({batch_logits, std::nullopt, std::nullopt, 1.0f, DataType::TYPE_FP32, std::nullopt});
        }
        for (int i = 0; i < current_batch_size; ++i) {
            memcpy(new_tokens->dataWithOffset<int32_t>(i),
                   new_all_token_ids->dataWithOffset<int32_t>(batch_idx * step + step - 1),
                   sizeof(int32_t));
            if (stream->calculateSoftmaxProbs()) {
                device_->copy({(*current_softmax_result)[0].view(i, 1),
                               (*batch_softmax_result)[i].view(*(new_tokens->dataWithOffset<int32_t>(i)), 1)});
            }
            batch_idx += 1;
        }
        RTP_LLM_LOG_DEBUG(
            "stream [%ld], new_tokens = [%s]", stream->streamId(), new_tokens->debugStringWithData<int32_t>().c_str());
        auto update_info = StreamUpdateInfo{new_tokens,
                                            1,
                                            batch_hidden_states,
                                            batch_logits,
                                            current_softmax_result,
                                            nullptr,
                                            all_probs,
                                            loss,
                                            nullptr,
                                            all_hidden_states};
        stream->updateOutput(update_info);
        token_offset += token_size;
    }
    RTP_LLM_LOG_DEBUG("dispatch done");
    return absl::OkStatus();
}

}  // namespace rtp_llm
