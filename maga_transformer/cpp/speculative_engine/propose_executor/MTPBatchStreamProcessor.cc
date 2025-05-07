#include <random>

#include "maga_transformer/cpp/speculative_engine/propose_executor/MTPBatchStreamProcessor.h"
#include "maga_transformer/cpp/utils/StatusUtil.h"
#include "maga_transformer/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "maga_transformer/cpp/core/Types.h"
#include "maga_transformer/cpp/utils/AssertUtils.h"
#include "maga_transformer/cpp/utils/Logger.h"
#include "maga_transformer/cpp/models/MTPModel.h"

using namespace std;

namespace rtp_llm {

absl::StatusOr<GptModelInputs> MTPBatchStreamProcessor::gatherModelInput(const StreamGroups& stream_groups) const {
    auto model_input = NormalBatchStreamProcessor::gatherModelInput(stream_groups);
    RTP_LLM_CHECK(model_input.ok());
    auto all_streams = stream_groups.allStreams();
    rtp_llm::DataType type = rtp_llm::DataType::TYPE_INVALID;
    size_t hidden_size = 0;
    // Here we need to check that all streams have hidden states
    // and the shape is aligned with the current number of tokens.
    size_t all_hidden_tokens_num = 0;
    for (auto& stream : all_streams) {
        auto hidden_states = stream->getLastHiddenStates();
        RTP_LLM_CHECK(hidden_states != nullptr);
        RTP_LLM_CHECK(hidden_states->dim() == 2);
        if (type == rtp_llm::DataType::TYPE_INVALID) {
            type = hidden_states->type();
        } else {
            // check all hidden states has same type
            RTP_LLM_CHECK(type == hidden_states->type());
        }
        if (hidden_size == 0) {
            hidden_size = hidden_states->shape()[1];
        } else {
            // check all hidden states has same shape[1]
            RTP_LLM_CHECK(hidden_size == hidden_states->shape()[1]);
        }
        RTP_LLM_CHECK_WITH_INFO(hidden_states->shape()[0] == stream->currentExecuteTokenSize(),
            "hidden states num: %d, token num: %d", hidden_states->shape()[0],
            stream->currentExecuteTokenSize());
        all_hidden_tokens_num += stream->currentExecuteTokenSize();
    }


    // Here, for the MTP model, we must ensure that the input contains the result of the last hidden states.
    // The second is to ensure that the first dimension of the token is aligned with the hidden states,
    // that is, each execution must be truncated
    auto all_hidden_states = device_->allocateBuffer({type,
                                                 {all_hidden_tokens_num, hidden_size},
                                                 rtp_llm::AllocationType::DEVICE}, {});
    size_t index = 0;
    for (auto& stream : all_streams) {
        auto hidden_states = stream->getLastHiddenStates();
        auto hidden_num = hidden_states->shape()[0];
        device_->copy({all_hidden_states->view(index, hidden_num),
                       *hidden_states});
        index += hidden_num;
    }
    model_input.value().last_hidden_states = all_hidden_states;
    return model_input;
};

}  // namespace rtp_llm
