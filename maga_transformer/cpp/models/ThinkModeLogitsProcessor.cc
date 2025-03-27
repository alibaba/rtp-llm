#include "maga_transformer/cpp/models/ThinkModeLogitsProcessor.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

ThinkModeLogitsProcessor::ThinkModeLogitsProcessor(
    ft::DeviceBase* device, std::deque<bool> think_modes, std::vector<int> max_thinking_tokens,
    std::vector<std::vector<int>> end_think_token_ids,
    std::vector<std::shared_ptr<StringContainDFA<size_t, int>>> think_status_dfa_ptrs) :
    BaseLogitsProcessor(device),
    think_modes_(think_modes), max_thinking_tokens_(max_thinking_tokens),
    end_think_token_ids_(end_think_token_ids), think_status_dfa_ptrs_(think_status_dfa_ptrs) {};

void ThinkModeLogitsProcessor::process(const SamplerInputs& inputs) {
    for (size_t i = 0; i < inputs.batch_size; i++) {
        if (think_modes_[i]) {
            int* input_lengths = inputs.input_lengths->data<int32_t>();
            int* sequence_lengths = inputs.sequence_lengths->data<int32_t>();
            int num_new_tokens = 1;
            bool enforce = (sequence_lengths[i] + num_new_tokens >= max_thinking_tokens_[i] + input_lengths[i]);
            auto logits = inputs.logits->index(i);
            setVocabMask(think_status_dfa_ptrs_[i], logits, num_new_tokens, end_think_token_ids_[i], inputs.vocab_size, enforce);
        }
    }
}

void ThinkModeLogitsProcessor::setVocabMask(
    std::shared_ptr<StringContainDFA<size_t, int>> dfa_ptr, 
    ft::BufferPtr new_tokens_logits, int num_new_tokens, 
    std::vector<int> template_token_ids, size_t vocab_size, bool enforce) 
{
    if (!dfa_ptr->isFinished() && enforce) {
        int offset = 0;
        for (size_t pos = dfa_ptr->status(); pos < template_token_ids.size() && offset < num_new_tokens; pos++, offset++) {
            FT_LOG_INFO("sampler enforce transfer status");
            memFill(new_tokens_logits, vocab_size, (size_t) template_token_ids[pos]);
        }
    }
}

void ThinkModeLogitsProcessor::updateStatus(const SamplerInputs& inputs) {
    for (size_t i = 0; i < inputs.batch_size; i++) {
        if (think_modes_[i]) {
            auto dfa_ptr = think_status_dfa_ptrs_[i];
            auto token_ids = inputs.token_ids->index(i);
            int num_new_tokens = 1;
            const size_t step = token_ids->shape()[0];
            for (size_t j = 0; j < num_new_tokens; ++j) {
                auto current_token_id = *(token_ids->dataWithOffset<int>(step - num_new_tokens + j));
                if (!dfa_ptr->isFinished()) {
                    dfa_ptr->next(current_token_id);
                }
            }
        }
    }
}

std::vector<size_t> ThinkModeLogitsProcessor::thinkEndTokensStatus() {
    std::vector<size_t> status;
    for (auto dfa: think_status_dfa_ptrs_) {
        status.push_back(dfa->status());
    }
    return status;
}

}