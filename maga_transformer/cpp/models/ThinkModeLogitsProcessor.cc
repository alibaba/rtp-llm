#include "maga_transformer/cpp/models/ThinkModeLogitsProcessor.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

ThinkModeLogitsProcessor::ThinkModeLogitsProcessor(
    ft::DeviceBase* device, std::vector<StreamThinkInfo> think_infos) :
    BaseLogitsProcessor(device), think_infos_(think_infos) {};

void ThinkModeLogitsProcessor::process(const SamplerInputs& inputs) {
    for (size_t i = 0; i < inputs.batch_size; i++) {
        if (think_infos_[i].in_think_mode) {
            int* input_lengths = inputs.input_lengths->data<int32_t>();
            int* sequence_lengths = inputs.sequence_lengths->data<int32_t>();
            int num_new_tokens = 1;
            auto dfa_ptr = think_infos_[i].think_end_status_dfa_ptr;
            bool enforce = (sequence_lengths[i] + num_new_tokens >= think_infos_[i].max_thinking_tokens + input_lengths[i]);
            auto logits = inputs.logits->index(i);
            setVocabMask(dfa_ptr, logits, num_new_tokens, think_infos_[i].end_think_token_ids, inputs.vocab_size, enforce);
        }
    }
}

void ThinkModeLogitsProcessor::setVocabMask(
    std::shared_ptr<StringContainDFA<size_t, int>> dfa_ptr, 
    ft::BufferPtr new_tokens_logits, int num_new_tokens, 
    std::vector<int> template_token_ids, size_t vocab_size, bool enforce) 
{
    if (!dfa_ptr->isFinished() && enforce) {
        FT_LOG_INFO("sampler enforce transfer status");
        memFill(new_tokens_logits, vocab_size, (size_t) template_token_ids[dfa_ptr->status()]);
    }
}

void ThinkModeLogitsProcessor::updateStatus(const SamplerInputs& inputs) {
    for (size_t i = 0; i < inputs.batch_size; i++) {
        if (think_infos_[i].in_think_mode) {
            auto dfa_ptr = think_infos_[i].think_end_status_dfa_ptr;
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
    for (auto think_info: think_infos_) {
        auto dfa = think_info.think_end_status_dfa_ptr;
        status.push_back(dfa->status());
    }
    return status;
}

}