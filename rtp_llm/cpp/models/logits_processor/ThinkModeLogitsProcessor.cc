#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"

using namespace std;

namespace rtp_llm {

namespace {

constexpr int32_t kInvalidTokenId           = -1;
constexpr int32_t kDeepSeekNewlineTokenId   = 201;
constexpr int32_t kDeepSeekBlankLineTokenId = 271;
constexpr int32_t kQwenGlmNewlineTokenId    = 198;

bool isBoundaryPaddingToken(int32_t token_id) {
    return token_id == kDeepSeekNewlineTokenId || token_id == kDeepSeekBlankLineTokenId
           || token_id == kQwenGlmNewlineTokenId;
}

std::vector<int> normalizeThinkEndTokenIds(const std::vector<int>& end_think_token_ids) {
    if (end_think_token_ids.size() <= 1) {
        return end_think_token_ids;
    }

    // Some templates include surrounding newlines around the semantic </think> token.
    size_t begin = 0;
    size_t end   = end_think_token_ids.size();
    while (begin + 1 < end && isBoundaryPaddingToken(end_think_token_ids[begin])) {
        ++begin;
    }
    while (begin + 1 < end && isBoundaryPaddingToken(end_think_token_ids[end - 1])) {
        --end;
    }
    return std::vector<int>(end_think_token_ids.begin() + begin, end_think_token_ids.begin() + end);
}

int32_t firstTokenOrInvalid(const std::vector<int>& token_ids) {
    if (token_ids.empty()) {
        return kInvalidTokenId;
    }
    return token_ids.front();
}

void maskToken(const torch::Tensor& new_tokens_logits, size_t vocab_size, int32_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size) {
        return;
    }
    new_tokens_logits[token_id] = BaseLogitsProcessor::neg_inf;
}

bool isInThinkState(const StreamThinkInfo& info) {
    return info.process_state == ThinkProcessState::IN_THINK;
}

bool transitionToAfterThinkIfClosed(StreamThinkInfo& info) {
    if (!info.dfa_ptr || !info.dfa_ptr->isFinished()) {
        return false;
    }
    info.process_state = ThinkProcessState::AFTER_THINK;
    return true;
}

int generatedTokens(const SamplerInputs& inputs, size_t batch_idx) {
    int* input_lengths    = inputs.input_lengths.data_ptr<int32_t>();
    int* sequence_lengths = inputs.sequence_lengths.data_ptr<int32_t>();
    return sequence_lengths[batch_idx] - input_lengths[batch_idx];
}

bool shouldForceThinkEnd(const SamplerInputs& inputs, size_t batch_idx, const StreamThinkInfo& info) {
    if (!info.dfa_ptr || info.end_think_token_ids.empty() || info.max_thinking_tokens <= 0) {
        return false;
    }

    bool close_in_progress = info.dfa_ptr->status() > 0;
    bool budget_exhausted  = generatedTokens(inputs, batch_idx) >= info.max_thinking_tokens;
    return close_in_progress || budget_exhausted;
}

void maskThinkBoundaryTokens(const torch::Tensor& new_tokens_logits, size_t vocab_size, const StreamThinkInfo& info) {
    maskToken(new_tokens_logits, vocab_size, firstTokenOrInvalid(info.begin_think_token_ids));
    maskToken(new_tokens_logits, vocab_size, firstTokenOrInvalid(info.end_think_token_ids));
}

}  // namespace

ThinkModeLogitsProcessor::ThinkModeLogitsProcessor(std::vector<StreamThinkInfo> think_infos):
    think_infos_(think_infos) {};

void ThinkModeLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    RTP_LLM_CHECK(size() == finish_idx - start_idx);

    for (size_t i = 0; i < size(); ++i) {
        auto&  info      = think_infos_[i];
        size_t batch_idx = i + start_idx;

        switch (info.process_state) {
            case ThinkProcessState::NO_THINK:
            case ThinkProcessState::AFTER_THINK: {
                maskThinkBoundaryTokens(inputs.logits[batch_idx], inputs.vocab_size, info);
                break;
            }
            case ThinkProcessState::IN_THINK: {
                if (transitionToAfterThinkIfClosed(info)) {
                    maskThinkBoundaryTokens(inputs.logits[batch_idx], inputs.vocab_size, info);
                    break;
                }

                if (shouldForceThinkEnd(inputs, batch_idx, info)) {
                    forceThinkEndToken(inputs.logits[batch_idx], info, inputs.vocab_size);
                    break;
                }

                maskToken(inputs.logits[batch_idx], inputs.vocab_size, firstTokenOrInvalid(info.begin_think_token_ids));
                break;
            }
        }
    }
}

bool ThinkModeLogitsProcessor::forceThinkEndToken(const torch::Tensor&   new_tokens_logits,
                                                  const StreamThinkInfo& info,
                                                  size_t                 vocab_size) {
    if (!info.dfa_ptr || info.dfa_ptr->isFinished() || info.end_think_token_ids.empty()) {
        return false;
    }
    auto next_token_idx = info.dfa_ptr->status();
    if (next_token_idx >= info.end_think_token_ids.size()) {
        return false;
    }

    RTP_LLM_LOG_INFO("sampler enforce think end token");
    memFill(new_tokens_logits, vocab_size, (size_t)info.end_think_token_ids[next_token_idx]);
    return true;
}

void ThinkModeLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    std::vector<StreamThinkInfo> new_think_infos;
    for (auto src_batch_idx : src_batch_indices) {
        new_think_infos.push_back(think_infos_[src_batch_idx].copy());
    }
    think_infos_ = new_think_infos;
}

void ThinkModeLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    RTP_LLM_CHECK(2 == new_tokens.dim());
    RTP_LLM_CHECK(size() == (size_t)new_tokens.size(0));

    for (size_t i = 0; i < size(); i++) {
        auto& info = think_infos_[i];
        if (!isInThinkState(info))
            continue;
        if (info.max_thinking_tokens <= 0 || !info.dfa_ptr)
            continue;
        if (transitionToAfterThinkIfClosed(info)) {
            continue;
        }

        auto offset = info.is_beam_search ? (info.current_output_length + info.input_length) : 0;

        if (!info.is_beam_search) {
            RTP_LLM_CHECK(num_new_tokens == new_tokens.size(1));
        }

        for (size_t j = 0; j < num_new_tokens; ++j) {
            auto current_token_id = new_tokens.data_ptr<int>()[i * new_tokens.size(1) + j + offset];
            info.dfa_ptr->next(current_token_id);
            if (info.dfa_ptr->isFinished()) {
                info.process_state = ThinkProcessState::AFTER_THINK;
                break;
            }
        }

        info.current_output_length += num_new_tokens;
    }
}

ThinkModeLogitsProcessorPtr ThinkModeLogitsProcessor::fromGenerateInput(std::shared_ptr<GenerateInput> generate_input,
                                                                        int32_t                        num) {
    auto generate_config         = generate_input->generate_config;
    auto end_think_token_ids     = normalizeThinkEndTokenIds(generate_config->end_think_token_ids);
    bool has_think_boundary_mask = !generate_config->begin_think_token_ids.empty() || !end_think_token_ids.empty();
    bool has_think_budget =
        generate_config->in_think_mode && generate_config->max_thinking_tokens > 0 && !end_think_token_ids.empty();
    if (!has_think_boundary_mask) {
        return nullptr;
    }

    auto processor_ptr = std::make_shared<ThinkModeLogitsProcessor>();
    for (size_t i = 0; i < num; i++) {
        std::shared_ptr<StringContainDFA<size_t, int>> dfa_ptr;
        if (has_think_budget) {
            dfa_ptr = std::make_shared<StringContainDFA<size_t, int>>(end_think_token_ids);
        }
        StreamThinkInfo              think_info(generate_config->in_think_mode,
                                   generate_config->max_thinking_tokens,
                                   generate_config->begin_think_token_ids,
                                   end_think_token_ids,
                                   generate_input->inputLength(),
                                   0,
                                   generate_config->hasNumBeams() || generate_config->num_return_sequences > 1,
                                   dfa_ptr);
        std::vector<StreamThinkInfo> think_infos = {think_info};
        auto                         ptr         = std::make_shared<ThinkModeLogitsProcessor>(think_infos);

        processor_ptr->insert(ptr, 1);
    }
    return processor_ptr;
}

std::vector<size_t> ThinkModeLogitsProcessor::thinkEndTokensStatus() {
    std::vector<size_t> status;
    for (auto think_info : think_infos_) {
        auto dfa = think_info.dfa_ptr;
        status.push_back(dfa ? dfa->status() : 0);
    }
    return status;
}

}  // namespace rtp_llm
