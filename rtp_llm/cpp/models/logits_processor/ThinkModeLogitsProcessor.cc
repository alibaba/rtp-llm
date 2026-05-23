#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"

using namespace std;

namespace rtp_llm {

namespace {

constexpr int32_t kInvalidTokenId         = -1;
constexpr int32_t kDeepSeekNewlineTokenId = 201;
constexpr int32_t kQwenGlmNewlineTokenId  = 198;

int32_t inferThinkEndTokenId(const std::vector<int>& end_think_token_ids) {
    if (end_think_token_ids.empty()) {
        return kInvalidTokenId;
    }
    if (end_think_token_ids.size() > 1
        && (end_think_token_ids.front() == kDeepSeekNewlineTokenId
            || end_think_token_ids.front() == kQwenGlmNewlineTokenId)) {
        return end_think_token_ids[1];
    }
    return end_think_token_ids.front();
}

std::vector<int> effectiveThinkEndTokenIds(const std::vector<int>& end_think_token_ids) {
    if (end_think_token_ids.size() > 1
        && (end_think_token_ids.front() == kDeepSeekNewlineTokenId
            || end_think_token_ids.front() == kQwenGlmNewlineTokenId)) {
        return std::vector<int>(end_think_token_ids.begin() + 1, end_think_token_ids.end());
    }
    return end_think_token_ids;
}

int32_t inferThinkBeginTokenId(const std::vector<int>& begin_think_token_ids) {
    if (begin_think_token_ids.empty()) {
        return kInvalidTokenId;
    }
    return begin_think_token_ids.front();
}

void maskToken(const torch::Tensor& new_tokens_logits, size_t vocab_size, int32_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size) {
        return;
    }
    new_tokens_logits[token_id] = BaseLogitsProcessor::neg_inf;
}

bool inThinkProcess(const StreamThinkInfo& info) {
    return info.in_think_mode && info.process_state == ThinkProcessState::IN_THINK;
}

}  // namespace

ThinkModeLogitsProcessor::ThinkModeLogitsProcessor(std::vector<StreamThinkInfo> think_infos):
    think_infos_(think_infos) {};

void ThinkModeLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    RTP_LLM_CHECK(size() == finish_idx - start_idx);

    for (size_t i = 0; i < size(); ++i) {
        auto& info = think_infos_[i];

        if (inThinkProcess(info) && info.dfa_ptr && info.dfa_ptr->isFinished()) {
            info.process_state = ThinkProcessState::AFTER_THINK;
        }

        if (inThinkProcess(info) && info.max_thinking_tokens > 0 && info.dfa_ptr && !info.end_think_token_ids.empty()) {
            if (info.dfa_ptr->status() > 0 && !info.dfa_ptr->isFinished()) {
                setVocabMask(
                    info.dfa_ptr, inputs.logits[i + start_idx], 1, info.end_think_token_ids, inputs.vocab_size, true);
                continue;
            }
            int* input_lengths    = inputs.input_lengths.data_ptr<int32_t>();
            int* sequence_lengths = inputs.sequence_lengths.data_ptr<int32_t>();
            int  num_new_tokens   = 1;
            int  generated_tokens = sequence_lengths[i + start_idx] - input_lengths[i + start_idx];
            int  tokens_to_close  = static_cast<int>(info.end_think_token_ids.size() - info.dfa_ptr->status());
            bool enforce          = generated_tokens + tokens_to_close >= info.max_thinking_tokens;
            if (enforce && !info.dfa_ptr->isFinished()) {
                setVocabMask(info.dfa_ptr,
                             inputs.logits[i + start_idx],
                             num_new_tokens,
                             info.end_think_token_ids,
                             inputs.vocab_size,
                             enforce);
                continue;
            }
        }

        maskToken(inputs.logits[i + start_idx], inputs.vocab_size, inferThinkBeginTokenId(info.begin_think_token_ids));
        if (!inThinkProcess(info)) {
            maskToken(inputs.logits[i + start_idx], inputs.vocab_size, inferThinkEndTokenId(info.end_think_token_ids));
        }
    }
}

void ThinkModeLogitsProcessor::setVocabMask(std::shared_ptr<StringContainDFA<size_t, int>> dfa_ptr,
                                            const torch::Tensor&                           new_tokens_logits,
                                            int                                            num_new_tokens,
                                            std::vector<int>                               template_token_ids,
                                            size_t                                         vocab_size,
                                            bool                                           enforce) {
    if (dfa_ptr && !dfa_ptr->isFinished() && enforce && !template_token_ids.empty()) {
        RTP_LLM_LOG_INFO("sampler enforce transfer status");
        memFill(new_tokens_logits, vocab_size, (size_t)template_token_ids[dfa_ptr->status()]);
    }
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
        if (!inThinkProcess(info))
            continue;
        if (info.max_thinking_tokens <= 0 || !info.dfa_ptr)
            continue;

        auto offset = info.is_beam_search ? (info.current_output_length + info.input_length) : 0;

        if (!info.is_beam_search) {
            RTP_LLM_CHECK(num_new_tokens == new_tokens.size(1));
        }

        for (size_t j = 0; j < num_new_tokens; ++j) {
            auto current_token_id = new_tokens.data_ptr<int>()[i * new_tokens.size(1) + j + offset];
            info.dfa_ptr->next(current_token_id);
        }

        info.current_output_length += num_new_tokens;
        if (info.dfa_ptr->isFinished()) {
            info.process_state = ThinkProcessState::AFTER_THINK;
        }
    }
}

ThinkModeLogitsProcessorPtr ThinkModeLogitsProcessor::fromGenerateInput(std::shared_ptr<GenerateInput> generate_input,
                                                                        int32_t                        num) {
    auto generate_config         = generate_input->generate_config;
    auto end_think_token_ids     = effectiveThinkEndTokenIds(generate_config->end_think_token_ids);
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
